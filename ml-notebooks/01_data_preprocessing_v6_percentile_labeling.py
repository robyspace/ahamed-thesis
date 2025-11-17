#!/usr/bin/env python3
"""
Data Preprocessing with Percentile-Based Labeling (v6)
Hybrid Serverless-Container Thesis - Phase 3: ML Model Development

ISSUE: v4/v5 individual request labeling creates random variance
SOLUTION: Use percentile-based labeling to create learnable patterns

Strategy:
- For each workload, calculate performance percentiles (10th, 25th, 50th, 75th, 90th)
- Label requests based on which percentile bucket they fall into
- This creates smoother, more learnable decision boundaries

Example for lightweight_api:
- Fast Lambda requests (p10): Clearly better → Label as Lambda
- Slow Lambda requests (p90): Clearly worse → Label as ECS
- Medium requests (p50): Compare to ECS → Use margin-based decision
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# AWS Pricing Constants (eu-west-1)
LAMBDA_REQUEST_COST = 0.20 / 1_000_000
LAMBDA_DURATION_COST_PER_GB_SEC = 0.0000166667
ECS_VCPU_COST_PER_HOUR = 0.04656
ECS_MEMORY_COST_PER_GB_HOUR = 0.00511

ECS_TASK_CONFIGS = {
    'lightweight_api': {'vcpu': 0.25, 'memory_gb': 0.5},
    'thumbnail_processing': {'vcpu': 0.5, 'memory_gb': 1.0},
    'medium_processing': {'vcpu': 1.0, 'memory_gb': 2.0},
    'heavy_processing': {'vcpu': 1.0, 'memory_gb': 2.0}
}

LAMBDA_MEMORY_CONFIGS = {
    'lightweight_api': 128,
    'thumbnail_processing': 512,
    'medium_processing': 1024,
    'heavy_processing': 2048
}

# Decision margins (10% threshold)
LATENCY_MARGIN = 0.10  # Lambda must be 10% faster
COST_MARGIN = 0.10     # Lambda must be 10% cheaper

def calculate_lambda_cost(row):
    try:
        memory_mb = float(row['memory_used_mb'])
        execution_time_ms = float(row['execution_time_ms'])
        memory_gb = memory_mb / 1024
        execution_time_sec = execution_time_ms / 1000
        duration_cost = memory_gb * execution_time_sec * LAMBDA_DURATION_COST_PER_GB_SEC
        return duration_cost + LAMBDA_REQUEST_COST
    except:
        return np.nan

def calculate_ecs_cost(row):
    try:
        workload_type = row['workload_type']
        execution_time_ms = float(row['execution_time_ms'])
        config = ECS_TASK_CONFIGS[workload_type]
        execution_time_hours = execution_time_ms / 1000 / 3600
        cpu_cost = config['vcpu'] * ECS_VCPU_COST_PER_HOUR * execution_time_hours
        memory_cost = config['memory_gb'] * ECS_MEMORY_COST_PER_GB_HOUR * execution_time_hours
        return cpu_cost + memory_cost
    except:
        return np.nan

def load_jsonl_files(data_dir='data-output', date_filter=None):
    data_path = Path(data_dir)
    all_data = []

    for file_path in sorted(data_path.glob('*.jsonl')):
        if 'undefined' in file_path.name:
            continue
        if date_filter and not any(date in file_path.name for date in date_filter):
            continue

        print(f"📂 Loading: {file_path.name}")
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    all_data.append(json.loads(line.strip()))
                except:
                    continue

    df = pd.DataFrame(all_data)
    print(f"✅ Loaded {len(df):,} total requests\n")
    return df

def normalize_metrics(df):
    metrics_df = pd.json_normalize(df['metrics'])
    metrics_df.columns = ['metric_' + col for col in metrics_df.columns]
    df = df.drop('metrics', axis=1)
    df = pd.concat([df, metrics_df], axis=1)
    return df

def engineer_basic_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    workload_encoding = {
        'lightweight_api': 0,
        'thumbnail_processing': 1,
        'medium_processing': 2,
        'heavy_processing': 3
    }
    df['workload_type_encoded'] = df['workload_type'].map(workload_encoding)
    df['platform_encoded'] = (df['platform'] == 'lambda').astype(int)

    if 'metric_cold_start' in df.columns:
        df['cold_start'] = df['metric_cold_start'].fillna(False).astype(int)
    else:
        df['cold_start'] = 0

    return df

def engineer_advanced_features(df):
    print("🔧 Engineering advanced features...")

    df['lambda_memory_limit_mb'] = df['workload_type'].map(LAMBDA_MEMORY_CONFIGS)
    if 'metric_memory_limit_mb' in df.columns:
        df['lambda_memory_limit_mb'] = df['lambda_memory_limit_mb'].fillna(
            df['metric_memory_limit_mb'].astype(float)
        )

    df['payload_size_kb'] = df.get('metric_payload_size_kb', 0)
    df['payload_squared'] = df['payload_size_kb'] ** 2
    df['payload_log'] = np.log1p(df['payload_size_kb'])

    payload_quartiles = df['payload_size_kb'].quantile([0.25, 0.5, 0.75]).tolist()
    df['payload_category'] = pd.cut(
        df['payload_size_kb'],
        bins=[0] + payload_quartiles + [df['payload_size_kb'].max() + 1],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)

    def categorize_hour(hour):
        if 0 <= hour < 6:
            return 0
        elif 6 <= hour < 12:
            return 1
        elif 12 <= hour < 18:
            return 2
        else:
            return 3

    df['hour_period'] = df['hour_of_day'].apply(categorize_hour)
    df['workload_payload_interaction'] = df['workload_type_encoded'] * df['payload_size_kb']
    df['workload_memory_interaction'] = df['workload_type_encoded'] * df['lambda_memory_limit_mb']
    df['memory_payload_interaction'] = df['lambda_memory_limit_mb'] * df['payload_size_kb']
    df['workload_hour_interaction'] = df['workload_type_encoded'] * df['hour_of_day']
    df['payload_memory_ratio'] = df['payload_size_kb'] / (df['lambda_memory_limit_mb'] + 1)

    print(f"✅ Added 11 advanced features")
    return df

def add_cost_calculations(df):
    print("💰 Calculating costs...")

    lambda_df = df[df['platform'] == 'lambda'].copy()
    if len(lambda_df) > 0:
        lambda_df['cost_usd'] = lambda_df.apply(
            lambda row: calculate_lambda_cost({
                'memory_used_mb': row.get('metric_memory_used_mb', 128),
                'execution_time_ms': row.get('metric_execution_time_ms', 0)
            }), axis=1
        )
        print(f"   Lambda: {len(lambda_df):,} requests")

    ecs_df = df[df['platform'] == 'ecs'].copy()
    if len(ecs_df) > 0:
        ecs_df['cost_usd'] = ecs_df.apply(
            lambda row: calculate_ecs_cost({
                'workload_type': row.get('workload_type', 'lightweight_api'),
                'execution_time_ms': row.get('metric_execution_time_ms', 0)
            }), axis=1
        )
        print(f"   ECS: {len(ecs_df):,} requests")

    df = pd.concat([lambda_df, ecs_df], axis=0)
    df = df.dropna(subset=['cost_usd'])
    print(f"✅ Total: {len(df):,} requests\n")
    return df

def create_percentile_labels(df):
    """
    NEW v6: Percentile-based labeling with decision margins

    Instead of comparing each request to median, use percentile buckets:
    - Compare Lambda p25 vs ECS p75 (conservative estimate)
    - Use 10% margins to avoid labeling noise
    - This creates smoother, more learnable patterns
    """
    print("🏷️  Creating PERCENTILE-based labels (v6)...")

    paired_data = []

    for workload in df['workload_type'].unique():
        workload_df = df[df['workload_type'] == workload]
        lambda_df = workload_df[workload_df['platform'] == 'lambda']
        ecs_df = workload_df[workload_df['platform'] == 'ecs']

        print(f"\n   📊 {workload}:")
        print(f"      Lambda: {len(lambda_df):,} | ECS: {len(ecs_df):,}")

        # Calculate percentiles for both platforms
        lambda_lat_p25 = lambda_df['metric_execution_time_ms'].quantile(0.25)
        lambda_lat_p50 = lambda_df['metric_execution_time_ms'].quantile(0.50)
        lambda_lat_p75 = lambda_df['metric_execution_time_ms'].quantile(0.75)

        lambda_cost_p25 = lambda_df['cost_usd'].quantile(0.25)
        lambda_cost_p50 = lambda_df['cost_usd'].quantile(0.50)
        lambda_cost_p75 = lambda_df['cost_usd'].quantile(0.75)

        ecs_lat_p25 = ecs_df['metric_execution_time_ms'].quantile(0.25)
        ecs_lat_p50 = ecs_df['metric_execution_time_ms'].quantile(0.50)
        ecs_lat_p75 = ecs_df['metric_execution_time_ms'].quantile(0.75)

        ecs_cost_p25 = ecs_df['cost_usd'].quantile(0.25)
        ecs_cost_p50 = ecs_df['cost_usd'].quantile(0.50)
        ecs_cost_p75 = ecs_df['cost_usd'].quantile(0.75)

        print(f"      Lambda latency: p25={lambda_lat_p25:.1f}, p50={lambda_lat_p50:.1f}, p75={lambda_lat_p75:.1f}ms")
        print(f"      ECS latency:    p25={ecs_lat_p25:.1f}, p50={ecs_lat_p50:.1f}, p75={ecs_lat_p75:.1f}ms")
        print(f"      Lambda cost:    p25=${lambda_cost_p25:.10f}, p50=${lambda_cost_p50:.10f}")
        print(f"      ECS cost:       p25=${ecs_cost_p25:.10f}, p50=${ecs_cost_p50:.10f}")

        # For each request, determine percentile bucket and label accordingly
        for _, row in lambda_df.iterrows():
            actual_lat = row['metric_execution_time_ms']
            actual_cost = row['cost_usd']

            # Determine which percentile bucket this request falls into
            if actual_lat <= lambda_lat_p25:
                # Fast Lambda (p0-p25) vs typical ECS (p50)
                compare_ecs_lat = ecs_lat_p50
            elif actual_lat <= lambda_lat_p50:
                # Typical Lambda (p25-p50) vs typical ECS (p50)
                compare_ecs_lat = ecs_lat_p50
            elif actual_lat <= lambda_lat_p75:
                # Slow Lambda (p50-p75) vs fast ECS (p25)
                compare_ecs_lat = ecs_lat_p25
            else:
                # Very slow Lambda (p75-p100) vs fast ECS (p25)
                compare_ecs_lat = ecs_lat_p25

            if actual_cost <= lambda_cost_p25:
                compare_ecs_cost = ecs_cost_p50
            elif actual_cost <= lambda_cost_p50:
                compare_ecs_cost = ecs_cost_p50
            elif actual_cost <= lambda_cost_p75:
                compare_ecs_cost = ecs_cost_p25
            else:
                compare_ecs_cost = ecs_cost_p25

            # Apply decision margins (Lambda must be SIGNIFICANTLY better)
            lambda_faster = actual_lat < (compare_ecs_lat * (1 - LATENCY_MARGIN))
            lambda_cheaper = actual_cost < (compare_ecs_cost * (1 - COST_MARGIN))

            cost_optimal = 1 if lambda_cheaper else 0
            latency_optimal = 1 if lambda_faster else 0
            balanced_optimal = 1 if (lambda_faster and lambda_cheaper) else 0

            paired_data.append({
                'workload_type': workload,
                'workload_type_encoded': row['workload_type_encoded'],
                'payload_size_kb': row['payload_size_kb'],
                'hour_of_day': row['hour_of_day'],
                'day_of_week': row['day_of_week'],
                'is_weekend': row['is_weekend'],
                'lambda_memory_limit_mb': row['lambda_memory_limit_mb'],
                'payload_squared': row['payload_squared'],
                'payload_log': row['payload_log'],
                'payload_category': row['payload_category'],
                'hour_period': row['hour_period'],
                'workload_payload_interaction': row['workload_payload_interaction'],
                'workload_memory_interaction': row['workload_memory_interaction'],
                'memory_payload_interaction': row['memory_payload_interaction'],
                'workload_hour_interaction': row['workload_hour_interaction'],
                'payload_memory_ratio': row['payload_memory_ratio'],
                'lambda_latency_ms': actual_lat,
                'lambda_cost_usd': actual_cost,
                'lambda_memory_mb': row.get('metric_memory_used_mb', 128),
                'lambda_cold_start': row.get('cold_start', 0),
                'ecs_latency_ms': compare_ecs_lat,
                'ecs_cost_usd': compare_ecs_cost,
                'cost_optimal': cost_optimal,
                'latency_optimal': latency_optimal,
                'balanced_optimal': balanced_optimal,
                'optimal_platform': balanced_optimal
            })

        # Similar logic for ECS requests
        for _, row in ecs_df.iterrows():
            actual_lat = row['metric_execution_time_ms']
            actual_cost = row['cost_usd']

            if actual_lat <= ecs_lat_p25:
                compare_lambda_lat = lambda_lat_p75
            elif actual_lat <= ecs_lat_p50:
                compare_lambda_lat = lambda_lat_p50
            elif actual_lat <= ecs_lat_p75:
                compare_lambda_lat = lambda_lat_p25
            else:
                compare_lambda_lat = lambda_lat_p25

            if actual_cost <= ecs_cost_p25:
                compare_lambda_cost = lambda_cost_p75
            elif actual_cost <= ecs_cost_p50:
                compare_lambda_cost = lambda_cost_p50
            elif actual_cost <= ecs_cost_p75:
                compare_lambda_cost = lambda_cost_p25
            else:
                compare_lambda_cost = lambda_cost_p25

            lambda_faster = compare_lambda_lat < (actual_lat * (1 - LATENCY_MARGIN))
            lambda_cheaper = compare_lambda_cost < (actual_cost * (1 - COST_MARGIN))

            cost_optimal = 1 if lambda_cheaper else 0
            latency_optimal = 1 if lambda_faster else 0
            balanced_optimal = 1 if (lambda_faster and lambda_cheaper) else 0

            paired_data.append({
                'workload_type': workload,
                'workload_type_encoded': row['workload_type_encoded'],
                'payload_size_kb': row['payload_size_kb'],
                'hour_of_day': row['hour_of_day'],
                'day_of_week': row['day_of_week'],
                'is_weekend': row['is_weekend'],
                'lambda_memory_limit_mb': row['lambda_memory_limit_mb'],
                'payload_squared': row['payload_squared'],
                'payload_log': row['payload_log'],
                'payload_category': row['payload_category'],
                'hour_period': row['hour_period'],
                'workload_payload_interaction': row['workload_payload_interaction'],
                'workload_memory_interaction': row['workload_memory_interaction'],
                'memory_payload_interaction': row['memory_payload_interaction'],
                'workload_hour_interaction': row['workload_hour_interaction'],
                'payload_memory_ratio': row['payload_memory_ratio'],
                'lambda_latency_ms': compare_lambda_lat,
                'lambda_cost_usd': compare_lambda_cost,
                'lambda_memory_mb': LAMBDA_MEMORY_CONFIGS[workload],
                'lambda_cold_start': lambda_df['cold_start'].mean() if 'cold_start' in lambda_df.columns else 0,
                'ecs_latency_ms': actual_lat,
                'ecs_cost_usd': actual_cost,
                'cost_optimal': cost_optimal,
                'latency_optimal': latency_optimal,
                'balanced_optimal': balanced_optimal,
                'optimal_platform': balanced_optimal
            })

    paired_df = pd.DataFrame(paired_data)

    print(f"\n✅ Created {len(paired_df):,} samples with percentile-based labels")
    print(f"\n   📊 Label distributions:")

    for label_col in ['cost_optimal', 'latency_optimal', 'balanced_optimal']:
        counts = paired_df[label_col].value_counts()
        print(f"      {label_col}:")
        print(f"        Lambda (1): {counts.get(1, 0):,} ({counts.get(1, 0) / len(paired_df) * 100:.1f}%)")
        print(f"        ECS (0):    {counts.get(0, 0):,} ({counts.get(0, 0) / len(paired_df) * 100:.1f}%)")

    print(f"\n   📊 Label variance by workload (balanced_optimal):")
    variance = paired_df.groupby('workload_type')['balanced_optimal'].agg(['mean', 'std', 'nunique'])
    print(variance)

    return paired_df

def main():
    print("=" * 80)
    print("HYBRID THESIS - PERCENTILE-BASED LABELING (v6)")
    print("=" * 80)
    print("GOAL: Create learnable patterns by using percentile comparisons")
    print("      with 10% decision margins")
    print("=" * 80)
    print()

    DATA_DIR = 'data-output'
    OUTPUT_DIR = 'ml-notebooks/processed-data'
    DATE_FILTER = ['2025-11-16', '2025-11-17']

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    df = load_jsonl_files(DATA_DIR, date_filter=DATE_FILTER)
    df = normalize_metrics(df)
    df = engineer_basic_features(df)
    df = engineer_advanced_features(df)
    df = add_cost_calculations(df)
    paired_df = create_percentile_labels(df)

    paired_output_path = f"{OUTPUT_DIR}/ml_training_data_v6_percentile.csv"
    paired_df.to_csv(paired_output_path, index=False)
    print(f"\n💾 Saved: {paired_output_path}")

    feature_columns = [
        'workload_type_encoded', 'payload_size_kb', 'hour_of_day', 'day_of_week', 'is_weekend',
        'lambda_memory_limit_mb', 'payload_squared', 'payload_log', 'payload_category', 'hour_period',
        'workload_payload_interaction', 'workload_memory_interaction', 'memory_payload_interaction',
        'workload_hour_interaction', 'payload_memory_ratio',
    ]

    summary = {
        'total_requests': len(df),
        'total_training_samples': len(paired_df),
        'date_range': DATE_FILTER,
        'num_features': len(feature_columns),
        'feature_columns': feature_columns,
        'labeling_method': 'percentile_based_with_margins',
        'latency_margin': LATENCY_MARGIN,
        'cost_margin': COST_MARGIN,
        'label_distribution_balanced': paired_df['balanced_optimal'].value_counts().to_dict(),
    }

    summary_path = f"{OUTPUT_DIR}/preprocessing_summary_v6.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved: {summary_path}")

    print("\n" + "=" * 80)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\n🎯 Upload to Colab: {paired_output_path}")
    print(f"\n🚀 Expected improvements:")
    print(f"   - Percentile-based comparisons reduce noise")
    print(f"   - 10% decision margins create clearer boundaries")
    print(f"   - Models should learn patterns beyond simple lookup table")
    print(f"   - Target: 85%+ accuracy with actual feature usage")

if __name__ == '__main__':
    main()
