#!/usr/bin/env python3
"""
Data Preprocessing with Advanced Feature Engineering (v5)
Hybrid Serverless-Container Thesis - Phase 3: ML Model Development

TARGET: Improve accuracy from 77.6% to 85%+

IMPROVEMENTS:
1. Add memory_limit_mb (Lambda configuration - known at prediction time)
2. Interaction features (workload × payload, memory × payload)
3. Polynomial features (payload², log(payload))
4. Categorical binning (payload categories, hour periods)

These features help models learn non-linear decision boundaries and feature interactions.
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

# Standard Lambda configurations per workload (known at prediction time)
LAMBDA_MEMORY_CONFIGS = {
    'lightweight_api': 128,
    'thumbnail_processing': 512,
    'medium_processing': 1024,
    'heavy_processing': 2048
}

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
    """Basic features from v4"""
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
    """
    NEW: Advanced feature engineering for improved model performance

    All features added here ARE available at prediction time!
    """
    print("🔧 Engineering advanced features...")

    # 1. Lambda memory configuration (known at prediction time)
    df['lambda_memory_limit_mb'] = df['workload_type'].map(LAMBDA_MEMORY_CONFIGS)

    # For ECS requests, use the standard Lambda config for that workload type
    # (because we're predicting whether to use Lambda or ECS)
    if 'metric_memory_limit_mb' in df.columns:
        df['lambda_memory_limit_mb'] = df['lambda_memory_limit_mb'].fillna(
            df['metric_memory_limit_mb'].astype(float)
        )

    # 2. Payload size features
    df['payload_size_kb'] = df.get('metric_payload_size_kb', 0)

    # Polynomial features
    df['payload_squared'] = df['payload_size_kb'] ** 2
    df['payload_log'] = np.log1p(df['payload_size_kb'])  # log(1 + x) to handle zeros

    # Categorical binning (quartiles)
    payload_quartiles = df['payload_size_kb'].quantile([0.25, 0.5, 0.75]).tolist()
    df['payload_category'] = pd.cut(
        df['payload_size_kb'],
        bins=[0] + payload_quartiles + [df['payload_size_kb'].max() + 1],
        labels=[0, 1, 2, 3],  # small, medium, large, xlarge
        include_lowest=True
    ).astype(int)

    # 3. Time-based categorical features
    def categorize_hour(hour):
        if 0 <= hour < 6:
            return 0  # night
        elif 6 <= hour < 12:
            return 1  # morning
        elif 12 <= hour < 18:
            return 2  # afternoon
        else:
            return 3  # evening

    df['hour_period'] = df['hour_of_day'].apply(categorize_hour)

    # 4. Interaction features (capture non-linear relationships)
    df['workload_payload_interaction'] = df['workload_type_encoded'] * df['payload_size_kb']
    df['workload_memory_interaction'] = df['workload_type_encoded'] * df['lambda_memory_limit_mb']
    df['memory_payload_interaction'] = df['lambda_memory_limit_mb'] * df['payload_size_kb']
    df['workload_hour_interaction'] = df['workload_type_encoded'] * df['hour_of_day']

    # 5. Ratio features (help model understand relative sizes)
    df['payload_memory_ratio'] = df['payload_size_kb'] / (df['lambda_memory_limit_mb'] + 1)

    print(f"✅ Added {11} advanced features")
    print(f"   - memory_limit_mb (Lambda config)")
    print(f"   - payload_squared, payload_log, payload_category")
    print(f"   - hour_period")
    print(f"   - 4 interaction features")
    print(f"   - payload_memory_ratio")

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

def create_request_level_labels_v2(df):
    """
    TRUE request-level labeling using actual performance variance
    (Same as v4, but now with advanced features)
    """
    print("🏷️  Creating TRUE request-level labels...")

    paired_data = []

    for workload in df['workload_type'].unique():
        workload_df = df[df['workload_type'] == workload]
        lambda_df = workload_df[workload_df['platform'] == 'lambda']
        ecs_df = workload_df[workload_df['platform'] == 'ecs']

        print(f"\n   📊 {workload}:")
        print(f"      Lambda: {len(lambda_df):,} | ECS: {len(ecs_df):,}")

        # Calculate median performance for each platform
        lambda_median_latency = lambda_df['metric_execution_time_ms'].median()
        lambda_median_cost = lambda_df['cost_usd'].median()
        ecs_median_latency = ecs_df['metric_execution_time_ms'].median()
        ecs_median_cost = ecs_df['cost_usd'].median()

        print(f"      Medians: λ_lat={lambda_median_latency:.1f}ms, ecs_lat={ecs_median_latency:.1f}ms")
        print(f"      Medians: λ_cost=${lambda_median_cost:.10f}, ecs_cost=${ecs_median_cost:.10f}")

        # For each Lambda request, compare its actual performance vs median ECS
        for _, row in lambda_df.iterrows():
            actual_lambda_lat = row['metric_execution_time_ms']
            actual_lambda_cost = row['cost_usd']

            # Compare this specific request vs median ECS
            lambda_faster = actual_lambda_lat < ecs_median_latency
            lambda_cheaper = actual_lambda_cost < ecs_median_cost

            cost_optimal = 1 if lambda_cheaper else 0
            latency_optimal = 1 if lambda_faster else 0
            balanced_optimal = 1 if (lambda_faster and lambda_cheaper) else 0

            # Include ALL features (basic + advanced)
            paired_data.append({
                # Basic features
                'workload_type': workload,
                'workload_type_encoded': row['workload_type_encoded'],
                'payload_size_kb': row['payload_size_kb'],
                'hour_of_day': row['hour_of_day'],
                'day_of_week': row['day_of_week'],
                'is_weekend': row['is_weekend'],

                # Advanced features (NEW!)
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

                # Ground truth (NOT features)
                'lambda_latency_ms': actual_lambda_lat,
                'lambda_cost_usd': actual_lambda_cost,
                'lambda_memory_mb': row.get('metric_memory_used_mb', 128),
                'lambda_cold_start': row.get('cold_start', 0),
                'ecs_latency_ms': ecs_median_latency,
                'ecs_cost_usd': ecs_median_cost,

                # Labels
                'cost_optimal': cost_optimal,
                'latency_optimal': latency_optimal,
                'balanced_optimal': balanced_optimal,
                'optimal_platform': balanced_optimal
            })

        # For each ECS request, compare vs median Lambda
        for _, row in ecs_df.iterrows():
            actual_ecs_lat = row['metric_execution_time_ms']
            actual_ecs_cost = row['cost_usd']

            # Compare median Lambda vs this specific ECS request
            lambda_faster = lambda_median_latency < actual_ecs_lat
            lambda_cheaper = lambda_median_cost < actual_ecs_cost

            cost_optimal = 1 if lambda_cheaper else 0
            latency_optimal = 1 if lambda_faster else 0
            balanced_optimal = 1 if (lambda_faster and lambda_cheaper) else 0

            paired_data.append({
                # Basic features
                'workload_type': workload,
                'workload_type_encoded': row['workload_type_encoded'],
                'payload_size_kb': row['payload_size_kb'],
                'hour_of_day': row['hour_of_day'],
                'day_of_week': row['day_of_week'],
                'is_weekend': row['is_weekend'],

                # Advanced features (NEW!)
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

                # Ground truth (NOT features)
                'lambda_latency_ms': lambda_median_latency,
                'lambda_cost_usd': lambda_median_cost,
                'lambda_memory_mb': LAMBDA_MEMORY_CONFIGS[workload],
                'lambda_cold_start': lambda_df['cold_start'].mean() if 'cold_start' in lambda_df.columns else 0,
                'ecs_latency_ms': actual_ecs_lat,
                'ecs_cost_usd': actual_ecs_cost,

                # Labels
                'cost_optimal': cost_optimal,
                'latency_optimal': latency_optimal,
                'balanced_optimal': balanced_optimal,
                'optimal_platform': balanced_optimal
            })

    paired_df = pd.DataFrame(paired_data)

    print(f"\n✅ Created {len(paired_df):,} request-level samples")
    print(f"\n   📊 Label distributions:")

    for label_col in ['cost_optimal', 'latency_optimal', 'balanced_optimal']:
        counts = paired_df[label_col].value_counts()
        print(f"      {label_col}:")
        print(f"        Lambda (1): {counts.get(1, 0):,} ({counts.get(1, 0) / len(paired_df) * 100:.1f}%)")
        print(f"        ECS (0):    {counts.get(0, 0):,} ({counts.get(0, 0) / len(paired_df) * 100:.1f}%)")

    # Check variance within workloads
    print(f"\n   📊 Label variance by workload (balanced_optimal):")
    variance = paired_df.groupby('workload_type')['balanced_optimal'].agg(['mean', 'std', 'nunique'])
    print(variance)

    return paired_df

def main():
    print("=" * 80)
    print("HYBRID THESIS - ADVANCED FEATURE ENGINEERING (v5)")
    print("=" * 80)
    print("TARGET: Improve accuracy from 77.6% to 85%+")
    print("=" * 80)
    print()

    DATA_DIR = 'data-output'
    OUTPUT_DIR = 'ml-notebooks/processed-data'
    DATE_FILTER = ['2025-11-16', '2025-11-17']

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    df = load_jsonl_files(DATA_DIR, date_filter=DATE_FILTER)
    df = normalize_metrics(df)
    df = engineer_basic_features(df)
    df = engineer_advanced_features(df)  # NEW!
    df = add_cost_calculations(df)
    paired_df = create_request_level_labels_v2(df)

    # Save outputs
    paired_output_path = f"{OUTPUT_DIR}/ml_training_data_v5_advanced.csv"
    paired_df.to_csv(paired_output_path, index=False)
    print(f"\n💾 Saved: {paired_output_path}")

    # List all features
    feature_columns = [
        # Basic features (5)
        'workload_type_encoded',
        'payload_size_kb',
        'hour_of_day',
        'day_of_week',
        'is_weekend',
        # Advanced features (11)
        'lambda_memory_limit_mb',
        'payload_squared',
        'payload_log',
        'payload_category',
        'hour_period',
        'workload_payload_interaction',
        'workload_memory_interaction',
        'memory_payload_interaction',
        'workload_hour_interaction',
        'payload_memory_ratio',
    ]

    summary = {
        'total_requests': len(df),
        'total_training_samples': len(paired_df),
        'date_range': DATE_FILTER,
        'num_features': len(feature_columns),
        'feature_columns': feature_columns,
        'label_distribution_cost': paired_df['cost_optimal'].value_counts().to_dict(),
        'label_distribution_latency': paired_df['latency_optimal'].value_counts().to_dict(),
        'label_distribution_balanced': paired_df['balanced_optimal'].value_counts().to_dict(),
    }

    summary_path = f"{OUTPUT_DIR}/preprocessing_summary_v5.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved: {summary_path}")

    # Feature correlation analysis
    print("\n" + "=" * 80)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 80)
    correlations = paired_df[feature_columns + ['balanced_optimal']].corr()['balanced_optimal'].drop('balanced_optimal').sort_values(ascending=False)
    print("\nTop 10 most correlated features:")
    for feat, corr in correlations.head(10).items():
        print(f"  {feat:40s}: {corr:+.4f}")

    print("\n" + "=" * 80)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Total features: {len(feature_columns)} (5 basic + 11 advanced)")
    print(f"🎯 Upload to Colab: {paired_output_path}")
    print(f"\n🚀 Expected improvements:")
    print(f"   - Interaction features capture non-linear relationships")
    print(f"   - Polynomial features help models learn complex boundaries")
    print(f"   - Memory config adds Lambda-specific cost context")
    print(f"   - Target accuracy: 85%+ (up from 77.6%)")

if __name__ == '__main__':
    main()
