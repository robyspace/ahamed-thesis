#!/usr/bin/env python3
"""
Data Preprocessing with REQUEST-LEVEL Labeling (Fixed Overfitting)
Hybrid Serverless-Container Thesis - Phase 3: ML Model Development

ISSUE FIXED: Previous version created ONE label per workload type
→ Model achieved 100% accuracy by memorizing workload→label mapping
→ Other features (payload_size, time) were irrelevant

NEW APPROACH: Create labels at request level with payload-size binning
→ Each workload type split into payload size bins
→ Labels vary based on workload + payload size combination
→ Model must learn from multiple features, not just workload type

This creates realistic variance and prevents overfitting.
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

        print(f" Loading: {file_path.name}")
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    all_data.append(json.loads(line.strip()))
                except:
                    continue

    df = pd.DataFrame(all_data)
    print(f" Loaded {len(df):,} total requests\n")
    return df

def normalize_metrics(df):
    metrics_df = pd.json_normalize(df['metrics'])
    metrics_df.columns = ['metric_' + col for col in metrics_df.columns]
    df = df.drop('metrics', axis=1)
    df = pd.concat([df, metrics_df], axis=1)
    return df

def engineer_features(df):
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

def add_cost_calculations(df):
    print(" Calculating costs...")

    lambda_df = df[df['platform'] == 'lambda'].copy()
    if len(lambda_df) > 0:
        lambda_df['cost_usd'] = lambda_df.apply(
            lambda row: calculate_lambda_cost({
                'memory_used_mb': row.get('metric_memory_used_mb', 128),
                'execution_time_ms': row.get('metric_execution_time_ms', 0)
            }), axis=1
        )
        print(f"   Lambda: {len(lambda_df):,} requests, avg cost ${lambda_df['cost_usd'].mean():.10f}")

    ecs_df = df[df['platform'] == 'ecs'].copy()
    if len(ecs_df) > 0:
        ecs_df['cost_usd'] = ecs_df.apply(
            lambda row: calculate_ecs_cost({
                'workload_type': row.get('workload_type', 'lightweight_api'),
                'execution_time_ms': row.get('metric_execution_time_ms', 0)
            }), axis=1
        )
        print(f"   ECS: {len(ecs_df):,} requests, avg cost ${ecs_df['cost_usd'].mean():.10f}")

    df = pd.concat([lambda_df, ecs_df], axis=0)
    df = df.dropna(subset=['cost_usd'])
    print(f" Total: {len(df):,} requests with costs\n")
    return df

def create_request_level_labels(df, payload_bins=5):
    """
    Create labels at REQUEST LEVEL with payload-size binning

    This fixes the overfitting issue by creating label variation within each workload type.

    Strategy:
    1. For each workload type, split into payload size bins (quintiles)
    2. For each bin, calculate optimal platform based on aggregate stats
    3. Assign labels based on workload + payload_size_bin combination
    4. This creates ~20 combinations (4 workloads × 5 bins) instead of just 4
    """
    print("  Creating REQUEST-LEVEL labels with payload binning...")
    print(f"   Using {payload_bins} payload size bins per workload\n")

    paired_data = []
    label_stats = {}

    for workload in df['workload_type'].unique():
        workload_df = df[df['workload_type'] == workload].copy()
        lambda_df = workload_df[workload_df['platform'] == 'lambda']
        ecs_df = workload_df[workload_df['platform'] == 'ecs']

        print(f"    {workload}:")
        print(f"      Lambda: {len(lambda_df):,} | ECS: {len(ecs_df):,}")

        # Create payload size bins using quantile-based binning
        # pd.qcut automatically handles duplicate edge values
        all_payloads = pd.concat([
            lambda_df['metric_payload_size_kb'],
            ecs_df['metric_payload_size_kb']
        ])

        # Assign bins to Lambda and ECS data using qcut (quantile cut)
        try:
            lambda_df['payload_bin'] = pd.qcut(
                lambda_df['metric_payload_size_kb'],
                q=payload_bins,
                labels=False,
                duplicates='drop'
            )
        except ValueError:
            # If not enough unique values for 5 bins, use fewer bins
            lambda_df['payload_bin'] = pd.qcut(
                lambda_df['metric_payload_size_kb'],
                q=min(payload_bins, lambda_df['metric_payload_size_kb'].nunique()),
                labels=False,
                duplicates='drop'
            )

        try:
            ecs_df['payload_bin'] = pd.qcut(
                ecs_df['metric_payload_size_kb'],
                q=payload_bins,
                labels=False,
                duplicates='drop'
            )
        except ValueError:
            # If not enough unique values for 5 bins, use fewer bins
            ecs_df['payload_bin'] = pd.qcut(
                ecs_df['metric_payload_size_kb'],
                q=min(payload_bins, ecs_df['metric_payload_size_kb'].nunique()),
                labels=False,
                duplicates='drop'
            )

        # For each bin, determine optimal platform
        bin_labels_map = {}

        # Get actual unique bins (might be less than payload_bins if duplicates were dropped)
        actual_bins = sorted(pd.concat([lambda_df['payload_bin'], ecs_df['payload_bin']]).dropna().unique())

        for bin_id in actual_bins:
            lambda_bin = lambda_df[lambda_df['payload_bin'] == bin_id]
            ecs_bin = ecs_df[ecs_df['payload_bin'] == bin_id]

            if len(lambda_bin) == 0 or len(ecs_bin) == 0:
                # Default to overall workload optimal if bin is empty
                lambda_faster = lambda_df['metric_execution_time_ms'].median() < ecs_df['metric_execution_time_ms'].median()
                lambda_cheaper = lambda_df['cost_usd'].median() < ecs_df['cost_usd'].median()
                cost_optimal = 1 if lambda_cheaper else 0
                latency_optimal = 1 if lambda_faster else 0
                balanced_optimal = 1 if (lambda_faster and lambda_cheaper) else 0
            else:
                # Compare bin-specific statistics
                lambda_lat = lambda_bin['metric_execution_time_ms'].median()
                ecs_lat = ecs_bin['metric_execution_time_ms'].median()
                lambda_cost = lambda_bin['cost_usd'].median()
                ecs_cost = ecs_bin['cost_usd'].median()

                lambda_faster = lambda_lat < ecs_lat
                lambda_cheaper = lambda_cost < ecs_cost

                cost_optimal = 1 if lambda_cheaper else 0
                latency_optimal = 1 if lambda_faster else 0
                balanced_optimal = 1 if (lambda_faster and lambda_cheaper) else 0

            bin_labels_map[bin_id] = {
                'cost_optimal': cost_optimal,
                'latency_optimal': latency_optimal,
                'balanced_optimal': balanced_optimal
            }

        # Create records with bin-specific labels
        for _, row in lambda_df.iterrows():
            bin_id = row['payload_bin']
            if pd.isna(bin_id):
                continue

            bin_labels = bin_labels_map.get(int(bin_id), bin_labels_map[0])

            paired_data.append({
                'workload_type': workload,
                'workload_type_encoded': row['workload_type_encoded'],
                'payload_size_kb': row.get('metric_payload_size_kb', 0),
                'payload_bin': int(bin_id),
                'hour_of_day': row['hour_of_day'],
                'day_of_week': row['day_of_week'],
                'is_weekend': row['is_weekend'],
                'lambda_latency_ms': row['metric_execution_time_ms'],
                'lambda_cost_usd': row['cost_usd'],
                'lambda_memory_mb': row.get('metric_memory_used_mb', 128),
                'lambda_cold_start': row.get('cold_start', 0),
                'ecs_latency_ms': ecs_df['metric_execution_time_ms'].median(),
                'ecs_cost_usd': ecs_df['cost_usd'].median(),
                'cost_optimal': bin_labels['cost_optimal'],
                'latency_optimal': bin_labels['latency_optimal'],
                'balanced_optimal': bin_labels['balanced_optimal'],
                'optimal_platform': bin_labels['balanced_optimal']
            })

        for _, row in ecs_df.iterrows():
            bin_id = row['payload_bin']
            if pd.isna(bin_id):
                continue

            bin_labels = bin_labels_map.get(int(bin_id), bin_labels_map[0])

            paired_data.append({
                'workload_type': workload,
                'workload_type_encoded': row['workload_type_encoded'],
                'payload_size_kb': row.get('metric_payload_size_kb', 0),
                'payload_bin': int(bin_id),
                'hour_of_day': row['hour_of_day'],
                'day_of_week': row['day_of_week'],
                'is_weekend': row['is_weekend'],
                'lambda_latency_ms': lambda_df['metric_execution_time_ms'].median(),
                'lambda_cost_usd': lambda_df['cost_usd'].median(),
                'lambda_memory_mb': 128,
                'lambda_cold_start': lambda_df['cold_start'].mean() if 'cold_start' in lambda_df.columns else 0,
                'ecs_latency_ms': row['metric_execution_time_ms'],
                'ecs_cost_usd': row['cost_usd'],
                'cost_optimal': bin_labels['cost_optimal'],
                'latency_optimal': bin_labels['latency_optimal'],
                'balanced_optimal': bin_labels['balanced_optimal'],
                'optimal_platform': bin_labels['balanced_optimal']
            })

        # Print bin statistics
        print(f"      Payload bins created: {len(bin_labels_map)}")
        for bin_id, labels in bin_labels_map.items():
            print(f"        Bin {bin_id}: balanced_optimal = {labels['balanced_optimal']}")

    paired_df = pd.DataFrame(paired_data)

    print(f"\n Created {len(paired_df):,} paired samples with request-level labels")
    print(f"\n   Label distributions:")

    for label_col in ['cost_optimal', 'latency_optimal', 'balanced_optimal']:
        counts = paired_df[label_col].value_counts()
        print(f"      {label_col}:")
        print(f"        Lambda (1): {counts.get(1, 0):,} ({counts.get(1, 0) / len(paired_df) * 100:.1f}%)")
        print(f"        ECS (0):    {counts.get(0, 0):,} ({counts.get(0, 0) / len(paired_df) * 100:.1f}%)")

    # Check label variance by workload
    print(f"\n   Label variance by workload (balanced_optimal):")
    variance_check = paired_df.groupby('workload_type')['balanced_optimal'].agg(['mean', 'std', 'count'])
    print(variance_check)

    return paired_df

def main():
    print("=" * 80)
    print("HYBRID THESIS - REQUEST-LEVEL LABELING (OVERFITTING FIX)")
    print("=" * 80)
    print()

    DATA_DIR = 'data-output'
    OUTPUT_DIR = 'ml-notebooks/processed-data'
    DATE_FILTER = ['2025-11-16', '2025-11-17']

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    df = load_jsonl_files(DATA_DIR, date_filter=DATE_FILTER)
    df = normalize_metrics(df)
    df = engineer_features(df)
    df = add_cost_calculations(df)
    paired_df = create_request_level_labels(df, payload_bins=5)

    # Save outputs
    full_output_path = f"{OUTPUT_DIR}/full_processed_data.csv"
    df.to_csv(full_output_path, index=False)
    print(f"\n Saved: {full_output_path}")

    paired_output_path = f"{OUTPUT_DIR}/ml_training_data_fixed.csv"
    paired_df.to_csv(paired_output_path, index=False)
    print(f" Saved: {paired_output_path}")

    summary = {
        'total_requests': len(df),
        'total_training_samples': len(paired_df),
        'date_range': DATE_FILTER,
        'label_distribution_cost': paired_df['cost_optimal'].value_counts().to_dict(),
        'label_distribution_latency': paired_df['latency_optimal'].value_counts().to_dict(),
        'label_distribution_balanced': paired_df['balanced_optimal'].value_counts().to_dict(),
        'unique_combinations': int(paired_df.groupby(['workload_type_encoded', 'balanced_optimal']).ngroups),
    }

    summary_path = f"{OUTPUT_DIR}/preprocessing_summary_fixed.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f" Saved: {summary_path}")

    print("\n" + "=" * 80)
    print(" PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\n Key Improvement:")
    print(f"   Unique (workload, label) combinations: {summary['unique_combinations']}")
    print(f"   (Previously: 4 combinations → 100% accuracy)")
    print(f"   (Now: {summary['unique_combinations']} combinations → Model must learn patterns!)")
    print(f"\n Upload to Colab: {paired_output_path}")

if __name__ == '__main__':
    main()
