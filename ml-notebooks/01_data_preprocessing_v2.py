#!/usr/bin/env python3
"""
Data Preprocessing Pipeline with DUAL-LABEL Strategy (Cost + Latency)
Hybrid Serverless-Container Thesis - Phase 3: ML Model Development

UPDATED: Addresses 100% Lambda label issue by creating separate optimization targets

Changes:
1. Creates TWO labels instead of one:
   - cost_optimal: 1=Lambda cheaper, 0=ECS cheaper
   - latency_optimal: 1=Lambda faster, 0=ECS faster
2. Allows training separate models or multi-output model
3. Captures cost-latency trade-offs realistically

This script:
1. Loads all JSONL data files
2. Calculates per-request costs for Lambda and ECS
3. Pairs Lambda and ECS requests by workload type
4. Generates TWO ground truth labels (cost + latency)
5. Exports clean dataset for ML training

Author: Ahamed Thesis Project
Date: November 2025 (Updated)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# AWS PRICING CONSTANTS (eu-west-1 - Ireland Region)
# ============================================================================

# Lambda Pricing
LAMBDA_REQUEST_COST = 0.20 / 1_000_000  # $0.20 per 1M requests
LAMBDA_DURATION_COST_PER_GB_SEC = 0.0000166667  # $0.0000166667 per GB-second

# ECS Fargate Pricing
ECS_VCPU_COST_PER_HOUR = 0.04656  # $0.04656 per vCPU-hour
ECS_MEMORY_COST_PER_GB_HOUR = 0.00511  # $0.00511 per GB-hour

# ECS Task Configurations (from task definitions)
ECS_TASK_CONFIGS = {
    'lightweight_api': {
        'vcpu': 256 / 1024,  # 0.25 vCPU
        'memory_gb': 512 / 1024  # 0.5 GB
    },
    'thumbnail_processing': {
        'vcpu': 512 / 1024,  # 0.5 vCPU
        'memory_gb': 1024 / 1024  # 1.0 GB
    },
    'medium_processing': {
        'vcpu': 1024 / 1024,  # 1.0 vCPU
        'memory_gb': 2048 / 1024  # 2.0 GB
    },
    'heavy_processing': {
        'vcpu': 1024 / 1024,  # 1.0 vCPU
        'memory_gb': 2048 / 1024  # 2.0 GB
    }
}

# ============================================================================
# COST CALCULATION FUNCTIONS
# ============================================================================

def calculate_lambda_cost(row):
    """Calculate cost for a single Lambda request"""
    try:
        memory_mb = float(row['memory_used_mb'])
        execution_time_ms = float(row['execution_time_ms'])

        memory_gb = memory_mb / 1024
        execution_time_sec = execution_time_ms / 1000

        duration_cost = memory_gb * execution_time_sec * LAMBDA_DURATION_COST_PER_GB_SEC
        total_cost = duration_cost + LAMBDA_REQUEST_COST

        return total_cost
    except (KeyError, ValueError, TypeError) as e:
        return np.nan


def calculate_ecs_cost(row):
    """Calculate cost for a single ECS request"""
    try:
        workload_type = row['workload_type']
        execution_time_ms = float(row['execution_time_ms'])

        config = ECS_TASK_CONFIGS[workload_type]
        vcpu = config['vcpu']
        memory_gb = config['memory_gb']

        execution_time_hours = execution_time_ms / 1000 / 3600

        cpu_cost = vcpu * ECS_VCPU_COST_PER_HOUR * execution_time_hours
        memory_cost = memory_gb * ECS_MEMORY_COST_PER_GB_HOUR * execution_time_hours
        total_cost = cpu_cost + memory_cost

        return total_cost
    except (KeyError, ValueError, TypeError) as e:
        return np.nan


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_jsonl_files(data_dir='data-output', platform=None, date_filter=None):
    """Load all JSONL files from data-output directory"""
    data_path = Path(data_dir)
    all_data = []

    pattern = '*.jsonl'
    if platform:
        pattern = f'{platform}_*.jsonl'

    jsonl_files = sorted(data_path.glob(pattern))

    print(f" Found {len(jsonl_files)} JSONL files")

    for file_path in jsonl_files:
        if 'undefined' in file_path.name:
            print(f"⏭  Skipping: {file_path.name}")
            continue

        if date_filter:
            if not any(date in file_path.name for date in date_filter):
                continue

        print(f" Loading: {file_path.name}")

        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    all_data.append(data)
                except json.JSONDecodeError:
                    continue

    df = pd.DataFrame(all_data)
    print(f"\n Loaded {len(df):,} total requests")

    return df


def normalize_metrics(df):
    """Normalize nested metrics structure into flat columns"""
    print("\n Normalizing metrics...")

    metrics_df = pd.json_normalize(df['metrics'])
    metrics_df.columns = ['metric_' + col for col in metrics_df.columns]

    df = df.drop('metrics', axis=1)
    df = pd.concat([df, metrics_df], axis=1)

    print(f" Normalized {len(metrics_df.columns)} metric fields")

    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """Create additional features for ML models"""
    print("\n  Engineering features...")

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Temporal features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Workload type encoding
    workload_encoding = {
        'lightweight_api': 0,
        'thumbnail_processing': 1,
        'medium_processing': 2,
        'heavy_processing': 3
    }
    df['workload_type_encoded'] = df['workload_type'].map(workload_encoding)

    # Platform encoding
    df['platform_encoded'] = (df['platform'] == 'lambda').astype(int)

    # Cold start flag (Lambda only)
    if 'metric_cold_start' in df.columns:
        df['cold_start'] = df['metric_cold_start'].fillna(False).astype(int)
    else:
        df['cold_start'] = 0

    print(" Created temporal and categorical features")

    return df


# ============================================================================
# COST CALCULATION
# ============================================================================

def add_cost_calculations(df):
    """Calculate per-request costs for both platforms"""
    print("\n Calculating costs...")

    lambda_mask = df['platform'] == 'lambda'
    ecs_mask = df['platform'] == 'ecs'

    lambda_df = df[lambda_mask].copy()
    if len(lambda_df) > 0:
        lambda_df['cost_usd'] = lambda_df.apply(
            lambda row: calculate_lambda_cost({
                'memory_used_mb': row.get('metric_memory_used_mb', 128),
                'execution_time_ms': row.get('metric_execution_time_ms', 0)
            }),
            axis=1
        )
        print(f"      Calculated Lambda costs: {len(lambda_df):,} requests")
        print(f"      Average: ${lambda_df['cost_usd'].mean():.10f} per request")
        print(f"      Total: ${lambda_df['cost_usd'].sum():.4f}")

    ecs_df = df[ecs_mask].copy()
    if len(ecs_df) > 0:
        ecs_df['cost_usd'] = ecs_df.apply(
            lambda row: calculate_ecs_cost({
                'workload_type': row.get('workload_type', 'lightweight_api'),
                'execution_time_ms': row.get('metric_execution_time_ms', 0)
            }),
            axis=1
        )
        print(f"      Calculated ECS costs: {len(ecs_df):,} requests")
        print(f"      Average: ${ecs_df['cost_usd'].mean():.10f} per request")
        print(f"      Total: ${ecs_df['cost_usd'].sum():.4f}")

    df = pd.concat([lambda_df, ecs_df], axis=0)
    df = df.dropna(subset=['cost_usd'])

    print(f"\n Cost calculation complete: {len(df):,} valid requests")

    return df


# ============================================================================
# DUAL-LABEL GROUND TRUTH (UPDATED)
# ============================================================================

def create_dual_label_dataset(df):
    """
    Create dataset with DUAL labels: cost_optimal AND latency_optimal

    This addresses the 100% Lambda label issue by separating optimization objectives.

    Labels:
    - cost_optimal: 1 if Lambda cheaper, 0 if ECS cheaper
    - latency_optimal: 1 if Lambda faster, 0 if ECS faster
    - balanced_optimal: 1 if Lambda better on BOTH metrics

    This allows:
    1. Training separate models for cost vs latency optimization
    2. Understanding trade-offs
    3. Multi-objective optimization
    """
    print("\n  Creating DUAL ground truth labels (cost + latency)...")

    paired_data = []

    # Group by workload type
    for workload in df['workload_type'].unique():
        workload_df = df[df['workload_type'] == workload]

        lambda_df = workload_df[workload_df['platform'] == 'lambda']
        ecs_df = workload_df[workload_df['platform'] == 'ecs']

        print(f"\n    {workload}:")
        print(f"      Lambda: {len(lambda_df):,} | ECS: {len(ecs_df):,}")

        # Calculate aggregate statistics
        lambda_stats = {
            'latency_median': lambda_df['metric_execution_time_ms'].median(),
            'cost_median': lambda_df['cost_usd'].median(),
            'cold_start_rate': lambda_df['cold_start'].mean() if 'cold_start' in lambda_df.columns else 0
        }

        ecs_stats = {
            'latency_median': ecs_df['metric_execution_time_ms'].median(),
            'cost_median': ecs_df['cost_usd'].median()
        }

        # Determine winners for each objective
        lambda_faster = lambda_stats['latency_median'] < ecs_stats['latency_median']
        lambda_cheaper = lambda_stats['cost_median'] < ecs_stats['cost_median']

        cost_optimal_label = 1 if lambda_cheaper else 0
        latency_optimal_label = 1 if lambda_faster else 0
        balanced_optimal_label = 1 if (lambda_faster and lambda_cheaper) else 0

        print(f"      Latency: λ={lambda_stats['latency_median']:.2f}ms vs ECS={ecs_stats['latency_median']:.2f}ms → {' Lambda' if lambda_faster else ' ECS'}")
        print(f"      Cost: λ=${lambda_stats['cost_median']:.10f} vs ECS=${ecs_stats['cost_median']:.10f} → {' Lambda' if lambda_cheaper else ' ECS'}")
        print(f"      Labels: cost_optimal={cost_optimal_label}, latency_optimal={latency_optimal_label}, balanced={balanced_optimal_label}")

        # Create records for Lambda samples
        for _, row in lambda_df.iterrows():
            paired_record = {
                # Identifiers
                'workload_type': workload,
                'workload_type_encoded': row['workload_type_encoded'],

                # Input features
                'payload_size_kb': row.get('metric_payload_size_kb', 0),
                'hour_of_day': row['hour_of_day'],
                'day_of_week': row['day_of_week'],
                'is_weekend': row['is_weekend'],

                # Lambda performance (actual)
                'lambda_latency_ms': row['metric_execution_time_ms'],
                'lambda_cost_usd': row['cost_usd'],
                'lambda_memory_mb': row.get('metric_memory_used_mb', 128),
                'lambda_cold_start': row.get('cold_start', 0),

                # ECS performance (aggregate estimate)
                'ecs_latency_ms': ecs_stats['latency_median'],
                'ecs_cost_usd': ecs_stats['cost_median'],

                # DUAL LABELS
                'cost_optimal': cost_optimal_label,          # 1=Lambda cheaper
                'latency_optimal': latency_optimal_label,    # 1=Lambda faster
                'balanced_optimal': balanced_optimal_label,  # 1=Lambda better on both

                # Legacy (for compatibility)
                'optimal_platform': balanced_optimal_label
            }

            paired_data.append(paired_record)

        # Create records for ECS samples
        for _, row in ecs_df.iterrows():
            paired_record = {
                # Identifiers
                'workload_type': workload,
                'workload_type_encoded': row['workload_type_encoded'],

                # Input features
                'payload_size_kb': row.get('metric_payload_size_kb', 0),
                'hour_of_day': row['hour_of_day'],
                'day_of_week': row['day_of_week'],
                'is_weekend': row['is_weekend'],

                # Lambda performance (aggregate estimate)
                'lambda_latency_ms': lambda_stats['latency_median'],
                'lambda_cost_usd': lambda_stats['cost_median'],
                'lambda_memory_mb': 128,
                'lambda_cold_start': lambda_stats['cold_start_rate'],

                # ECS performance (actual)
                'ecs_latency_ms': row['metric_execution_time_ms'],
                'ecs_cost_usd': row['cost_usd'],

                # DUAL LABELS
                'cost_optimal': cost_optimal_label,          # 1=Lambda cheaper
                'latency_optimal': latency_optimal_label,    # 1=Lambda faster
                'balanced_optimal': balanced_optimal_label,  # 1=Lambda better on both

                # Legacy (for compatibility)
                'optimal_platform': balanced_optimal_label
            }

            paired_data.append(paired_record)

    paired_df = pd.DataFrame(paired_data)

    print(f"\n Created {len(paired_df):,} paired training samples with DUAL labels")
    print(f"\n    Label distributions:")
    print(f"      cost_optimal:")
    print(f"        Lambda (1): {(paired_df['cost_optimal'] == 1).sum():,} ({(paired_df['cost_optimal'] == 1).mean() * 100:.1f}%)")
    print(f"        ECS (0):    {(paired_df['cost_optimal'] == 0).sum():,} ({(paired_df['cost_optimal'] == 0).mean() * 100:.1f}%)")

    print(f"      latency_optimal:")
    print(f"        Lambda (1): {(paired_df['latency_optimal'] == 1).sum():,} ({(paired_df['latency_optimal'] == 1).mean() * 100:.1f}%)")
    print(f"        ECS (0):    {(paired_df['latency_optimal'] == 0).sum():,} ({(paired_df['latency_optimal'] == 0).mean() * 100:.1f}%)")

    print(f"      balanced_optimal:")
    print(f"        Lambda (1): {(paired_df['balanced_optimal'] == 1).sum():,} ({(paired_df['balanced_optimal'] == 1).mean() * 100:.1f}%)")
    print(f"        ECS (0):    {(paired_df['balanced_optimal'] == 0).sum():,} ({(paired_df['balanced_optimal'] == 0).mean() * 100:.1f}%)")

    return paired_df


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def main():
    """Main preprocessing pipeline"""
    print("=" * 80)
    print("HYBRID SERVERLESS-CONTAINER THESIS")
    print("Data Preprocessing with DUAL-LABEL Strategy (UPDATED)")
    print("=" * 80)

    # Configuration
    DATA_DIR = 'data-output'
    OUTPUT_DIR = 'ml-notebooks/processed-data'
    DATE_FILTER = ['2025-11-16', '2025-11-17']

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    df = load_jsonl_files(DATA_DIR, date_filter=DATE_FILTER)

    # Step 2: Normalize metrics
    print("\n" + "=" * 80)
    print("STEP 2: NORMALIZING METRICS")
    print("=" * 80)
    df = normalize_metrics(df)

    # Step 3: Engineer features
    print("\n" + "=" * 80)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 80)
    df = engineer_features(df)

    # Step 4: Calculate costs
    print("\n" + "=" * 80)
    print("STEP 4: COST CALCULATION")
    print("=" * 80)
    df = add_cost_calculations(df)

    # Step 5: Create dual-label dataset
    print("\n" + "=" * 80)
    print("STEP 5: DUAL-LABEL GROUND TRUTH (UPDATED)")
    print("=" * 80)
    paired_df = create_dual_label_dataset(df)

    # Step 6: Save processed data
    print("\n" + "=" * 80)
    print("STEP 6: SAVING PROCESSED DATA")
    print("=" * 80)

    # Save full processed data
    full_output_path = f"{OUTPUT_DIR}/full_processed_data.csv"
    df.to_csv(full_output_path, index=False)
    print(f" Saved full data: {full_output_path} ({len(df):,} rows)")

    # Save paired training data
    paired_output_path = f"{OUTPUT_DIR}/ml_training_data.csv"
    paired_df.to_csv(paired_output_path, index=False)
    print(f" Saved ML training data: {paired_output_path} ({len(paired_df):,} rows)")

    # Save summary statistics
    summary = {
        'total_requests': len(df),
        'total_training_samples': len(paired_df),
        'date_range': DATE_FILTER,
        'platforms': df['platform'].value_counts().to_dict(),
        'workload_types': df['workload_type'].value_counts().to_dict(),
        'label_distribution_cost': paired_df['cost_optimal'].value_counts().to_dict(),
        'label_distribution_latency': paired_df['latency_optimal'].value_counts().to_dict(),
        'label_distribution_balanced': paired_df['balanced_optimal'].value_counts().to_dict(),
        'total_cost_lambda': float(df[df['platform'] == 'lambda']['cost_usd'].sum()),
        'total_cost_ecs': float(df[df['platform'] == 'ecs']['cost_usd'].sum()),
    }

    summary_path = f"{OUTPUT_DIR}/preprocessing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f" Saved summary: {summary_path}")

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE (DUAL-LABEL VERSION)!")
    print("=" * 80)
    print(f"\n Summary:")
    print(f"   Total requests processed: {len(df):,}")
    print(f"   ML training samples: {len(paired_df):,}")
    print(f"   Labels: cost_optimal, latency_optimal, balanced_optimal")
    print(f"\n Output files:")
    print(f"   1. {full_output_path}")
    print(f"   2. {paired_output_path}")
    print(f"   3. {summary_path}")
    print(f"\n Next step: Upload to Google Colab and train models!")
    print(f"\n💡 TIP: You can now train 3 different models:")
    print(f"   - Cost optimization model (using cost_optimal label)")
    print(f"   - Latency optimization model (using latency_optimal label)")
    print(f"   - Balanced model (using balanced_optimal label)")


if __name__ == '__main__':
    main()
