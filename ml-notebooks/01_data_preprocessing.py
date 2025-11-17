#!/usr/bin/env python3
"""
Data Preprocessing Pipeline with Cost Calculation
Hybrid Serverless-Container Thesis - Phase 3: ML Model Development

This script:
1. Loads all JSONL data files
2. Calculates per-request costs for Lambda and ECS
3. Pairs Lambda and ECS requests by workload type
4. Generates ground truth labels (optimal platform)
5. Exports clean dataset for ML training

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
    """
    Calculate cost for a single Lambda request

    Cost = Duration Cost + Request Cost
    Duration Cost = (Memory in GB) × (Execution time in seconds) × $0.0000166667
    Request Cost = $0.0000002 per request
    """
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
    """
    Calculate cost for a single ECS request

    ECS charges by task runtime. We calculate the cost as if this request
    was the only one handled by the task (worst-case scenario).

    Cost = (vCPU Cost + Memory Cost) × Execution Time
    vCPU Cost = vCPU × $0.04656 per hour
    Memory Cost = Memory GB × $0.00511 per hour
    """
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
    """
    Load all JSONL files from data-output directory

    Args:
        data_dir: Directory containing JSONL files
        platform: Filter by platform ('lambda' or 'ecs')
        date_filter: List of dates to include (e.g., ['2025-11-16', '2025-11-17'])

    Returns:
        pandas DataFrame with all data
    """
    data_path = Path(data_dir)
    all_data = []

    # Find all JSONL files
    pattern = '*.jsonl'
    if platform:
        pattern = f'{platform}_*.jsonl'

    jsonl_files = sorted(data_path.glob(pattern))

    print(f" Found {len(jsonl_files)} JSONL files")

    for file_path in jsonl_files:
        # Skip undefined files
        if 'undefined' in file_path.name:
            print(f" Skipping: {file_path.name}")
            continue

        # Apply date filter if specified
        if date_filter:
            if not any(date in file_path.name for date in date_filter):
                continue

        print(f" Loading: {file_path.name}")

        # Read JSONL file line by line
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
    """
    Normalize nested metrics structure into flat columns
    """
    print("\n Normalizing metrics...")

    # Expand metrics dictionary into separate columns
    metrics_df = pd.json_normalize(df['metrics'])

    # Add prefix to avoid column name conflicts
    metrics_df.columns = ['metric_' + col for col in metrics_df.columns]

    # Drop original metrics column and concat normalized metrics
    df = df.drop('metrics', axis=1)
    df = pd.concat([df, metrics_df], axis=1)

    print(f" Normalized {len(metrics_df.columns)} metric fields")

    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """
    Create additional features for ML models
    """
    print("\n  Engineering features...")

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Temporal features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Workload type encoding (categorical to numeric)
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
    """
    Calculate per-request costs for both platforms
    """
    print("\n Calculating costs...")

    # Separate Lambda and ECS data
    lambda_mask = df['platform'] == 'lambda'
    ecs_mask = df['platform'] == 'ecs'

    # Calculate Lambda costs
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

    # Calculate ECS costs
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

    # Combine back
    df = pd.concat([lambda_df, ecs_df], axis=0)

    # Remove rows with invalid costs
    df = df.dropna(subset=['cost_usd'])

    print(f"\n Cost calculation complete: {len(df):,} valid requests")

    return df


# ============================================================================
# GROUND TRUTH LABELING
# ============================================================================

def create_paired_dataset(df, latency_tolerance=1.1, cost_tolerance=1.1):
    """
    Create paired dataset with ground truth labels

    For each workload type, we compare Lambda vs ECS performance and create
    labels indicating which platform is optimal.

    Ground Truth Logic:
    - optimal_platform = 1 (Lambda) if:
        * Lambda latency <= ECS latency × latency_tolerance AND
        * Lambda cost <= ECS cost × cost_tolerance
    - optimal_platform = 0 (ECS) otherwise

    Args:
        df: DataFrame with both Lambda and ECS data
        latency_tolerance: Allow Lambda to be X% slower (default: 1.1 = 10%)
        cost_tolerance: Allow Lambda to be X% more expensive (default: 1.1 = 10%)

    Returns:
        DataFrame with paired comparisons and labels
    """
    print("\n Creating ground truth labels...")
    print(f"   Latency tolerance: {(latency_tolerance - 1) * 100:.0f}%")
    print(f"   Cost tolerance: {(cost_tolerance - 1) * 100:.0f}%")

    paired_data = []

    # Group by workload type
    for workload in df['workload_type'].unique():
        workload_df = df[df['workload_type'] == workload]

        lambda_df = workload_df[workload_df['platform'] == 'lambda']
        ecs_df = workload_df[workload_df['platform'] == 'ecs']

        print(f"\n   {workload}:")
        print(f"      Lambda: {len(lambda_df):,} | ECS: {len(ecs_df):,}")

        # Calculate aggregate statistics
        lambda_stats = {
            'latency_mean': lambda_df['metric_execution_time_ms'].mean(),
            'latency_median': lambda_df['metric_execution_time_ms'].median(),
            'latency_p95': lambda_df['metric_execution_time_ms'].quantile(0.95),
            'cost_mean': lambda_df['cost_usd'].mean(),
            'cost_median': lambda_df['cost_usd'].median(),
            'cold_start_rate': lambda_df['cold_start'].mean() if 'cold_start' in lambda_df.columns else 0
        }

        ecs_stats = {
            'latency_mean': ecs_df['metric_execution_time_ms'].mean(),
            'latency_median': ecs_df['metric_execution_time_ms'].median(),
            'latency_p95': ecs_df['metric_execution_time_ms'].quantile(0.95),
            'cost_mean': ecs_df['cost_usd'].mean(),
            'cost_median': ecs_df['cost_usd'].median()
        }

        # Determine optimal platform using median values
        lambda_latency = lambda_stats['latency_median']
        ecs_latency = ecs_stats['latency_median']
        lambda_cost = lambda_stats['cost_median']
        ecs_cost = ecs_stats['cost_median']

        latency_ok = lambda_latency <= ecs_latency * latency_tolerance
        cost_ok = lambda_cost <= ecs_cost * cost_tolerance

        optimal_platform = 1 if (latency_ok and cost_ok) else 0
        optimal_platform_name = 'Lambda' if optimal_platform == 1 else 'ECS'

        print(f"      Latency: λ={lambda_latency:.2f}ms vs ECS={ecs_latency:.2f}ms → {'OK' if latency_ok else 'KO'}")
        print(f"      Cost: λ=${lambda_cost:.10f} vs ECS=${ecs_cost:.10f} → {'OK' if cost_ok else 'KO'}")
        print(f"      → Optimal: {optimal_platform_name}")

        # Create paired records (sample-based approach for training)
        # We'll create one record per request with comparison features
        for _, row in lambda_df.iterrows():
            paired_record = {
                # Request identifiers
                'workload_type': workload,
                'workload_type_encoded': row['workload_type_encoded'],

                # Input features (what ML model will see at runtime)
                'payload_size_kb': row.get('metric_payload_size_kb', 0),
                'hour_of_day': row['hour_of_day'],
                'day_of_week': row['day_of_week'],
                'is_weekend': row['is_weekend'],

                # Lambda performance (actual)
                'lambda_latency_ms': row['metric_execution_time_ms'],
                'lambda_cost_usd': row['cost_usd'],
                'lambda_memory_mb': row.get('metric_memory_used_mb', 128),
                'lambda_cold_start': row.get('cold_start', 0),

                # ECS performance (estimated from stats)
                'ecs_latency_ms': ecs_stats['latency_median'],
                'ecs_cost_usd': ecs_stats['cost_median'],

                # Ground truth label
                'optimal_platform': optimal_platform  # 1=Lambda, 0=ECS
            }

            paired_data.append(paired_record)

        # Also add ECS records for balance
        for _, row in ecs_df.iterrows():
            paired_record = {
                # Request identifiers
                'workload_type': workload,
                'workload_type_encoded': row['workload_type_encoded'],

                # Input features
                'payload_size_kb': row.get('metric_payload_size_kb', 0),
                'hour_of_day': row['hour_of_day'],
                'day_of_week': row['day_of_week'],
                'is_weekend': row['is_weekend'],

                # Lambda performance (estimated from stats)
                'lambda_latency_ms': lambda_stats['latency_median'],
                'lambda_cost_usd': lambda_stats['cost_median'],
                'lambda_memory_mb': 128,  # Will vary by workload
                'lambda_cold_start': lambda_stats['cold_start_rate'],

                # ECS performance (actual)
                'ecs_latency_ms': row['metric_execution_time_ms'],
                'ecs_cost_usd': row['cost_usd'],

                # Ground truth label
                'optimal_platform': optimal_platform  # 1=Lambda, 0=ECS
            }

            paired_data.append(paired_record)

    paired_df = pd.DataFrame(paired_data)

    print(f"\n Created {len(paired_df):,} paired training samples")
    print(f"   Label distribution:")
    print(f"      Lambda optimal (1): {(paired_df['optimal_platform'] == 1).sum():,} ({(paired_df['optimal_platform'] == 1).mean() * 100:.1f}%)")
    print(f"      ECS optimal (0): {(paired_df['optimal_platform'] == 0).sum():,} ({(paired_df['optimal_platform'] == 0).mean() * 100:.1f}%)")

    return paired_df


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def main():
    """
    Main preprocessing pipeline
    """
    print("=" * 80)
    print("HYBRID SERVERLESS-CONTAINER THESIS")
    print("Data Preprocessing Pipeline with Cost Calculation")
    print("=" * 80)

    # Configuration
    DATA_DIR = 'data-output'
    OUTPUT_DIR = 'ml-notebooks/processed-data'
    DATE_FILTER = ['2025-11-16', '2025-11-17']  # Only Nov 16-17 data

    # Create output directory
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

    # Step 5: Create paired dataset with labels
    print("\n" + "=" * 80)
    print("STEP 5: GROUND TRUTH LABELING")
    print("=" * 80)
    paired_df = create_paired_dataset(df)

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
        'label_distribution': paired_df['optimal_platform'].value_counts().to_dict(),
        'total_cost_lambda': float(df[df['platform'] == 'lambda']['cost_usd'].sum()),
        'total_cost_ecs': float(df[df['platform'] == 'ecs']['cost_usd'].sum()),
    }

    summary_path = f"{OUTPUT_DIR}/preprocessing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f" Saved summary: {summary_path}")

    print("\n" + "=" * 80)
    print(" PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\n Summary:")
    print(f"   Total requests processed: {len(df):,}")
    print(f"   ML training samples: {len(paired_df):,}")
    print(f"   Features ready for model training")
    print(f"\n Output files:")
    print(f"   1. {full_output_path}")
    print(f"   2. {paired_output_path}")
    print(f"   3. {summary_path}")
    print(f"\n Next step: Upload to Google Colab and start model training!")


if __name__ == '__main__':
    main()
