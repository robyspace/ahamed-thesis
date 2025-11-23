#!/usr/bin/env python3
"""
Variance Experiment Data Preprocessing
Hybrid Serverless-Container Thesis - ML Model with Meaningful Variance

KEY DIFFERENCE FROM v6:
- NO actual performance metrics as features (prevents data leakage)
- Adds variance features: time_window, load_pattern, varying payload sizes
- Creates labels from actual performance but doesn't expose those metrics
- Validates label variance (target: 20-80% per workload, NOT 100%)

This forces the model to LEARN decision boundaries rather than memorize lookups.
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

# Decision margins
LATENCY_MARGIN = 0.10  # Lambda must be 10% faster
COST_MARGIN = 0.10     # Lambda must be 10% cheaper

def calculate_lambda_cost(memory_mb, execution_time_ms):
    """Calculate Lambda cost based on memory and execution time"""
    try:
        memory_gb = float(memory_mb) / 1024
        execution_time_sec = float(execution_time_ms) / 1000
        duration_cost = memory_gb * execution_time_sec * LAMBDA_DURATION_COST_PER_GB_SEC
        return duration_cost + LAMBDA_REQUEST_COST
    except:
        return np.nan

def calculate_ecs_cost(workload_type, execution_time_ms):
    """Calculate ECS cost based on task configuration and execution time"""
    try:
        config = ECS_TASK_CONFIGS[workload_type]
        execution_time_hours = float(execution_time_ms) / 1000 / 3600
        cpu_cost = config['vcpu'] * ECS_VCPU_COST_PER_HOUR * execution_time_hours
        memory_cost = config['memory_gb'] * ECS_MEMORY_COST_PER_GB_HOUR * execution_time_hours
        return cpu_cost + memory_cost
    except:
        return np.nan

def load_variance_experiment_data(base_dir='../data-output/variance-experiment'):
    """
    Load JSONL files from variance experiment directory structure.

    Structure:
      variance-experiment/
        early_morning_low_load/
          lambda_*.jsonl
          ecs_*.jsonl
        morning_peak_ramp_load/
        midday_medium_load/
        evening_burst_load/
        late_night_low_load/
        late_night_ramp_load/
    """
    print("="*80)
    print("LOADING VARIANCE EXPERIMENT DATA")
    print("="*80)

    data_path = Path(base_dir)
    all_data = []

    if not data_path.exists():
        print(f"❌ ERROR: Directory not found: {base_dir}")
        return pd.DataFrame()

    # Find all subdirectories (time_window + load_pattern combinations)
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]

    print(f"\n📂 Found {len(subdirs)} test run directories:")
    for subdir in sorted(subdirs):
        jsonl_files = list(subdir.glob('*.jsonl'))
        if jsonl_files:
            print(f"   - {subdir.name}: {len(jsonl_files)} files")

    print("\n📊 Loading data...")

    for subdir in sorted(subdirs):
        # Skip 'unknown' platform logs
        jsonl_files = [f for f in subdir.glob('*.jsonl') if 'unknown' not in f.name]

        for file_path in jsonl_files:
            print(f"   Loading: {subdir.name}/{file_path.name}...")

            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        all_data.append(record)
                    except json.JSONDecodeError:
                        continue

    df = pd.DataFrame(all_data)
    print(f"\n✅ Loaded {len(df):,} total requests")
    print(f"   Platforms: Lambda={len(df[df['platform']=='lambda']):,}, ECS={len(df[df['platform']=='ecs']):,}")
    print()

    return df

def normalize_metrics(df):
    """Extract nested 'metrics' field into flat columns"""
    if 'metrics' in df.columns:
        metrics_df = pd.json_normalize(df['metrics'])
        metrics_df.columns = ['metric_' + col for col in metrics_df.columns]
        df = df.drop('metrics', axis=1)
        df = pd.concat([df, metrics_df], axis=1)
    return df

def extract_variance_features(df):
    """
    Extract NEW variance features that were introduced in Phase 2.

    These are the key features that create variance:
    - time_window: early_morning, morning_peak, midday, evening, late_night
    - load_pattern: low_load, medium_load, burst_load, ramp_load
    - payload_size_kb: NOW VARIES (0.5-10240 KB)
    """
    print("🔧 Extracting variance features...")

    # 1. Encode time_window (categorical → numeric)
    time_window_encoding = {
        'early_morning': 0,
        'morning_peak': 1,
        'midday': 2,
        'evening': 3,
        'late_night': 4,
        'other': 5  # fallback
    }
    df['time_window_encoded'] = df['time_window'].map(time_window_encoding).fillna(5).astype(int)

    # 2. Encode load_pattern (categorical → numeric)
    load_pattern_encoding = {
        'low_load': 0,
        'medium_load': 1,
        'burst_load': 2,
        'ramp_load': 3,
        'quick_validation': 4,  # fallback for validation tests
        'unknown': 5  # fallback
    }
    df['load_pattern_encoded'] = df['load_pattern'].map(load_pattern_encoding).fillna(5).astype(int)

    # 3. Use target_payload_size_kb (now varies!)
    df['payload_size_kb'] = df.get('target_payload_size_kb', df.get('actual_payload_size_kb', 0))

    print(f"   ✓ Time windows: {df['time_window'].nunique()} unique ({df['time_window'].unique()})")
    print(f"   ✓ Load patterns: {df['load_pattern'].nunique()} unique ({df['load_pattern'].unique()})")
    print(f"   ✓ Payload sizes: {df['payload_size_kb'].nunique()} unique (range: {df['payload_size_kb'].min():.1f} - {df['payload_size_kb'].max():.1f} KB)")

    return df

def engineer_predictive_features(df):
    """
    Engineer features that the model can use for prediction.

    CRITICAL: Do NOT include actual performance metrics!
    - ❌ NO lambda_latency_ms, ecs_latency_ms
    - ❌ NO lambda_cost_usd, ecs_cost_usd
    - ❌ NO lambda_cold_start (actual observed)
    - ✅ YES to payload_size, time_window, load_pattern, workload_type
    """
    print("\n🔧 Engineering predictive features...")

    # Temporal features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Workload encoding
    workload_encoding = {
        'lightweight_api': 0,
        'thumbnail_processing': 1,
        'medium_processing': 2,
        'heavy_processing': 3
    }
    df['workload_type_encoded'] = df['workload_type'].map(workload_encoding)

    # Lambda memory config (predictable, not actual performance)
    df['lambda_memory_limit_mb'] = df['workload_type'].map(LAMBDA_MEMORY_CONFIGS)

    # Polynomial features for payload
    df['payload_squared'] = df['payload_size_kb'] ** 2
    df['payload_log'] = np.log1p(df['payload_size_kb'])

    # Interaction features (key for learning complex patterns)
    df['payload_workload_interaction'] = df['payload_size_kb'] * df['workload_type_encoded']
    df['payload_hour_interaction'] = df['payload_size_kb'] * df['hour_of_day']

    # NEW: Variance-specific interactions
    df['payload_time_window_interaction'] = df['payload_size_kb'] * df['time_window_encoded']
    df['workload_time_window_interaction'] = df['workload_type_encoded'] * df['time_window_encoded']
    df['payload_load_pattern_interaction'] = df['payload_size_kb'] * df['load_pattern_encoded']
    df['time_window_load_pattern_interaction'] = df['time_window_encoded'] * df['load_pattern_encoded']

    print(f"   ✓ Added 14 predictive features")
    print(f"   ✓ NO actual performance metrics included (prevents data leakage)")

    return df

def add_cost_calculations(df):
    """
    Calculate costs for label creation only.
    These will NOT be used as features!
    """
    print("\n💰 Calculating costs (for labels only, NOT features)...")

    # For Lambda requests
    lambda_mask = df['platform'] == 'lambda'
    df.loc[lambda_mask, 'cost_usd'] = df.loc[lambda_mask].apply(
        lambda row: calculate_lambda_cost(
            row.get('metric_memory_used_mb', LAMBDA_MEMORY_CONFIGS.get(row['workload_type'], 128)),
            row.get('metric_execution_time_ms', 0)
        ), axis=1
    )

    # For ECS requests
    ecs_mask = df['platform'] == 'ecs'
    df.loc[ecs_mask, 'cost_usd'] = df.loc[ecs_mask].apply(
        lambda row: calculate_ecs_cost(
            row['workload_type'],
            row.get('metric_execution_time_ms', 0)
        ), axis=1
    )

    # Get execution time (for labels, not features)
    df['execution_time_ms'] = df.get('metric_execution_time_ms', 0)

    # Remove invalid entries
    df = df.dropna(subset=['cost_usd', 'execution_time_ms'])

    print(f"   Lambda requests: {len(df[df['platform']=='lambda']):,}")
    print(f"   ECS requests: {len(df[df['platform']=='ecs']):,}")
    print(f"   Total valid: {len(df):,}")

    return df

def create_variance_labels(df):
    """
    Create labels by comparing Lambda vs ECS performance.

    CRITICAL: Labels are created from actual performance,
    but actual performance metrics are NOT included as features!

    This forces the model to learn patterns from:
    - payload_size_kb (now varies!)
    - time_window (cold start proxy)
    - load_pattern (scaling behavior)
    - workload_type (baseline characteristic)
    """
    print("\n🏷️  Creating variance-aware labels...")
    print("="*80)

    paired_data = []
    label_stats = []

    for workload in sorted(df['workload_type'].unique()):
        workload_df = df[df['workload_type'] == workload]
        lambda_df = workload_df[workload_df['platform'] == 'lambda']
        ecs_df = workload_df[workload_df['platform'] == 'ecs']

        print(f"\n📊 {workload}:")
        print(f"   Lambda: {len(lambda_df):,} requests | ECS: {len(ecs_df):,} requests")

        if len(lambda_df) == 0 or len(ecs_df) == 0:
            print(f"   ⚠️  Skipping (insufficient data)")
            continue

        # Calculate percentiles for comparison
        lambda_lat_p50 = lambda_df['execution_time_ms'].quantile(0.50)
        lambda_cost_p50 = lambda_df['cost_usd'].quantile(0.50)
        ecs_lat_p50 = ecs_df['execution_time_ms'].quantile(0.50)
        ecs_cost_p50 = ecs_df['cost_usd'].quantile(0.50)

        print(f"   Lambda median: {lambda_lat_p50:.1f}ms, ${lambda_cost_p50:.10f}")
        print(f"   ECS median:    {ecs_lat_p50:.1f}ms, ${ecs_cost_p50:.10f}")

        # Label Lambda requests
        lambda_optimal_count = 0
        for _, row in lambda_df.iterrows():
            actual_lat = row['execution_time_ms']
            actual_cost = row['cost_usd']
            compare_ecs_lat = ecs_lat_p50
            compare_ecs_cost = ecs_cost_p50

            # Lambda optimal if significantly better in both metrics
            lambda_faster = actual_lat < (compare_ecs_lat * (1 - LATENCY_MARGIN))
            lambda_cheaper = actual_cost < (compare_ecs_cost * (1 - COST_MARGIN))

            balanced_optimal = 1 if (lambda_faster and lambda_cheaper) else 0
            lambda_optimal_count += balanced_optimal

            # Extract ONLY predictive features (NO performance metrics!)
            paired_data.append({
                # Core predictive features
                'workload_type': workload,
                'workload_type_encoded': row['workload_type_encoded'],
                'payload_size_kb': row['payload_size_kb'],
                'time_window_encoded': row['time_window_encoded'],
                'load_pattern_encoded': row['load_pattern_encoded'],
                'hour_of_day': row['hour_of_day'],
                'is_weekend': row['is_weekend'],
                'lambda_memory_limit_mb': row['lambda_memory_limit_mb'],

                # Engineered features
                'payload_squared': row['payload_squared'],
                'payload_log': row['payload_log'],
                'payload_workload_interaction': row['payload_workload_interaction'],
                'payload_hour_interaction': row['payload_hour_interaction'],
                'payload_time_window_interaction': row['payload_time_window_interaction'],
                'workload_time_window_interaction': row['workload_time_window_interaction'],
                'payload_load_pattern_interaction': row['payload_load_pattern_interaction'],
                'time_window_load_pattern_interaction': row['time_window_load_pattern_interaction'],

                # Label (target variable)
                'balanced_optimal': balanced_optimal,

                # Metadata for analysis (NOT used as features)
                'time_window': row['time_window'],
                'load_pattern': row['load_pattern']
            })

        # Label ECS requests
        ecs_optimal_count = 0
        for _, row in ecs_df.iterrows():
            actual_lat = row['execution_time_ms']
            actual_cost = row['cost_usd']
            compare_lambda_lat = lambda_lat_p50
            compare_lambda_cost = lambda_cost_p50

            # Lambda optimal if significantly better than ECS
            lambda_faster = compare_lambda_lat < (actual_lat * (1 - LATENCY_MARGIN))
            lambda_cheaper = compare_lambda_cost < (actual_cost * (1 - COST_MARGIN))

            balanced_optimal = 1 if (lambda_faster and lambda_cheaper) else 0
            lambda_optimal_count += balanced_optimal

            paired_data.append({
                'workload_type': workload,
                'workload_type_encoded': row['workload_type_encoded'],
                'payload_size_kb': row['payload_size_kb'],
                'time_window_encoded': row['time_window_encoded'],
                'load_pattern_encoded': row['load_pattern_encoded'],
                'hour_of_day': row['hour_of_day'],
                'is_weekend': row['is_weekend'],
                'lambda_memory_limit_mb': row['lambda_memory_limit_mb'],
                'payload_squared': row['payload_squared'],
                'payload_log': row['payload_log'],
                'payload_workload_interaction': row['payload_workload_interaction'],
                'payload_hour_interaction': row['payload_hour_interaction'],
                'payload_time_window_interaction': row['payload_time_window_interaction'],
                'workload_time_window_interaction': row['workload_time_window_interaction'],
                'payload_load_pattern_interaction': row['payload_load_pattern_interaction'],
                'time_window_load_pattern_interaction': row['time_window_load_pattern_interaction'],
                'balanced_optimal': balanced_optimal,
                'time_window': row['time_window'],
                'load_pattern': row['load_pattern']
            })

        total_requests = len(lambda_df) + len(ecs_df)
        lambda_optimal_pct = (lambda_optimal_count / total_requests) * 100

        print(f"   Lambda optimal: {lambda_optimal_count}/{total_requests} ({lambda_optimal_pct:.1f}%)")

        label_stats.append({
            'workload': workload,
            'total_requests': total_requests,
            'lambda_optimal_count': lambda_optimal_count,
            'lambda_optimal_pct': lambda_optimal_pct
        })

    paired_df = pd.DataFrame(paired_data)
    stats_df = pd.DataFrame(label_stats)

    print("\n" + "="*80)
    print("LABEL DISTRIBUTION SUMMARY")
    print("="*80)
    print(stats_df.to_string(index=False))

    return paired_df, stats_df

def validate_label_variance(stats_df):
    """
    Validate that labels have sufficient variance.

    CRITICAL SUCCESS CRITERIA:
    - Each workload should have 20-80% Lambda optimal (NOT 0% or 100%)
    - This ensures the model has meaningful patterns to learn

    If labels are deterministic (0% or 100%), the model will still
    achieve 100% accuracy by memorizing workload_type → label mapping.
    """
    print("\n" + "="*80)
    print("VARIANCE VALIDATION")
    print("="*80)

    failed_workloads = []

    for _, row in stats_df.iterrows():
        workload = row['workload']
        pct = row['lambda_optimal_pct']

        if pct < 20:
            status = "❌ TOO LOW"
            failed_workloads.append(f"{workload}: {pct:.1f}% (need 20-80%)")
        elif pct > 80:
            status = "❌ TOO HIGH"
            failed_workloads.append(f"{workload}: {pct:.1f}% (need 20-80%)")
        else:
            status = "✅ GOOD"

        print(f"{workload:25s} {pct:5.1f}% Lambda optimal  {status}")

    print("="*80)

    if failed_workloads:
        print("\n⚠️  WARNING: Some workloads have insufficient variance:")
        for msg in failed_workloads:
            print(f"   - {msg}")
        print("\nThis may indicate:")
        print("   1. Data collection didn't capture enough variance")
        print("   2. Payload sizes not varying enough")
        print("   3. Time windows not diverse enough")
        print("\nModel may still learn a lookup table! Proceed with caution.")
        return False
    else:
        print("\n✅ ALL WORKLOADS HAVE SUFFICIENT VARIANCE!")
        print("   Model should learn meaningful decision boundaries.")
        return True

def save_preprocessed_data(df, stats_df, output_dir='processed-data'):
    """Save preprocessed data and statistics"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save training data (WITHOUT actual performance metrics!)
    training_file = output_path / f'ml_training_data_variance_v1_{timestamp}.csv'
    df.to_csv(training_file, index=False)
    print(f"\n💾 Saved training data: {training_file}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")

    # Save label statistics
    stats_file = output_path / f'label_statistics_{timestamp}.csv'
    stats_df.to_csv(stats_file, index=False)
    print(f"💾 Saved label stats: {stats_file}")

    # Print feature summary
    print("\n" + "="*80)
    print("FEATURE COLUMNS (for ML training)")
    print("="*80)

    feature_cols = [col for col in df.columns if col not in ['workload_type', 'time_window', 'load_pattern', 'balanced_optimal']]

    print("\nPredictive Features (used by model):")
    for i, col in enumerate(sorted(feature_cols), 1):
        print(f"   {i:2d}. {col}")

    print(f"\nTarget Variable:")
    print(f"    - balanced_optimal (0=ECS, 1=Lambda)")

    print(f"\nMetadata (NOT used as features):")
    print(f"    - workload_type, time_window, load_pattern")

    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*80)

    return training_file, stats_file

def main():
    """Main preprocessing pipeline"""
    print("\n" + "="*80)
    print("VARIANCE EXPERIMENT PREPROCESSING")
    print("Addressing 100% accuracy issue with meaningful variance")
    print("="*80)

    # Step 1: Load variance experiment data
    df = load_variance_experiment_data('../data-output/variance-experiment')

    if len(df) == 0:
        print("❌ ERROR: No data loaded!")
        return

    # Step 2: Normalize nested metrics
    df = normalize_metrics(df)

    # Step 3: Extract variance features
    df = extract_variance_features(df)

    # Step 4: Engineer predictive features (NO actual performance!)
    df = engineer_predictive_features(df)

    # Step 5: Add cost calculations (for labels only)
    df = add_cost_calculations(df)

    # Step 6: Create labels
    paired_df, stats_df = create_variance_labels(df)

    # Step 7: Validate variance
    variance_ok = validate_label_variance(stats_df)

    # Step 8: Save preprocessed data
    training_file, stats_file = save_preprocessed_data(paired_df, stats_df)

    # Final summary
    print(f"\n📊 FINAL STATISTICS:")
    print(f"   Total samples: {len(paired_df):,}")
    print(f"   Lambda optimal: {paired_df['balanced_optimal'].sum():,} ({paired_df['balanced_optimal'].mean()*100:.1f}%)")
    print(f"   ECS optimal: {(~paired_df['balanced_optimal'].astype(bool)).sum():,} ({(1-paired_df['balanced_optimal'].mean())*100:.1f}%)")
    print(f"\n   Variance validation: {'✅ PASSED' if variance_ok else '⚠️  WARNING'}")

    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Review label variance statistics")
    print(f"   2. Train ML models using: {training_file.name}")
    print(f"   3. Target accuracy: 75-90% (NOT 100%!)")
    print(f"   4. Verify feature importance is distributed (no single feature >40%)")

    return paired_df, stats_df

if __name__ == '__main__':
    paired_df, stats_df = main()
