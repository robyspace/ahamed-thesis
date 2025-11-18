# ML Variance Experiment - Project Checklist
## Step-by-Step Execution Guide

**Status:** Ready to Start
**Last Updated:** 2025-11-18
**Estimated Time:** 3 days + 1 week validation

---

## Phase 0: Pre-Flight Checks ✈️

**Before starting, verify:**

### Environment Setup
- [ ] Artillery.io installed: `artillery --version`
- [ ] Node.js installed: `node --version`
- [ ] Python 3.8+ installed: `python3 --version`
- [ ] Sufficient disk space: ~5GB free
- [ ] Git repository clean: `git status`

### AWS Infrastructure
- [ ] Lambda endpoint accessible:
  ```bash
  curl -X POST https://jt67vt5uwj.execute-api.eu-west-1.amazonaws.com/prod/lightweight \
    -H "Content-Type: application/json" \
    -d '{"request_id": "test", "workload_type": "lightweight_api", "payload": {"test": "data"}}'
  ```
- [ ] ECS endpoint accessible:
  ```bash
  curl -X POST http://hybrid-thesis-alb-811686247.eu-west-1.elb.amazonaws.com/lightweight/process \
    -H "Content-Type: application/json" \
    -d '{"request_id": "test", "workload_type": "lightweight_api", "payload": {"test": "data"}}'
  ```

### Documentation Review
- [ ] Read: [CURRENT_STATE_SUMMARY.md](../CURRENT_STATE_SUMMARY.md)
- [ ] Read: [ML_VARIANCE_EXPERIMENT_PLAN.md](../ML_VARIANCE_EXPERIMENT_PLAN.md)
- [ ] Read: [NEW_PROJECT_REQUIREMENTS.md](../NEW_PROJECT_REQUIREMENTS.md)

---

## Phase 1: Artillery Setup 🎯

**Objective:** Create variance-based load testing scripts
**Time Estimate:** 4-6 hours
**Directory:** `ml-variance-experiment/artillery-tests/`

### Step 1.1: Create variance-functions.js

- [ ] Navigate to artillery-tests directory:
  ```bash
  cd /home/user/ahamed-thesis/ml-variance-experiment/artillery-tests
  ```

- [ ] Create `variance-functions.js` with the following structure:
  ```javascript
  // See NEW_PROJECT_REQUIREMENTS.md Section 3.2 for full implementation
  const fs = require('fs');
  const path = require('path');

  const PAYLOAD_VARIANCE_CONFIGS = {
    lightweight_api: { sizes_kb: [0.5, 1, 5, 10], ... },
    thumbnail_processing: { sizes_kb: [50, 100, 200, 500, 1024], ... },
    medium_processing: { sizes_kb: [1024, 2048, 3072, 5120], ... },
    heavy_processing: { sizes_kb: [3072, 5120, 8192, 10240], ... }
  };

  // Implement:
  // - generateRequestId()
  // - getCurrentTimeWindow()
  // - generateRandomData(sizeKB)
  // - generateBase64Data(targetSizeKB)
  // - generateVariancePayload(workloadType, loadPattern)
  // - setLightweightVariancePayload()
  // - setThumbnailVariancePayload()
  // - setMediumVariancePayload()
  // - setHeavyVariancePayload()
  // - logDetailedResponse()

  module.exports = { ... };
  ```

- [ ] **Key Implementation Requirements:**
  - Randomize payload size selection from configured ranges
  - Track metadata: target_size_kb, time_window, load_pattern
  - Log to structured directories: `data-output/variance-experiment/{time_window}_{load_pattern}/`
  - Include all fields: target_payload_size_kb, actual_payload_size_kb, time_window, load_pattern

- [ ] Test payload generation locally:
  ```bash
  node -e "
  const funcs = require('./variance-functions.js');
  const testPayload = funcs.generateVariancePayload('lightweight_api', 'low_load');
  console.log(JSON.stringify(testPayload, null, 2));
  "
  ```

- [ ] Verify output includes:
  - `request_id`
  - `workload_type`
  - `timestamp`
  - `metadata.target_size_kb`
  - `metadata.time_window`
  - `metadata.load_pattern`
  - `payload` (sized correctly)

### Step 1.2: Create Artillery Test Configurations

**Create 4 test files:**

- [ ] **variance-test-low-load.yml**
  ```yaml
  config:
    target: "https://jt67vt5uwj.execute-api.eu-west-1.amazonaws.com/prod"
    processor: "./variance-functions.js"
    phases:
      - duration: 3600  # 1 hour
        arrivalRate: 5
    variables:
      ecs_base: "http://hybrid-thesis-alb-811686247.eu-west-1.elb.amazonaws.com"
      load_pattern: "low_load"

  scenarios:
    # See NEW_PROJECT_REQUIREMENTS.md Section 3.3 for full scenario list
    # 8 scenarios × 12.5% weight each (equal distribution)
  ```

- [ ] **variance-test-medium-load.yml**
  ```yaml
  phases:
    - duration: 3600
      arrivalRate: 15
  variables:
    load_pattern: "medium_load"
  # Same scenarios as low-load
  ```

- [ ] **variance-test-burst-load.yml**
  ```yaml
  phases:
    - duration: 600
      arrivalRate: 5
    - duration: 300
      arrivalRate: 30
    - duration: 300
      arrivalRate: 5
  variables:
    load_pattern: "burst_load"
  # Same scenarios
  ```

- [ ] **variance-test-ramp-load.yml**
  ```yaml
  phases:
    - duration: 3600
      arrivalRate: 5
      rampTo: 25
  variables:
    load_pattern: "ramp_load"
  # Same scenarios
  ```

- [ ] **Verify all configs have:**
  - Correct endpoints (Lambda and ECS)
  - Equal weight distribution (12.5% × 8 = 100%)
  - All 4 workload types covered
  - Both platforms tested
  - Processor points to `./variance-functions.js`

### Step 1.3: Create Test Orchestration Script

- [ ] Create `run-variance-tests.sh`:
  ```bash
  #!/bin/bash
  # See NEW_PROJECT_REQUIREMENTS.md Section 3.4 for full implementation

  # Auto-detect time window
  # Suggest appropriate test config
  # Run artillery test
  # Log results
  ```

- [ ] Make executable:
  ```bash
  chmod +x run-variance-tests.sh
  ```

- [ ] Test dry-run:
  ```bash
  ./run-variance-tests.sh --help
  ```

### Step 1.4: Validation Test

- [ ] Run a SHORT test (5 minutes) to verify setup:
  ```bash
  # Temporarily edit variance-test-low-load.yml
  # Change duration: 300 (5 minutes)
  artillery run variance-test-low-load.yml
  ```

- [ ] Verify:
  - [ ] Test runs without errors
  - [ ] Logs created in `data-output/variance-experiment/`
  - [ ] JSONL files contain all required fields
  - [ ] Payload sizes are varying (check multiple log entries)
  - [ ] Both Lambda and ECS requests logged

- [ ] Inspect log file:
  ```bash
  head -5 ../data-output/variance-experiment/*/lambda_lightweight_api_*.jsonl | jq .
  ```

- [ ] Check for required fields:
  - `target_payload_size_kb` ✓
  - `actual_payload_size_kb` ✓
  - `time_window` ✓
  - `load_pattern` ✓
  - `metrics.execution_time_ms` ✓
  - `metrics.cold_start` ✓

**If validation fails, debug before proceeding to data collection!**

---

## Phase 2: Data Collection 📊

**Objective:** Collect ~180K samples across 5 time windows
**Time Estimate:** 5 hours runtime (spread across 1 day)
**Directory:** `ml-variance-experiment/data-output/variance-experiment/`

### Test Execution Schedule

**IMPORTANT:** Run each test during its designated time window for optimal variance!

### Test 1: Early Morning (2-3 AM) - Low Load

- [ ] **Time Window:** 2:00 AM - 3:00 AM
- [ ] **Config:** variance-test-low-load.yml
- [ ] **Expected Characteristic:** High cold starts, cold infrastructure
- [ ] **Pre-flight:**
  - Check time: `date`
  - Verify within window (2-4 AM)
  - Clear previous test data (if any)

- [ ] **Run Test:**
  ```bash
  cd /home/user/ahamed-thesis/ml-variance-experiment/artillery-tests
  artillery run variance-test-low-load.yml > logs/test1_early_morning.log 2>&1
  ```

- [ ] **Monitor:**
  - Watch log output for errors
  - Check system resources: `top` (ensure not overloading)
  - Verify files being written: `ls -lh ../data-output/variance-experiment/early_morning*/`

- [ ] **Post-Test Validation:**
  - [ ] Test completed successfully (60 minutes)
  - [ ] Logs present: `ls ../data-output/variance-experiment/early_morning_low_load/`
  - [ ] Sample count: ~18,000 requests
    ```bash
    wc -l ../data-output/variance-experiment/early_morning_low_load/*.jsonl
    ```
  - [ ] No error messages in artillery log
  - [ ] Both Lambda and ECS data present

### Test 2: Morning Peak (8-9 AM) - Ramp Load

- [ ] **Time Window:** 8:00 AM - 9:00 AM
- [ ] **Config:** variance-test-ramp-load.yml
- [ ] **Expected Characteristic:** Medium cold starts, infrastructure warming up
- [ ] **Pre-flight:** Verify within window (8-10 AM)

- [ ] **Run Test:**
  ```bash
  artillery run variance-test-ramp-load.yml > logs/test2_morning_peak.log 2>&1
  ```

- [ ] **Post-Test Validation:**
  - [ ] Sample count: ~54,000 requests
  - [ ] Logs in: `morning_peak_ramp_load/`
  - [ ] Load ramped as expected (check request rate in logs)

### Test 3: Midday (12-1 PM) - Medium Load

- [ ] **Time Window:** 12:00 PM - 1:00 PM
- [ ] **Config:** variance-test-medium-load.yml
- [ ] **Expected Characteristic:** Low cold starts, peak performance
- [ ] **Pre-flight:** Verify within window (12-2 PM)

- [ ] **Run Test:**
  ```bash
  artillery run variance-test-medium-load.yml > logs/test3_midday.log 2>&1
  ```

- [ ] **Post-Test Validation:**
  - [ ] Sample count: ~54,000 requests
  - [ ] Logs in: `midday_medium_load/`
  - [ ] Consistent arrival rate (15 req/s)

### Test 4: Evening (6-7 PM) - Burst Load

- [ ] **Time Window:** 6:00 PM - 7:00 PM
- [ ] **Config:** variance-test-burst-load.yml
- [ ] **Expected Characteristic:** Second peak, stable performance
- [ ] **Pre-flight:** Verify within window (6-8 PM)

- [ ] **Run Test:**
  ```bash
  artillery run variance-test-burst-load.yml > logs/test4_evening.log 2>&1
  ```

- [ ] **Post-Test Validation:**
  - [ ] Sample count: ~36,000 requests
  - [ ] Logs in: `evening_burst_load/`
  - [ ] Burst pattern visible (spike in request rate)

### Test 5: Late Night (10-11 PM) - Low Load

- [ ] **Time Window:** 10:00 PM - 11:00 PM
- [ ] **Config:** variance-test-low-load.yml
- [ ] **Expected Characteristic:** Increasing cold starts, infrastructure cooling down
- [ ] **Pre-flight:** Verify within window (10 PM - 1 AM)

- [ ] **Run Test:**
  ```bash
  artillery run variance-test-low-load.yml > logs/test5_late_night.log 2>&1
  ```

- [ ] **Post-Test Validation:**
  - [ ] Sample count: ~18,000 requests
  - [ ] Logs in: `late_night_low_load/`

### Phase 2 Summary Validation

- [ ] **Total sample count:** ~180,000 requests
  ```bash
  wc -l ../data-output/variance-experiment/*/*.jsonl | tail -1
  ```

- [ ] **All 5 test runs completed successfully**

- [ ] **Data quality check:**
  ```bash
  # Check for valid JSON
  cat ../data-output/variance-experiment/*/lambda_lightweight_api*.jsonl | head -100 | jq . > /dev/null
  echo "✓ Valid JSON"

  # Check field presence
  cat ../data-output/variance-experiment/*/lambda_lightweight_api*.jsonl | head -1 | jq 'keys'
  # Should include: target_payload_size_kb, time_window, load_pattern
  ```

- [ ] **Payload variance check:**
  ```bash
  # Extract unique payload sizes per workload
  for workload in lightweight_api thumbnail_processing medium_processing heavy_processing; do
    echo "=== $workload ==="
    cat ../data-output/variance-experiment/*/*_${workload}*.jsonl | \
      jq -r '.target_payload_size_kb' | sort -u
  done
  ```
  - Lightweight: Should show 0.5, 1, 5, 10
  - Thumbnail: Should show 50, 100, 200, 500, 1024
  - Medium: Should show 1024, 2048, 3072, 5120
  - Heavy: Should show 3072, 5120, 8192, 10240

**If any validation fails, investigate before proceeding to preprocessing!**

---

## Phase 3: Data Preprocessing 🔧

**Objective:** Transform JSONL logs into ML training data
**Time Estimate:** 2-3 hours
**Directory:** `ml-variance-experiment/preprocessing/`

### Step 3.1: Create Preprocessing Script

- [ ] Navigate to preprocessing directory:
  ```bash
  cd /home/user/ahamed-thesis/ml-variance-experiment/preprocessing
  ```

- [ ] Create `variance_preprocessing.py` based on reference script
  - Use `../ml-notebooks/01_data_preprocessing_v6_percentile_labeling.py` as base
  - **Key modifications needed:**

- [ ] **Modification 1: Load from variance experiment directories**
  ```python
  DATA_DIR = '../data-output/variance-experiment'
  # Load all JSONL files from all subdirectories
  ```

- [ ] **Modification 2: Extract new metadata fields**
  ```python
  df['target_payload_size_kb'] = df['target_payload_size_kb']
  df['time_window'] = df['time_window']
  df['load_pattern'] = df['load_pattern']
  ```

- [ ] **Modification 3: Encode new categorical features**
  ```python
  time_window_encoding = {
      'early_morning': 0,
      'morning_peak': 1,
      'midday': 2,
      'evening': 3,
      'late_night': 4
  }
  df['time_window_encoded'] = df['time_window'].map(time_window_encoding)

  load_pattern_encoding = {
      'low_load': 0,
      'medium_load': 1,
      'burst_load': 2,
      'ramp_load': 3
  }
  df['load_pattern_encoded'] = df['load_pattern'].map(load_pattern_encoding)
  ```

- [ ] **Modification 4: Create new interaction features**
  ```python
  df['payload_time_window_interaction'] = df['payload_size_kb'] * df['time_window_encoded']
  df['workload_time_window_interaction'] = df['workload_type_encoded'] * df['time_window_encoded']
  df['payload_load_pattern_interaction'] = df['payload_size_kb'] * df['load_pattern_encoded']
  ```

- [ ] **Modification 5: Define FINAL feature columns (NO PERFORMANCE METRICS!)**
  ```python
  FEATURE_COLUMNS = [
      # Core features
      'workload_type_encoded',
      'payload_size_kb',  # NOW VARIES!
      'hour_of_day',
      'time_window_encoded',  # NEW
      'load_pattern_encoded',  # NEW
      'is_weekend',
      'lambda_memory_limit_mb',

      # Engineered features
      'payload_squared',
      'payload_log',
      'payload_workload_interaction',
      'payload_hour_interaction',
      'payload_time_window_interaction',  # NEW
      'workload_time_window_interaction',  # NEW
      'payload_load_pattern_interaction'  # NEW
  ]

  # CRITICAL: DO NOT INCLUDE THESE
  FORBIDDEN_FEATURES = [
      'lambda_latency_ms',
      'lambda_cost_usd',
      'ecs_latency_ms',
      'ecs_cost_usd',
      'lambda_cold_start',
      'lambda_memory_mb'
  ]
  ```

- [ ] **Modification 6: Add variance validation**
  ```python
  def validate_label_variance(paired_df):
      variance_check = paired_df.groupby('workload_type')['balanced_optimal'].agg([
          'count', 'mean', 'std', 'min', 'max'
      ])

      print("\n" + "="*80)
      print("LABEL VARIANCE VALIDATION")
      print("="*80)
      print(variance_check)
      print()

      # Check variance requirements
      failed = []
      for workload, row in variance_check.iterrows():
          if row['mean'] < 0.20 or row['mean'] > 0.80:
              failed.append(f"{workload}: mean={row['mean']:.2f} (should be 0.20-0.80)")
          if row['std'] < 0.30:
              failed.append(f"{workload}: std={row['std']:.2f} (should be >0.30)")
          if row['min'] != 0 or row['max'] != 1:
              failed.append(f"{workload}: range [{row['min']}, {row['max']}] (should be [0, 1])")

      if failed:
          print("⚠️  WARNING: Insufficient variance detected:")
          for msg in failed:
              print(f"   - {msg}")
          print("\nConsider collecting more data with greater variance!")
          return False
      else:
          print("✅ All workloads have sufficient label variance!")
          return True

  # Call before saving
  validate_label_variance(paired_df)
  ```

### Step 3.2: Run Preprocessing

- [ ] Execute preprocessing:
  ```bash
  python variance_preprocessing.py
  ```

- [ ] Monitor output for:
  - [ ] Total requests loaded (~180K)
  - [ ] Successful cost calculations
  - [ ] Paired samples created
  - [ ] Label distribution by workload
  - [ ] Variance validation results

### Step 3.3: Validate Preprocessed Data

- [ ] Check output file exists:
  ```bash
  ls -lh processed-data/ml_training_data_variance_v1.csv
  ```

- [ ] **Critical Validation Checks:**

  ```python
  import pandas as pd
  import numpy as np

  df = pd.read_csv('processed-data/ml_training_data_variance_v1.csv')

  print("="*80)
  print("PREPROCESSING VALIDATION")
  print("="*80)

  # 1. Sample count
  print(f"\nTotal samples: {len(df):,}")
  assert len(df) > 150000, "Too few samples!"

  # 2. Feature columns
  expected_features = [
      'workload_type_encoded', 'payload_size_kb', 'hour_of_day',
      'time_window_encoded', 'load_pattern_encoded', 'is_weekend',
      'lambda_memory_limit_mb', 'payload_squared', 'payload_log',
      'payload_workload_interaction', 'payload_hour_interaction',
      'payload_time_window_interaction', 'workload_time_window_interaction',
      'payload_load_pattern_interaction'
  ]

  for feat in expected_features:
      assert feat in df.columns, f"Missing feature: {feat}"
  print(f"\n✓ All {len(expected_features)} features present")

  # 3. NO forbidden features
  forbidden = ['lambda_latency_ms', 'lambda_cost_usd', 'ecs_latency_ms', 'ecs_cost_usd']
  leaked = [f for f in forbidden if f in df.columns]
  assert len(leaked) == 0, f"DATA LEAKAGE: {leaked}"
  print("✓ No data leakage detected")

  # 4. Payload variance
  print("\nPayload size variance:")
  for workload in df['workload_type'].unique():
      wdf = df[df['workload_type'] == workload]
      unique_payloads = wdf['payload_size_kb'].nunique()
      print(f"  {workload}: {unique_payloads} unique sizes")
      assert unique_payloads >= 4, f"{workload} has too few payload sizes!"
  print("✓ Payload variance sufficient")

  # 5. Label variance
  print("\nLabel variance (balanced_optimal):")
  variance = df.groupby('workload_type')['balanced_optimal'].agg(['mean', 'std'])
  print(variance)

  for workload, row in variance.iterrows():
      assert 0.20 <= row['mean'] <= 0.80, f"{workload} labels too deterministic (mean={row['mean']})"
      assert row['std'] >= 0.30, f"{workload} has too little variance (std={row['std']})"
  print("✓ Label variance sufficient")

  # 6. No missing values
  missing = df[expected_features + ['balanced_optimal']].isnull().sum()
  assert missing.sum() == 0, f"Missing values detected:\n{missing[missing > 0]}"
  print("✓ No missing values")

  print("\n" + "="*80)
  print("✅ PREPROCESSING VALIDATION PASSED!")
  print("="*80)
  ```

- [ ] **If validation passes:** Proceed to training
- [ ] **If validation fails:** Debug preprocessing script, check data collection logs

### Step 3.4: Exploratory Analysis (Optional but Recommended)

- [ ] Create exploratory notebook:
  ```bash
  cd /home/user/ahamed-thesis/ml-variance-experiment/notebooks
  jupyter notebook
  ```

- [ ] Explore:
  - [ ] Feature distributions (histograms)
  - [ ] Correlation matrix
  - [ ] Label distribution by workload, time_window, load_pattern
  - [ ] Payload size distribution
  - [ ] Cold start rates by time_window

---

## Phase 4: Model Training 🤖

**Objective:** Train and evaluate ML models
**Time Estimate:** 3-4 hours
**Directory:** `ml-variance-experiment/training/`

### Step 4.1: Set Up Training Environment

- [ ] Create requirements.txt:
  ```bash
  cd /home/user/ahamed-thesis/ml-variance-experiment/training
  ```

- [ ] Install dependencies:
  ```bash
  pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn shap
  ```

- [ ] Verify installation:
  ```bash
  python -c "import sklearn, xgboost, lightgbm, shap; print('✓ All packages installed')"
  ```

### Step 4.2: Create Training Script

- [ ] Create `train_models.py` or `train_models.ipynb`

- [ ] **Implementation Structure:**

  ```python
  # 1. Load Data
  import pandas as pd
  from sklearn.model_selection import train_test_split

  df = pd.read_csv('../preprocessing/processed-data/ml_training_data_variance_v1.csv')

  # 2. Define Features and Target
  FEATURES = [
      'workload_type_encoded', 'payload_size_kb', 'hour_of_day',
      'time_window_encoded', 'load_pattern_encoded', 'is_weekend',
      'lambda_memory_limit_mb', 'payload_squared', 'payload_log',
      'payload_workload_interaction', 'payload_hour_interaction',
      'payload_time_window_interaction', 'workload_time_window_interaction',
      'payload_load_pattern_interaction'
  ]

  X = df[FEATURES]
  y = df['balanced_optimal']

  # 3. Train/Val/Test Split (60/20/20)
  X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
  X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

  # 4. Train Models
  from sklearn.ensemble import RandomForestClassifier
  from xgboost import XGBClassifier
  from lightgbm import LGBMClassifier
  from sklearn.neural_network import MLPClassifier

  models = {
      'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
      'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
      'LightGBM': LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1),
      'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500, random_state=42)
  }

  results = {}
  for name, model in models.items():
      print(f"\nTraining {name}...")
      model.fit(X_train, y_train)

      train_acc = model.score(X_train, y_train)
      val_acc = model.score(X_val, y_val)
      test_acc = model.score(X_test, y_test)

      results[name] = {
          'model': model,
          'train_acc': train_acc,
          'val_acc': val_acc,
          'test_acc': test_acc,
          'train_test_gap': train_acc - test_acc
      }

      print(f"  Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")

  # 5. Evaluate Best Model
  # (Continue with detailed evaluation...)
  ```

### Step 4.3: Train All Models

- [ ] **Run training:**
  ```bash
  python train_models.py
  # Or in Jupyter:
  jupyter notebook train_models.ipynb
  ```

- [ ] **Monitor training:**
  - [ ] No errors during training
  - [ ] Training completes in reasonable time (<30 min total)
  - [ ] Accuracy printed for each model

### Step 4.4: Evaluate Models

- [ ] **Collect metrics for each model:**
  - [ ] Accuracy (train, val, test)
  - [ ] Precision, Recall, F1
  - [ ] Confusion Matrix
  - [ ] ROC-AUC
  - [ ] Feature Importance

- [ ] **Success Criteria Validation:**

  ```python
  for name, res in results.items():
      print(f"\n=== {name} ===")

      # 1. Accuracy range
      test_acc = res['test_acc']
      if test_acc > 0.95:
          print("⚠️  TOO HIGH! Likely still a lookup table")
      elif test_acc < 0.70:
          print("⚠️  TOO LOW! Data quality issue")
      elif 0.75 <= test_acc <= 0.90:
          print(f"✓ Accuracy: {test_acc:.2%} (target range)")

      # 2. Overfitting check
      gap = res['train_test_gap']
      if gap < 0.05:
          print(f"✓ Generalization: {gap:.2%} (good)")
      else:
          print(f"⚠️  Overfitting: {gap:.2%} (>5% gap)")

      # 3. Feature importance (if available)
      if hasattr(res['model'], 'feature_importances_'):
          importances = res['model'].feature_importances_
          max_importance = max(importances)
          if max_importance < 0.40:
              print(f"✓ Feature diversity: max={max_importance:.2%}")
          else:
              print(f"⚠️  Single feature dominates: {max_importance:.2%}")
  ```

- [ ] **Select best model based on:**
  - [ ] Test accuracy (highest in 75-90% range)
  - [ ] Low train/test gap (<5%)
  - [ ] Distributed feature importance
  - [ ] Inference speed

### Step 4.5: Feature Importance Analysis

- [ ] **For best model, analyze feature importance:**

  ```python
  import matplotlib.pyplot as plt
  import numpy as np

  best_model = results['XGBoost']['model']  # or whichever is best

  importances = best_model.feature_importances_
  indices = np.argsort(importances)[::-1]

  plt.figure(figsize=(10, 6))
  plt.title("Feature Importance")
  plt.bar(range(len(FEATURES)), importances[indices])
  plt.xticks(range(len(FEATURES)), [FEATURES[i] for i in indices], rotation=45, ha='right')
  plt.tight_layout()
  plt.savefig('../analysis/feature_importance.png')
  plt.show()

  # Print importance
  for i in indices:
      print(f"{FEATURES[i]:40s} {importances[i]:.4f}")
  ```

- [ ] **Verify expectations:**
  - [ ] `workload_type_encoded`: 25-35% (important but not dominant)
  - [ ] `payload_size_kb`: 20-30% (strong predictor)
  - [ ] `time_window_encoded`: 10-15% (cold start proxy)
  - [ ] Multiple features contributing (not just top 2)

### Step 4.6: SHAP Analysis (Advanced)

- [ ] **Compute SHAP values:**

  ```python
  import shap

  # Create explainer
  explainer = shap.TreeExplainer(best_model)
  shap_values = explainer.shap_values(X_test)

  # Summary plot
  shap.summary_plot(shap_values, X_test, feature_names=FEATURES, show=False)
  plt.savefig('../analysis/shap_summary.png', bbox_inches='tight', dpi=150)
  plt.show()

  # Feature importance from SHAP
  shap.summary_plot(shap_values, X_test, feature_names=FEATURES, plot_type="bar", show=False)
  plt.savefig('../analysis/shap_importance.png', bbox_inches='tight', dpi=150)
  plt.show()
  ```

- [ ] **Analyze interaction effects:**
  - [ ] Are payload_workload_interaction and similar features contributing?
  - [ ] Do SHAP values show non-linear patterns?
  - [ ] Are there clear decision boundaries visible?

### Step 4.7: Save Best Model

- [ ] **Save model and metadata:**

  ```python
  import joblib
  import json
  from datetime import datetime

  best_model_name = 'XGBoost'  # or whichever performed best
  best_model = results[best_model_name]['model']
  best_metrics = results[best_model_name]

  # Save model
  joblib.dump(best_model, '../models/variance_model_v1.pkl')

  # Save feature columns
  with open('../models/feature_columns.json', 'w') as f:
      json.dump(FEATURES, f, indent=2)

  # Save metadata
  metadata = {
      'model_type': best_model_name,
      'train_accuracy': float(best_metrics['train_acc']),
      'val_accuracy': float(best_metrics['val_acc']),
      'test_accuracy': float(best_metrics['test_acc']),
      'train_test_gap': float(best_metrics['train_test_gap']),
      'num_features': len(FEATURES),
      'feature_columns': FEATURES,
      'training_date': datetime.now().isoformat(),
      'training_samples': len(X_train),
      'test_samples': len(X_test)
  }

  with open('../models/model_metadata.json', 'w') as f:
      json.dump(metadata, f, indent=2)

  print("\n✅ Model saved:")
  print("   - variance_model_v1.pkl")
  print("   - feature_columns.json")
  print("   - model_metadata.json")
  ```

- [ ] **Verify saved files:**
  ```bash
  ls -lh ../models/
  ```

### Step 4.8: Test Inference Locally

- [ ] **Test model loading and prediction:**

  ```python
  import joblib
  import json
  import numpy as np

  # Load model
  loaded_model = joblib.load('../models/variance_model_v1.pkl')

  # Load features
  with open('../models/feature_columns.json') as f:
      feature_columns = json.load(f)

  # Test prediction
  test_input = {
      'workload_type_encoded': 1,  # thumbnail
      'payload_size_kb': 200,
      'hour_of_day': 14,
      'time_window_encoded': 2,  # midday
      'load_pattern_encoded': 1,  # medium
      'is_weekend': 0,
      'lambda_memory_limit_mb': 512,
      'payload_squared': 200**2,
      'payload_log': np.log1p(200),
      'payload_workload_interaction': 1 * 200,
      'payload_hour_interaction': 200 * 14,
      'payload_time_window_interaction': 200 * 2,
      'workload_time_window_interaction': 1 * 2,
      'payload_load_pattern_interaction': 200 * 1
  }

  feature_vector = [test_input[col] for col in feature_columns]
  prediction = loaded_model.predict([feature_vector])[0]
  probability = loaded_model.predict_proba([feature_vector])[0]

  print(f"\nTest Prediction:")
  print(f"  Recommended Platform: {'Lambda' if prediction == 1 else 'ECS'}")
  print(f"  Confidence: {max(probability):.2%}")
  print(f"  Lambda Probability: {probability[1]:.2%}")
  print(f"  ECS Probability: {probability[0]:.2%}")
  ```

- [ ] **Expected result:** Prediction completes in <10ms with reasonable probability distribution

---

## Phase 5: Deployment (Optional) 🚀

**Objective:** Deploy model to AWS Lambda for inference
**Time Estimate:** 3-4 hours
**Directory:** `ml-variance-experiment/deployment/`

### Step 5.1: Package Model for Lambda

- [ ] Create Lambda deployment directory:
  ```bash
  cd /home/user/ahamed-thesis/ml-variance-experiment/deployment
  mkdir lambda-inference
  cd lambda-inference
  ```

- [ ] Create handler.py (see NEW_PROJECT_REQUIREMENTS.md Section 6.1)

- [ ] Install dependencies:
  ```bash
  pip install scikit-learn xgboost numpy -t .
  ```

- [ ] Copy model files:
  ```bash
  cp ../../models/variance_model_v1.pkl .
  cp ../../models/feature_columns.json .
  ```

- [ ] Create deployment package:
  ```bash
  zip -r deployment.zip .
  ```

### Step 5.2: Deploy to AWS Lambda

- [ ] Create Lambda function (if not exists)
- [ ] Upload deployment.zip
- [ ] Configure memory: 512 MB
- [ ] Configure timeout: 10 seconds
- [ ] Test invoke

**Note:** Detailed deployment instructions in NEW_PROJECT_REQUIREMENTS.md Section 6

---

## Phase 6: Documentation & Analysis 📝

**Objective:** Document results for thesis
**Time Estimate:** 2-3 hours

### Step 6.1: Generate Final Report

- [ ] **Create comprehensive results summary:**
  - [ ] Data collection summary (5 test runs, 180K samples)
  - [ ] Preprocessing validation results
  - [ ] Model comparison table
  - [ ] Feature importance analysis
  - [ ] SHAP analysis insights
  - [ ] Best model selection rationale

### Step 6.2: Create Visualizations

- [ ] Feature importance bar chart
- [ ] SHAP summary plot
- [ ] Confusion matrix
- [ ] ROC curve
- [ ] Learning curves (if available)
- [ ] Label distribution by workload

### Step 6.3: Write Thesis Section

- [ ] Introduction: Problem statement (100% accuracy issue)
- [ ] Methodology: Variance experiment design
- [ ] Data Collection: 5 time windows, payload variance
- [ ] Preprocessing: Feature engineering
- [ ] Model Training: 4 models compared
- [ ] Results: Best model performance, feature importance
- [ ] Discussion: Why variance matters, learned patterns
- [ ] Conclusion: Model ready for deployment

---

## Final Validation Checklist ✅

Before considering project complete:

### Data Collection
- [ ] All 5 test runs completed
- [ ] ~180K total requests collected
- [ ] Payload variance confirmed (4-5 sizes per workload)
- [ ] Time windows distributed correctly
- [ ] No missing or corrupted data

### Preprocessing
- [ ] Labels show 20-80% variance per workload
- [ ] No forbidden features included
- [ ] All 14 features present
- [ ] No missing values
- [ ] Variance validation passed

### Model Training
- [ ] Test accuracy: 75-90% ✓
- [ ] Train/test gap: <5% ✓
- [ ] Top feature: <40% importance ✓
- [ ] At least 5 features contribute ✓
- [ ] SHAP analysis shows interactions ✓

### Documentation
- [ ] Training metrics logged
- [ ] Feature importance documented
- [ ] Model saved and tested
- [ ] Thesis section drafted

---

## Troubleshooting Guide 🔧

**Problem: Artillery test fails**
- Check AWS endpoints are active
- Verify network connectivity
- Check artillery.io installation

**Problem: Labels still deterministic**
- Verify variance-functions.js is randomizing
- Check payload sizes in logs
- Ensure time windows are different

**Problem: Model accuracy >95%**
- Check for data leakage (forbidden features)
- Verify labels have variance
- Inspect feature importance

**Problem: Model accuracy <70%**
- Check data quality (missing values, outliers)
- Verify sufficient samples (~180K)
- Check label distribution balance

---

## Quick Reference Commands 📋

```bash
# Navigate to project
cd /home/user/ahamed-thesis/ml-variance-experiment

# Run artillery test
cd artillery-tests && artillery run variance-test-low-load.yml

# Check sample count
wc -l data-output/variance-experiment/*/*.jsonl

# Run preprocessing
cd preprocessing && python variance_preprocessing.py

# Validate data
python -c "import pandas as pd; df = pd.read_csv('preprocessing/processed-data/ml_training_data_variance_v1.csv'); print(df.groupby('workload_type')['balanced_optimal'].agg(['mean', 'std']))"

# Train models
cd training && python train_models.py

# Test model
python -c "import joblib; model = joblib.load('models/variance_model_v1.pkl'); print('Model loaded successfully')"
```

---

**Good luck with the new experiment! 🎯**

Remember: The goal is 75-90% accuracy with distributed feature importance - NOT 100%!
