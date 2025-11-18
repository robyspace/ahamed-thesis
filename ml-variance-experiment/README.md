# ML Variance Experiment
## Hybrid Serverless-Container Workload Optimization with Meaningful Variance

**Project Status:** Ready for Data Collection
**Created:** 2025-11-18
**Thesis Chapter:** Machine Learning Model Development

---

## Project Overview

This experiment redesigns the ML data collection process to introduce **meaningful variance** in features and labels. The goal is to train models that learn **context-dependent decision boundaries** rather than trivial lookup tables.

### Why This Experiment?

The initial ML attempt (in `../ml-notebooks/`) achieved 100% accuracy because:
- Labels perfectly correlated with `workload_type` (0% variance within each type)
- Payload sizes were fixed per workload type
- No variation in time-of-day or load patterns

This experiment fixes those issues by:
- ✅ Testing 4-5 different payload sizes per workload type
- ✅ Collecting data across 5 different time windows
- ✅ Using 4 different load patterns (low, medium, burst, ramp)
- ✅ Creating scenarios where the same workload can favor Lambda OR ECS

**Expected Result:** Models achieve 75-90% accuracy with feature importance distributed across multiple features.

---

## Quick Start

### Prerequisites

- Artillery.io installed
- Python 3.8+
- Node.js (for Artillery)
- Existing AWS infrastructure (Lambda + ECS endpoints already deployed)

### Step 1: Review Documentation

Read these documents (in order):

1. **[CURRENT_STATE_SUMMARY.md](../CURRENT_STATE_SUMMARY.md)** - What we learned from the first attempt
2. **[ML_VARIANCE_EXPERIMENT_PLAN.md](../ML_VARIANCE_EXPERIMENT_PLAN.md)** - Complete experimental design
3. **[NEW_PROJECT_REQUIREMENTS.md](../NEW_PROJECT_REQUIREMENTS.md)** - Technical implementation details
4. **[PROJECT_CHECKLIST.md](./PROJECT_CHECKLIST.md)** - Step-by-step execution guide

### Step 2: Set Up Environment

```bash
# Navigate to project directory
cd ml-variance-experiment

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify Artillery is installed
artillery --version
```

### Step 3: Implement Artillery Scripts

Create the variance-based payload generation:

```bash
cd artillery-tests
# Implement variance-functions.js (see NEW_PROJECT_REQUIREMENTS.md Section 3.2)
# Create 4 test configuration files (see Section 3.3)
# Create orchestration script (see Section 3.4)
```

### Step 4: Collect Data

Run tests at designated times:

| Test # | Time Window | Config File | Duration |
|--------|-------------|-------------|----------|
| 1 | 2-3 AM | variance-test-low-load.yml | 60 min |
| 2 | 8-9 AM | variance-test-ramp-load.yml | 60 min |
| 3 | 12-1 PM | variance-test-medium-load.yml | 60 min |
| 4 | 6-7 PM | variance-test-burst-load.yml | 60 min |
| 5 | 10-11 PM | variance-test-low-load.yml | 60 min |

```bash
cd artillery-tests
./run-variance-tests.sh
```

### Step 5: Preprocess Data

```bash
cd preprocessing
python variance_preprocessing.py

# Verify variance
python -c "
import pandas as pd
df = pd.read_csv('../processed-data/ml_training_data_variance_v1.csv')
print(df.groupby('workload_type')['balanced_optimal'].agg(['mean', 'std', 'min', 'max']))
"
```

### Step 6: Train Models

```bash
cd training
jupyter notebook train_models.ipynb

# Or run Python script
python train_models.py
```

### Step 7: Deploy & Validate

```bash
cd deployment
python package_model.py
# Follow AWS deployment instructions in NEW_PROJECT_REQUIREMENTS.md Section 6
```

---

## Directory Structure

```
ml-variance-experiment/
├── README.md                          # This file
├── PROJECT_CHECKLIST.md               # Detailed execution checklist
├── requirements.txt                   # Python dependencies
│
├── artillery-tests/                   # Data collection
│   ├── variance-functions.js          # Payload generation with variance
│   ├── variance-test-low-load.yml     # Low load pattern config
│   ├── variance-test-medium-load.yml  # Medium load pattern config
│   ├── variance-test-burst-load.yml   # Burst load pattern config
│   ├── variance-test-ramp-load.yml    # Ramp load pattern config
│   └── run-variance-tests.sh          # Test orchestration script
│
├── data-output/                       # Raw JSONL logs
│   └── variance-experiment/
│       ├── early_morning_low_load/
│       ├── morning_peak_ramp_load/
│       ├── midday_medium_load/
│       ├── evening_burst_load/
│       └── late_night_low_load/
│
├── preprocessing/                     # Data preprocessing
│   ├── variance_preprocessing.py      # Main preprocessing script
│   ├── processed-data/                # Output CSV files
│   └── exploratory_analysis.ipynb     # Data validation notebook
│
├── training/                          # Model training
│   ├── train_models.py                # Training script
│   ├── train_models.ipynb             # Training notebook (Jupyter)
│   └── training_logs/                 # Training history
│
├── models/                            # Saved models
│   ├── variance_model_v1.pkl          # Best model
│   ├── feature_columns.json           # Feature names
│   └── model_metadata.json            # Performance metrics
│
├── deployment/                        # AWS deployment
│   ├── lambda-inference/              # Lambda function for inference
│   ├── package_model.py               # Model packaging script
│   └── test_inference.py              # Local testing
│
├── analysis/                          # Results analysis
│   ├── feature_importance.ipynb       # Feature analysis
│   ├── shap_analysis.ipynb            # SHAP values
│   └── ab_test_results.ipynb          # A/B test comparison
│
└── notebooks/                         # Exploratory notebooks
    └── variance_exploration.ipynb     # Data exploration
```

---

## Key Differences from Previous Experiment

| Aspect | Previous (ml-notebooks) | New (ml-variance-experiment) |
|--------|-------------------------|------------------------------|
| **Payload Sizes** | Fixed (1 per workload) | Variable (4-5 per workload) |
| **Time Windows** | Random | 5 controlled windows |
| **Load Patterns** | Constant | 4 distinct patterns |
| **Expected Accuracy** | 100% (trivial) | 75-90% (meaningful) |
| **Label Variance** | 0% within workload | 20-80% within workload |
| **Learning Task** | Lookup table | Complex decision boundaries |
| **Features Used** | Included performance metrics | Only predictive features |
| **Top Feature Importance** | 50% (workload_type) | <30% (distributed) |

---

## Data Collection Summary

### Payload Variance Configuration

| Workload | Previous | New Sizes | Purpose |
|----------|----------|-----------|---------|
| lightweight_api | 1 KB | 0.5, 1, 5, 10 KB | Find crossover point |
| thumbnail_processing | 200 KB | 50, 100, 200, 500, 1024 KB | Cold start impact varies |
| medium_processing | 2.7 MB | 1, 2, 3, 5 MB | Lambda efficiency boundary |
| heavy_processing | 5.4 MB | 3, 5, 8, 10 MB | Memory/timeout limits |

### Test Schedule

| Run | Time | Load Pattern | Arrival Rate | Expected Samples |
|-----|------|--------------|--------------|------------------|
| 1 | 2-3 AM | Low | 5/s | 18,000 |
| 2 | 8-9 AM | Ramp | 5→25/s | 54,000 |
| 3 | 12-1 PM | Medium | 15/s | 54,000 |
| 4 | 6-7 PM | Burst | 5/30/5/s | 36,000 |
| 5 | 10-11 PM | Low | 5/s | 18,000 |

**Total Expected:** ~180,000 requests

---

## Feature Engineering

### Predictive Features (Used in Model)

```python
FEATURES = [
    'workload_type_encoded',           # 0-3
    'payload_size_kb',                 # 0.5 to 10,240 (NOW VARIES!)
    'hour_of_day',                     # 0-23
    'time_window_encoded',             # 0-4 (NEW)
    'load_pattern_encoded',            # 0-3 (NEW)
    'is_weekend',                      # 0/1
    'lambda_memory_limit_mb',          # Config-based
    'payload_squared',                 # Engineered
    'payload_log',                     # Engineered
    'payload_workload_interaction',    # Engineered
    'payload_hour_interaction',        # Engineered
    'workload_time_window_interaction',# Engineered
    'payload_load_pattern_interaction' # Engineered
]
# Total: 14 features
```

### Forbidden Features (Data Leakage)

```python
FORBIDDEN = [
    'lambda_latency_ms',    # ❌ Reveals answer
    'lambda_cost_usd',      # ❌ Reveals answer
    'ecs_latency_ms',       # ❌ Reveals answer
    'ecs_cost_usd',         # ❌ Reveals answer
    'lambda_cold_start',    # ❌ Actual event
]
```

---

## Model Training Approach

### Models to Train

1. **Random Forest** (Baseline) - Expected: 70-80%
2. **XGBoost** (Primary) - Expected: 75-85%
3. **LightGBM** (Fast) - Expected: 75-85%
4. **Neural Network** (Deep Learning) - Expected: 73-83%
5. **Ensemble** (Best) - Expected: 78-88%

### Success Criteria

✅ **Acceptable Model:**
- Accuracy: 75-90%
- Train/test gap: <5%
- Top feature: <40% importance
- At least 5 features contribute

❌ **Failed Model:**
- Accuracy: >95% (still a lookup table)
- Accuracy: <70% (data quality issues)
- Train/test gap: >10% (overfitting)
- Single feature: >50% importance

### Validation Strategy

- **Train/Val/Test Split:** 60% / 20% / 20%
- **Split Method:** Temporal (by test run) or Stratified Random
- **Cross-Validation:** 5-fold for hyperparameter tuning
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC

---

## Expected Results

### Feature Importance (Target Distribution)

```
workload_type_encoded:       25-35%  ✅ Important but not dominant
payload_size_kb:             20-30%  ✅ Strong predictor
time_window_encoded:         10-15%  ✅ Cold start proxy
load_pattern_encoded:        8-12%   ✅ Scaling behavior
payload_workload_interaction: 8-12%  ✅ Key interaction
Other features:              10-20%  ✅ Remaining
```

### Label Distribution (Expected Variance)

```
Overall: 40-60% Lambda optimal

By Workload:
  lightweight_api:       20-50% Lambda (varies with payload)
  thumbnail_processing:  30-70% Lambda (varies with cold starts)
  medium_processing:     40-80% Lambda (varies with load)
  heavy_processing:      40-80% Lambda (varies with payload size)
```

### Example Learned Patterns

```
IF workload == lightweight_api AND payload > 7KB:
    → ECS (60% confidence)

IF workload == thumbnail AND time == early_morning AND payload > 300KB:
    → ECS (cold starts hurt Lambda)

IF workload == heavy AND payload > 8MB:
    → ECS (Lambda memory limit)

ELSE IF workload == thumbnail AND load == burst AND payload < 200KB:
    → Lambda (handles spikes well)
```

---

## Troubleshooting

### Issue: Artillery test fails

**Symptoms:** Connection errors, timeouts

**Solution:**
```bash
# Verify AWS endpoints are active
curl https://jt67vt5uwj.execute-api.eu-west-1.amazonaws.com/prod/lightweight
curl http://hybrid-thesis-alb-811686247.eu-west-1.elb.amazonaws.com/lightweight/process
```

### Issue: Labels still deterministic

**Symptoms:** 100% or 0% labels for a workload type

**Diagnosis:**
```python
import pandas as pd
df = pd.read_csv('processed-data/ml_training_data_variance_v1.csv')

# Check payload variance
print(df.groupby('workload_type')['payload_size_kb'].unique())

# Check label variance
print(df.groupby('workload_type')['balanced_optimal'].mean())
```

**Solution:** Verify variance-functions.js is randomizing payload sizes correctly

### Issue: Model accuracy >95%

**Symptoms:** Perfect or near-perfect accuracy

**Diagnosis:**
```python
# Check for data leakage
forbidden = ['lambda_latency_ms', 'lambda_cost_usd', 'ecs_latency_ms', 'ecs_cost_usd']
leakage = [col for col in df.columns if col in forbidden]
print(f"Leaked features: {leakage}")
```

**Solution:** Remove performance metrics from feature set

### Issue: Model accuracy <70%

**Symptoms:** Poor performance

**Diagnosis:**
```python
# Check data quality
print(df.isnull().sum())
print(df.describe())
print(df['balanced_optimal'].value_counts())
```

**Solution:** Check for missing values, outliers, or class imbalance

---

## References & Documentation

**Planning Documents:**
- [ML_VARIANCE_EXPERIMENT_PLAN.md](../ML_VARIANCE_EXPERIMENT_PLAN.md) - Complete experimental design
- [CURRENT_STATE_SUMMARY.md](../CURRENT_STATE_SUMMARY.md) - Lessons learned from first attempt
- [NEW_PROJECT_REQUIREMENTS.md](../NEW_PROJECT_REQUIREMENTS.md) - Technical specifications

**Existing Codebase (Reference Only):**
- `../artillery-tests/` - Original test configurations
- `../lambda-functions/` - Lambda implementations
- `../ecs-app/` - ECS implementations
- `../ml-notebooks/` - Previous ML attempt (archive)

**New Codebase (Active Development):**
- `./artillery-tests/` - Variance-based tests
- `./preprocessing/` - Updated preprocessing
- `./training/` - Model training
- `./deployment/` - AWS deployment

---

## Timeline

**Day 1: Setup**
- [ ] Create Artillery scripts with variance
- [ ] Test payload generation locally
- [ ] Run Test 1 (Early Morning - 2-3 AM)

**Day 2: Data Collection**
- [ ] Run Test 2 (Morning Peak - 8-9 AM)
- [ ] Run Test 3 (Midday - 12-1 PM)
- [ ] Run Test 4 (Evening - 6-7 PM)
- [ ] Run Test 5 (Late Night - 10-11 PM)

**Day 3: Preprocessing & Training**
- [ ] Preprocess all collected data
- [ ] Validate variance requirements
- [ ] Train all models
- [ ] Select best model

**Day 4: Deployment**
- [ ] Package model for Lambda
- [ ] Deploy to AWS
- [ ] Test inference endpoint
- [ ] Set up A/B test

**Week 2: Validation**
- [ ] Run A/B test
- [ ] Collect performance metrics
- [ ] Document results
- [ ] Write thesis section

---

## Success Indicators

**Data Collection Successful When:**
- ✅ All 5 test runs completed
- ✅ ~180K requests collected
- ✅ Logs contain all required fields (target_payload_size_kb, time_window, load_pattern)
- ✅ Each workload has 4-5 unique payload sizes

**Preprocessing Successful When:**
- ✅ Each workload shows 20-80% label variance
- ✅ No forbidden features included
- ✅ 14 predictive features present
- ✅ No missing values

**Training Successful When:**
- ✅ Accuracy: 75-90%
- ✅ Train/test gap: <5%
- ✅ Feature diversity: Top feature <40%
- ✅ SHAP analysis shows interactions

**Deployment Successful When:**
- ✅ Model inference: <10ms
- ✅ Model size: <50MB
- ✅ API responds correctly
- ✅ A/B test shows cost/latency improvements

---

## Contact & Support

**For New Thread:**

When starting the new thread, provide:

```
I'm continuing my thesis ML experiment with a variance-focused redesign.

CONTEXT:
- Previous attempt: 100% accuracy (trivial lookup table)
- Root cause: Fixed payload sizes, no variance in labels
- Solution: Redesigned data collection with variance

GOAL:
- Collect 180K samples with varying payloads, times, and loads
- Train models achieving 75-90% accuracy
- Deploy best model to AWS for testing

REFERENCE DOCUMENTS:
- /home/user/ahamed-thesis/ML_VARIANCE_EXPERIMENT_PLAN.md
- /home/user/ahamed-thesis/CURRENT_STATE_SUMMARY.md
- /home/user/ahamed-thesis/NEW_PROJECT_REQUIREMENTS.md
- /home/user/ahamed-thesis/ml-variance-experiment/PROJECT_CHECKLIST.md

CURRENT STEP:
See PROJECT_CHECKLIST.md for detailed execution plan
```

---

**Good luck with the experiment! 🚀**

Remember: If you achieve 100% accuracy again, something is still wrong. Aim for 75-90% with diverse feature importance!
