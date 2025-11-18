# New Project Technical Requirements
## ML Variance Experiment - Technical Implementation Specifications

**Document Version:** 1.0
**Created:** 2025-11-18
**Target Audience:** Implementation in new Claude Code thread

---

## 1. Project Overview

**Goal:** Collect variance-rich experimental data and train ML models that learn context-dependent platform selection.

**Success Metric:** Achieve 75-90% accuracy with feature importance distributed across multiple features (no single feature >40%).

**Timeline:** 3 days active work + 1 week validation

---

## 2. Infrastructure Requirements

### 2.1 Existing AWS Resources (Reuse)

**Lambda Functions:**
- API Gateway: `https://jt67vt5uwj.execute-api.eu-west-1.amazonaws.com/prod`
- Endpoints: `/lightweight`, `/thumbnail`, `/medium`, `/heavy`
- No changes needed to Lambda code

**ECS Cluster:**
- ALB: `http://hybrid-thesis-alb-811686247.eu-west-1.elb.amazonaws.com`
- Endpoints: `/lightweight/process`, `/thumbnail/process`, `/medium/process`, `/heavy/process`
- No changes needed to ECS code

**Verification:**
```bash
# Test Lambda
curl -X POST https://jt67vt5uwj.execute-api.eu-west-1.amazonaws.com/prod/lightweight \
  -H "Content-Type: application/json" \
  -d '{"request_id": "test", "workload_type": "lightweight_api", "payload": {"test": "data"}}'

# Test ECS
curl -X POST http://hybrid-thesis-alb-811686247.eu-west-1.elb.amazonaws.com/lightweight/process \
  -H "Content-Type: application/json" \
  -d '{"request_id": "test", "workload_type": "lightweight_api", "payload": {"test": "data"}}'
```

### 2.2 New Components Required

**Local Machine:**
- Node.js (for Artillery.io) - Already have
- Python 3.8+ (for preprocessing and training) - Already have
- Artillery.io - Already installed
- Sufficient disk space for logs (~2-3 GB for 180K requests)

**No new AWS resources needed!** Use existing Lambda and ECS deployments.

---

## 3. Artillery.io Configuration

### 3.1 Directory Structure

```
ml-variance-experiment/
├── artillery-tests/
│   ├── variance-functions.js          # NEW: Payload generation with variance
│   ├── variance-test-low-load.yml     # NEW: Low load pattern
│   ├── variance-test-medium-load.yml  # NEW: Medium load pattern
│   ├── variance-test-burst-load.yml   # NEW: Burst load pattern
│   ├── variance-test-ramp-load.yml    # NEW: Ramp load pattern
│   └── run-variance-tests.sh          # NEW: Orchestration script
└── data-output/
    └── variance-experiment/            # NEW: Separate from old data
        ├── early_morning_2am/
        ├── morning_peak_8am/
        ├── midday_12pm/
        ├── evening_6pm/
        └── late_night_10pm/
```

### 3.2 variance-functions.js Implementation

**File:** `ml-variance-experiment/artillery-tests/variance-functions.js`

**Key Requirements:**

1. **Randomized payload sizes** within defined ranges
2. **Track metadata** (target size, time window, load pattern)
3. **Enhanced logging** with all relevant context
4. **Base64 encoding** for processing workloads (consistent with current implementation)

**Payload Size Configurations:**

```javascript
const PAYLOAD_VARIANCE_CONFIGS = {
  lightweight_api: {
    sizes_kb: [0.5, 1, 5, 10],
    distribution: 'uniform',  // Equal probability
    baseContent: { user: 'test', action: 'validate' }
  },
  thumbnail_processing: {
    sizes_kb: [50, 100, 200, 500, 1024],
    distribution: 'uniform',
    generateBase64: true
  },
  medium_processing: {
    sizes_kb: [1024, 2048, 3072, 5120],
    distribution: 'uniform',
    generateBase64: true
  },
  heavy_processing: {
    sizes_kb: [3072, 5120, 8192, 10240],
    distribution: 'uniform',
    generateBase64: true
  }
};
```

**Implementation Template:**

```javascript
const fs = require('fs');
const path = require('path');

// Configuration
const PAYLOAD_VARIANCE_CONFIGS = { /* as above */ };

// Generate unique request ID
function generateRequestId() {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Determine current time window
function getCurrentTimeWindow() {
  const hour = new Date().getHours();
  if (hour >= 2 && hour < 4) return 'early_morning';
  if (hour >= 8 && hour < 10) return 'morning_peak';
  if (hour >= 12 && hour < 14) return 'midday';
  if (hour >= 18 && hour < 20) return 'evening';
  if (hour >= 22 || hour < 1) return 'late_night';
  return 'other';
}

// Generate random data of specified size
function generateRandomData(sizeKB) {
  const sizeBytes = Math.floor(sizeKB * 1024);
  return 'x'.repeat(sizeBytes);
}

// Generate base64-encoded data of specified size
function generateBase64Data(targetSizeKB) {
  // Base64 encoding increases size by ~33%, so adjust input size
  const inputSizeKB = targetSizeKB / 1.33;
  const data = generateRandomData(inputSizeKB);
  return Buffer.from(data).toString('base64');
}

// Main payload generation with variance
function generateVariancePayload(workloadType, loadPattern) {
  const config = PAYLOAD_VARIANCE_CONFIGS[workloadType];

  // Randomly select payload size
  const targetSizeKB = config.sizes_kb[
    Math.floor(Math.random() * config.sizes_kb.length)
  ];

  const requestId = generateRequestId();
  const timeWindow = getCurrentTimeWindow();

  let payload = {
    request_id: requestId,
    workload_type: workloadType,
    timestamp: new Date().toISOString(),
    metadata: {
      target_size_kb: targetSizeKB,
      time_window: timeWindow,
      load_pattern: loadPattern
    }
  };

  // Add workload-specific payload
  if (workloadType === 'lightweight_api') {
    payload.payload = {
      ...config.baseContent,
      data: generateRandomData(targetSizeKB)
    };
  } else {
    // For processing workloads, use base64 data
    payload.payload = generateBase64Data(targetSizeKB);
  }

  return payload;
}

// Artillery hook functions
function setLightweightVariancePayload(requestParams, context, ee, next) {
  const loadPattern = context.vars.load_pattern || 'unknown';
  context.vars.payload = generateVariancePayload('lightweight_api', loadPattern);
  return next();
}

function setThumbnailVariancePayload(requestParams, context, ee, next) {
  const loadPattern = context.vars.load_pattern || 'unknown';
  context.vars.payload = generateVariancePayload('thumbnail_processing', loadPattern);
  return next();
}

function setMediumVariancePayload(requestParams, context, ee, next) {
  const loadPattern = context.vars.load_pattern || 'unknown';
  context.vars.payload = generateVariancePayload('medium_processing', loadPattern);
  return next();
}

function setHeavyVariancePayload(requestParams, context, ee, next) {
  const loadPattern = context.vars.load_pattern || 'unknown';
  context.vars.payload = generateVariancePayload('heavy_processing', loadPattern);
  return next();
}

// Enhanced response logging
function logDetailedResponse(requestParams, response, context, ee, next) {
  if (response.body) {
    try {
      const data = JSON.parse(response.body);
      const requestPayload = context.vars.payload;

      const logEntry = {
        request_id: data.request_id,
        platform: data.platform,
        workload_type: data.workload_type,
        timestamp: data.timestamp,
        http_status: response.statusCode,
        response_time_ms: response.timings.phases.total,

        // Enhanced metadata
        target_payload_size_kb: requestPayload.metadata?.target_size_kb,
        actual_payload_size_kb: data.metrics?.payload_size_kb,
        time_window: requestPayload.metadata?.time_window,
        load_pattern: requestPayload.metadata?.load_pattern,

        // Performance metrics
        metrics: data.metrics
      };

      // Create timestamped log file with time window
      const timeWindow = requestPayload.metadata?.time_window || 'unknown';
      const loadPattern = requestPayload.metadata?.load_pattern || 'unknown';
      const date = new Date().toISOString().split('T')[0];

      const logDir = path.join(__dirname, '../data-output/variance-experiment', `${timeWindow}_${loadPattern}`);

      // Ensure directory exists
      if (!fs.existsSync(logDir)) {
        fs.mkdirSync(logDir, { recursive: true });
      }

      const logFile = path.join(logDir, `${data.platform}_${data.workload_type}_${date}.jsonl`);
      fs.appendFileSync(logFile, JSON.stringify(logEntry) + '\n');

    } catch (e) {
      console.error('Error logging response:', e.message);
    }
  }
  return next();
}

module.exports = {
  setLightweightVariancePayload,
  setThumbnailVariancePayload,
  setMediumVariancePayload,
  setHeavyVariancePayload,
  logDetailedResponse
};
```

### 3.3 Artillery Test Configurations

**Template for all load patterns:**

```yaml
# variance-test-low-load.yml
config:
  target: "https://jt67vt5uwj.execute-api.eu-west-1.amazonaws.com/prod"
  processor: "./variance-functions.js"
  phases:
    - duration: 3600  # 1 hour
      arrivalRate: 5
      name: "Low load baseline"
  variables:
    ecs_base: "http://hybrid-thesis-alb-811686247.eu-west-1.elb.amazonaws.com"
    load_pattern: "low_load"

scenarios:
  # Equal weight distribution: 12.5% each (8 scenarios × 12.5% = 100%)

  - name: "Lightweight Lambda - Variance"
    weight: 12.5
    flow:
      - function: "setLightweightVariancePayload"
      - post:
          url: "/lightweight"
          json: "{{ payload }}"
          afterResponse: "logDetailedResponse"

  - name: "Lightweight ECS - Variance"
    weight: 12.5
    flow:
      - function: "setLightweightVariancePayload"
      - post:
          url: "{{ ecs_base }}/lightweight/process"
          json: "{{ payload }}"
          afterResponse: "logDetailedResponse"

  - name: "Thumbnail Lambda - Variance"
    weight: 12.5
    flow:
      - function: "setThumbnailVariancePayload"
      - post:
          url: "/thumbnail"
          json: "{{ payload }}"
          afterResponse: "logDetailedResponse"

  - name: "Thumbnail ECS - Variance"
    weight: 12.5
    flow:
      - function: "setThumbnailVariancePayload"
      - post:
          url: "{{ ecs_base }}/thumbnail/process"
          json: "{{ payload }}"
          afterResponse: "logDetailedResponse"

  - name: "Medium Lambda - Variance"
    weight: 12.5
    flow:
      - function: "setMediumVariancePayload"
      - post:
          url: "/medium"
          json: "{{ payload }}"
          afterResponse: "logDetailedResponse"

  - name: "Medium ECS - Variance"
    weight: 12.5
    flow:
      - function: "setMediumVariancePayload"
      - post:
          url: "{{ ecs_base }}/medium/process"
          json: "{{ payload }}"
          afterResponse: "logDetailedResponse"

  - name: "Heavy Lambda - Variance"
    weight: 12.5
    flow:
      - function: "setHeavyVariancePayload"
      - post:
          url: "/heavy"
          json: "{{ payload }}"
          afterResponse: "logDetailedResponse"

  - name: "Heavy ECS - Variance"
    weight: 12.5
    flow:
      - function: "setHeavyVariancePayload"
      - post:
          url: "{{ ecs_base }}/heavy/process"
          json: "{{ payload }}"
          afterResponse: "logDetailedResponse"
```

**Variations for other load patterns:**

```yaml
# variance-test-medium-load.yml
  phases:
    - duration: 3600
      arrivalRate: 15
  variables:
    load_pattern: "medium_load"

# variance-test-burst-load.yml
  phases:
    - duration: 600
      arrivalRate: 5
    - duration: 300
      arrivalRate: 30
    - duration: 300
      arrivalRate: 5
  variables:
    load_pattern: "burst_load"

# variance-test-ramp-load.yml
  phases:
    - duration: 3600
      arrivalRate: 5
      rampTo: 25
  variables:
    load_pattern: "ramp_load"
```

### 3.4 Test Orchestration Script

**File:** `ml-variance-experiment/artillery-tests/run-variance-tests.sh`

```bash
#!/bin/bash

set -e

echo "=========================================="
echo "ML VARIANCE EXPERIMENT - DATA COLLECTION"
echo "=========================================="
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data-output/variance-experiment"

# Create data directory
mkdir -p "${DATA_DIR}"

# Function to run a test
run_test() {
  local test_name=$1
  local config_file=$2
  local time_window=$3

  echo ""
  echo "=========================================="
  echo "Running: ${test_name}"
  echo "Time: $(date)"
  echo "Config: ${config_file}"
  echo "Expected window: ${time_window}"
  echo "=========================================="
  echo ""

  # Run artillery test
  cd "${SCRIPT_DIR}"
  artillery run "${config_file}"

  echo ""
  echo "✅ Completed: ${test_name}"
  echo ""
}

# Display schedule
echo "📅 TEST SCHEDULE:"
echo ""
echo "1. Early Morning (2-3 AM)   → Low Load"
echo "2. Morning Peak (8-9 AM)    → Ramp Load"
echo "3. Midday (12-1 PM)         → Medium Load"
echo "4. Evening (6-7 PM)         → Burst Load"
echo "5. Late Night (10-11 PM)    → Low Load"
echo ""
echo "⚠️  IMPORTANT: Run each test during its designated time window!"
echo ""

# Check current time and suggest which test to run
current_hour=$(date +%H)
current_hour=$((10#$current_hour))  # Remove leading zero

if [ $current_hour -ge 2 ] && [ $current_hour -lt 4 ]; then
  echo "🕐 Current time suggests: Test 1 (Early Morning - Low Load)"
  suggested_test="variance-test-low-load.yml"
elif [ $current_hour -ge 8 ] && [ $current_hour -lt 10 ]; then
  echo "🕐 Current time suggests: Test 2 (Morning Peak - Ramp Load)"
  suggested_test="variance-test-ramp-load.yml"
elif [ $current_hour -ge 12 ] && [ $current_hour -lt 14 ]; then
  echo "🕐 Current time suggests: Test 3 (Midday - Medium Load)"
  suggested_test="variance-test-medium-load.yml"
elif [ $current_hour -ge 18 ] && [ $current_hour -lt 20 ]; then
  echo "🕐 Current time suggests: Test 4 (Evening - Burst Load)"
  suggested_test="variance-test-burst-load.yml"
elif [ $current_hour -ge 22 ] || [ $current_hour -lt 1 ]; then
  echo "🕐 Current time suggests: Test 5 (Late Night - Low Load)"
  suggested_test="variance-test-low-load.yml"
else
  echo "⚠️  Current time (${current_hour}:00) is not in a designated test window"
  echo "    Please run tests during scheduled windows for best variance"
  suggested_test=""
fi

echo ""
read -p "Press Enter to start test, or Ctrl+C to cancel..."

if [ -n "$suggested_test" ]; then
  run_test "Variance Test" "${suggested_test}" "auto-detected"
else
  echo "Please specify which test to run manually"
  echo "Usage: ./run-variance-tests.sh <config-file>"
fi

echo ""
echo "=========================================="
echo "Test completed!"
echo "Data saved to: ${DATA_DIR}"
echo "=========================================="
```

**Usage:**

```bash
# Make executable
chmod +x run-variance-tests.sh

# Run at appropriate time (script auto-detects)
./run-variance-tests.sh

# Or specify manually
artillery run variance-test-low-load.yml
```

---

## 4. Data Preprocessing Requirements

### 4.1 Input Data Format

**Expected JSONL structure:**

```json
{
  "request_id": "req_1731945823456_abc123",
  "platform": "lambda",
  "workload_type": "thumbnail_processing",
  "timestamp": "2025-11-18T14:23:45.123Z",
  "http_status": 200,
  "response_time_ms": 245.67,
  "target_payload_size_kb": 500,
  "actual_payload_size_kb": 512.34,
  "time_window": "midday",
  "load_pattern": "medium_load",
  "metrics": {
    "execution_time_ms": 234.12,
    "memory_used_mb": 256,
    "cold_start": false,
    "cpu_percent": 45.2,
    "payload_size_kb": 512.34
  }
}
```

### 4.2 Preprocessing Script

**File:** `ml-variance-experiment/preprocessing/variance_preprocessing.py`

**Requirements:**

1. Load all JSONL files from variance experiment directories
2. Calculate costs (Lambda and ECS) using same formulas as v6
3. Create paired comparison dataset
4. **ONLY include predictive features** (NO performance metrics)
5. Generate labels based on actual performance comparison
6. Validate variance in labels per workload type

**Feature Schema:**

```python
PREDICTIVE_FEATURES = [
    # Workload characteristics
    'workload_type_encoded',      # 0-3 (categorical)
    'payload_size_kb',             # 0.5 to 10,240 (continuous)

    # Temporal features
    'hour_of_day',                 # 0-23 (categorical)
    'time_window_encoded',         # 0-4 (early_morning, morning_peak, etc.)
    'is_weekend',                  # 0/1 (binary)

    # Load context
    'load_pattern_encoded',        # 0-3 (low, medium, burst, ramp)

    # Configuration
    'lambda_memory_limit_mb',      # Fixed per workload type

    # Engineered features
    'payload_squared',             # payload^2
    'payload_log',                 # log(1 + payload)
    'payload_workload_interaction',
    'payload_hour_interaction',
    'payload_time_window_interaction',
    'workload_time_window_interaction',
    'payload_load_pattern_interaction'
]

# Total: 14 features

LABEL_COLUMNS = [
    'cost_optimal',           # 1 if Lambda cheaper
    'latency_optimal',        # 1 if Lambda faster
    'balanced_optimal'        # 1 if both (primary target)
]
```

**CRITICAL: Do NOT include these features:**

```python
FORBIDDEN_FEATURES = [
    'lambda_latency_ms',     # ❌ Reveals the answer
    'lambda_cost_usd',       # ❌ Reveals the answer
    'ecs_latency_ms',        # ❌ Reveals the answer
    'ecs_cost_usd',          # ❌ Reveals the answer
    'lambda_cold_start',     # ❌ Actual event (can predict probability instead)
    'lambda_memory_mb',      # ❌ Actual usage (only include limit)
]
```

### 4.3 Label Validation

**Before proceeding to training, verify:**

```python
# Check variance within each workload type
variance_check = paired_df.groupby('workload_type')['balanced_optimal'].agg([
    'mean', 'std', 'min', 'max'
])

print("Label Variance Check:")
print(variance_check)

# REQUIREMENT: Each workload should have:
# - mean: 0.2 to 0.8 (20-80% Lambda optimal)
# - std: > 0.3 (meaningful variance)
# - min: 0 (some ECS samples)
# - max: 1 (some Lambda samples)

# If any workload has mean close to 0 or 1, STOP and collect more variance
```

### 4.4 Expected Output

**File:** `ml-variance-experiment/processed-data/ml_training_data_variance_v1.csv`

**Sample count:** ~180,000 rows

**Columns:** 14 features + 3 labels + metadata (workload_type, request_id, etc.)

**Validation metrics:**

```
Total samples: ~180,000
Train/Val/Test split: 60% / 20% / 20%

Label distribution (balanced_optimal):
  Overall: 40-60% Lambda optimal

  By workload:
    lightweight_api:       20-50% Lambda optimal
    thumbnail_processing:  30-70% Lambda optimal
    medium_processing:     40-80% Lambda optimal
    heavy_processing:      40-80% Lambda optimal

Feature distributions:
  payload_size_kb: 20 unique values (4-5 per workload)
  time_window: 5 unique values
  load_pattern: 4 unique values
  hour_of_day: ~15-20 unique values
```

---

## 5. Model Training Requirements

### 5.1 Training Environment

**Platform:** Google Colab (free tier sufficient) or local Jupyter

**Required Libraries:**

```python
# requirements.txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
matplotlib>=3.6.0
seaborn>=0.12.0
shap>=0.41.0
imbalanced-learn>=0.10.0
```

**Installation:**

```bash
pip install -r requirements.txt
```

### 5.2 Training Script Structure

**File:** `ml-variance-experiment/training/train_models.py`

**Phases:**

1. **Data Loading & Validation**
   - Load processed CSV
   - Validate variance requirements
   - Check for data leakage

2. **Train/Val/Test Split**
   - Temporal split (recommended) OR stratified random
   - 60% / 20% / 20%
   - Ensure balanced labels in each split

3. **Baseline Model**
   - Random Forest (100 trees)
   - No hyperparameter tuning
   - Establish performance floor

4. **Advanced Models**
   - XGBoost (with cross-validation)
   - LightGBM (with early stopping)
   - Neural Network (2-3 layers)

5. **Ensemble**
   - Voting classifier (soft voting)
   - Combine top 3 models

6. **Evaluation**
   - Accuracy, Precision, Recall, F1
   - Feature importance analysis
   - SHAP value analysis
   - Confusion matrices

### 5.3 Model Hyperparameters

**Random Forest (Baseline):**

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=100,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1
)
```

**XGBoost (Optimized):**

```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
```

**LightGBM (Fast):**

```python
LGBMClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)
```

**Neural Network:**

```python
MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=500,
    random_state=42
)
```

### 5.4 Success Criteria

**Model passes validation if:**

```python
# Accuracy: Good but not perfect
assert 0.75 <= test_accuracy <= 0.90, "Accuracy outside expected range"

# Generalization: Small train/test gap
train_test_gap = train_accuracy - test_accuracy
assert train_test_gap < 0.05, "Overfitting detected"

# Feature diversity: No single feature dominates
max_feature_importance = max(feature_importances.values())
assert max_feature_importance < 0.40, "Single feature dominates"

# Class balance: Similar performance on both classes
precision_gap = abs(precision_lambda - precision_ecs)
assert precision_gap < 0.15, "Class imbalance detected"
```

### 5.5 Output Artifacts

**Save for deployment:**

```python
# Best model
joblib.dump(best_model, 'models/variance_model_v1.pkl')

# Feature names (in order)
json.dump(feature_columns, open('models/feature_columns.json', 'w'))

# Performance metrics
json.dump({
    'test_accuracy': test_accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1_score,
    'feature_importance': feature_importances,
    'training_date': datetime.now().isoformat()
}, open('models/model_metadata.json', 'w'))
```

---

## 6. Deployment Requirements

### 6.1 Model Serving Options

**Option A: Lambda Function (Recommended for thesis)**

**Advantages:**
- Fast deployment
- Low cost for thesis demo
- Serverless (no infrastructure management)

**Implementation:**

```python
# lambda-model-inference/handler.py
import json
import joblib
import numpy as np

# Load model at cold start
model = joblib.load('variance_model_v1.pkl')
feature_columns = json.load(open('feature_columns.json'))

def lambda_handler(event, context):
    # Parse request
    request_features = event['features']

    # Validate features
    feature_vector = [request_features[col] for col in feature_columns]

    # Predict
    prediction = model.predict([feature_vector])[0]
    probability = model.predict_proba([feature_vector])[0]

    # Return recommendation
    return {
        'statusCode': 200,
        'body': json.dumps({
            'recommended_platform': 'lambda' if prediction == 1 else 'ecs',
            'confidence': float(max(probability)),
            'lambda_probability': float(probability[1]),
            'ecs_probability': float(probability[0])
        })
    }
```

**Deployment:**

```bash
# Package model with dependencies
cd lambda-model-inference
pip install -r requirements.txt -t .
zip -r deployment.zip .

# Deploy via AWS CLI
aws lambda create-function \
  --function-name hybrid-thesis-ml-router \
  --runtime python3.9 \
  --handler handler.lambda_handler \
  --zip-file fileb://deployment.zip \
  --role arn:aws:iam::ACCOUNT_ID:role/lambda-execution-role \
  --memory-size 512 \
  --timeout 10
```

**Option B: ECS Container (For high-volume production)**

See `ML_VARIANCE_EXPERIMENT_PLAN.md` Section 7.2 for details.

### 6.2 A/B Testing Framework

**Goal:** Compare ML routing vs static routing

**Metrics to track:**

```python
METRICS = {
    'cost': 'Total AWS cost (Lambda + ECS)',
    'latency_p50': 'Median latency',
    'latency_p95': '95th percentile latency',
    'error_rate': 'HTTP 5xx errors',
    'routing_accuracy': 'How often ML chose optimal platform (post-hoc)'
}
```

**Test setup:**

```
Control Group (50% traffic):
  → Use static routing (current workload-based logic)

Treatment Group (50% traffic):
  → Use ML model routing

Duration: 1 week
Sample size: ~500K requests minimum
```

---

## 7. Python Environment Setup

### 7.1 Required Packages

```
# requirements.txt
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
xgboost==1.7.5
lightgbm==3.3.5
matplotlib==3.7.1
seaborn==0.12.2
shap==0.41.0
imbalanced-learn==0.10.1
jupyter==1.0.0
notebook==6.5.4
```

### 7.2 Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import sklearn, xgboost, lightgbm, shap; print('✅ All packages installed')"
```

---

## 8. Testing & Validation Checklist

Before considering the experiment complete:

**Data Collection:**
- [ ] All 5 test runs completed successfully
- [ ] ~180K total requests collected
- [ ] Logs contain all required fields
- [ ] No missing or corrupted data

**Preprocessing:**
- [ ] Labels show 20-80% variance per workload type
- [ ] Payload sizes have 4-5 unique values per workload
- [ ] Time windows distributed across all 5 periods
- [ ] No forbidden features included

**Model Training:**
- [ ] Accuracy: 75-90%
- [ ] Train/test gap: <5%
- [ ] Top feature importance: <40%
- [ ] At least 5 features contribute meaningfully
- [ ] SHAP analysis shows interaction effects

**Deployment:**
- [ ] Model packaged and tested locally
- [ ] Inference time: <10ms
- [ ] Model size: <50MB
- [ ] API endpoint responds correctly

**Documentation:**
- [ ] Training metrics logged
- [ ] Feature importance documented
- [ ] Model limitations identified
- [ ] Thesis section drafted

---

## 9. Timeline & Dependencies

**Day 1: Setup & First Collection**
- Morning: Create new project structure
- Afternoon: Implement variance-functions.js
- Evening: Create all 4 Artillery configs
- Night (2-3 AM): Run Test 1 (Early Morning - Low Load)

**Day 2: Continue Collection**
- Morning (8-9 AM): Run Test 2 (Morning Peak - Ramp Load)
- Midday (12-1 PM): Run Test 3 (Midday - Medium Load)
- Afternoon: Start preprocessing script development
- Evening (6-7 PM): Run Test 4 (Evening - Burst Load)

**Day 2 (continued):**
- Night (10-11 PM): Run Test 5 (Late Night - Low Load)
- Late night: Finish preprocessing

**Day 3: Training & Analysis**
- Morning: Validate preprocessed data, fix issues
- Afternoon: Train all models, evaluate
- Evening: SHAP analysis, documentation
- Night: Select best model, prepare deployment

**Day 4: Deployment**
- Package model
- Deploy to Lambda
- Test inference endpoint
- Set up A/B test framework

**Week 2: Validation**
- Run A/B test
- Collect performance metrics
- Document results
- Write thesis section

---

## 10. Quick Reference Commands

**Start new test run:**
```bash
cd ml-variance-experiment/artillery-tests
./run-variance-tests.sh
```

**Preprocess data:**
```bash
cd ml-variance-experiment/preprocessing
python variance_preprocessing.py
```

**Train models:**
```bash
cd ml-variance-experiment/training
python train_models.py
# Or use Jupyter:
jupyter notebook train_models.ipynb
```

**Test model locally:**
```bash
python -c "
import joblib
import json

model = joblib.load('models/variance_model_v1.pkl')
features = json.load(open('models/feature_columns.json'))

# Example prediction
test_input = {
    'workload_type_encoded': 1,
    'payload_size_kb': 200,
    'hour_of_day': 14,
    # ... etc
}

prediction = model.predict([list(test_input.values())])[0]
print(f'Recommended: {'Lambda' if prediction == 1 else 'ECS'}')
"
```

---

## 11. Troubleshooting

**Issue: Artillery test fails with network errors**
- Solution: Check AWS endpoints are still active
- Verify: `curl` tests in Section 2.1

**Issue: Logs missing fields**
- Solution: Verify variance-functions.js implementation
- Check: Enhanced logging section in logDetailedResponse()

**Issue: Labels still 100% deterministic**
- Solution: Verify payload variance is working
- Check: `SELECT DISTINCT payload_size_kb FROM data GROUP BY workload_type`

**Issue: Model accuracy >95%**
- Solution: Feature leakage - check FORBIDDEN_FEATURES not included
- Verify: Print df.columns and ensure no performance metrics

**Issue: Model accuracy <70%**
- Solution: Data quality issues or insufficient samples
- Check: Label distribution, missing values, outliers

---

**Document End**

**Next Steps:**
1. Review this document
2. Review `ML_VARIANCE_EXPERIMENT_PLAN.md`
3. Review `CURRENT_STATE_SUMMARY.md`
4. Start new thread with reference to these documents
5. Begin implementation following Day 1 timeline
