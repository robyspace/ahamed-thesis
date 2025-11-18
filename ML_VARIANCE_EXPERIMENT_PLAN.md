# ML Variance Experiment Plan
## Hybrid Serverless-Container Workload Optimization with Meaningful Feature Variance

**Document Version:** 1.0
**Created:** 2025-11-18
**Purpose:** Design a comprehensive ML experiment with sufficient variance to learn nuanced decision boundaries

---

## 1. Problem Statement

### Current Issue
The initial ML experiment achieved 100% prediction accuracy because:
- Labels perfectly correlated with `workload_type_encoded` (single-feature lookup)
- Each workload type had **fixed payload sizes** (no variance)
- Labels were 100% deterministic within each workload type:
  - `lightweight_api`: 100% ECS optimal
  - `thumbnail_processing`: 100% Lambda optimal
  - `medium_processing`: 100% Lambda optimal
  - `heavy_processing`: 100% Lambda optimal

### Root Cause
The model learned a trivial lookup table instead of complex decision boundaries:
```python
if workload_type == lightweight_api:
    return ECS
else:
    return Lambda
```

### Objective
Create a **real machine learning problem** where:
- The same workload type can favor Lambda OR ECS depending on context (payload size, load, time of day)
- Models must learn nuanced patterns beyond simple workload classification
- Decision boundaries involve multiple features interacting in non-trivial ways

---

## 2. Experimental Design

### 2.1 Payload Variance Strategy

**For EACH workload type, test MULTIPLE payload sizes:**

| Workload Type | Current Fixed Size | New Variance Sizes | Purpose |
|---------------|-------------------|-------------------|---------|
| `lightweight_api` | 1 KB | **0.5 KB, 1 KB, 5 KB, 10 KB** | Create scenarios where small requests favor Lambda, larger ones favor ECS |
| `thumbnail_processing` | 200 KB | **50 KB, 100 KB, 200 KB, 500 KB, 1 MB** | Explore cold start penalty at different sizes |
| `medium_processing` | 2.7 MB | **1 MB, 2 MB, 3 MB, 5 MB** | Find crossover point where Lambda becomes too expensive |
| `heavy_processing` | 5.4 MB | **3 MB, 5 MB, 8 MB, 10 MB** | Test Lambda memory/timeout limits |

**Expected Variance Outcome:**
- Small payloads: Lambda wins (fast startup, low cost)
- Medium payloads: Depends on cold start probability
- Large payloads: ECS wins (sustained compute efficiency)

### 2.2 Time-of-Day Variance Strategy

**Constraint:** Cannot run over multiple days, but CAN run at different times of day.

**Test Windows (Each 60-90 minutes):**

| Time Window | Expected Characteristic | Lambda Behavior | ECS Behavior |
|-------------|------------------------|-----------------|--------------|
| **Early Morning (2-4 AM)** | Cold infrastructure | High cold starts (>20%) | Slow instance startup |
| **Morning Peak (8-10 AM)** | Warming up | Medium cold starts (~10%) | Instances already warm |
| **Midday (12-2 PM)** | Peak load | Low cold starts (<5%) | Fully scaled, efficient |
| **Evening (6-8 PM)** | Second peak | Low cold starts | Stable performance |
| **Late Night (10-12 PM)** | Cooling down | Increasing cold starts | Auto-scaling down |

**Why This Matters:**
- Cold starts significantly impact Lambda latency (2-5x slower)
- ECS tasks may be pre-scaled or need startup time
- Creates variance in the SAME workload+payload combination

### 2.3 Load Intensity Variance

**Artillery.io Load Patterns:**

```yaml
# Pattern 1: Low Load (Baseline)
phases:
  - duration: 900  # 15 minutes
    arrivalRate: 5
    name: "Low load - baseline performance"

# Pattern 2: Medium Load
phases:
  - duration: 900
    arrivalRate: 15
    name: "Medium load - typical usage"

# Pattern 3: Burst Load
phases:
  - duration: 600
    arrivalRate: 5
  - duration: 300
    arrivalRate: 30  # Spike
  - duration: 300
    arrivalRate: 5   # Cool down
    name: "Burst pattern - stress test"

# Pattern 4: Ramp Load
phases:
  - duration: 900
    arrivalRate: 5
    rampTo: 25
    name: "Gradual ramp - scaling test"
```

**Expected Impact:**
- Low load: More cold starts, favors Lambda for sporadic work
- Medium load: Lambda stays warm, performs well
- Burst load: ECS handles sustained load better
- Ramp load: Tests auto-scaling behavior

---

## 3. Artillery.io Test Implementation

### 3.1 Modified `functions.js`

**Key Changes:**
- Generate **randomized payload sizes** within defined ranges
- Track actual payload size in metadata
- Log more context (time window, load pattern)

```javascript
// Payload variance configuration
const PAYLOAD_CONFIGS = {
  lightweight_api: {
    sizes_kb: [0.5, 1, 5, 10],
    baseContent: { user: 'test', action: 'validate' }
  },
  thumbnail_processing: {
    sizes_kb: [50, 100, 200, 500, 1024],
    baseContent: 'IMAGE_DATA'
  },
  medium_processing: {
    sizes_kb: [1024, 2048, 3072, 5120],
    baseContent: 'PROCESSING_DATA'
  },
  heavy_processing: {
    sizes_kb: [3072, 5120, 8192, 10240],
    baseContent: 'HEAVY_DATA'
  }
};

function generatePayloadWithVariance(workloadType) {
  const config = PAYLOAD_CONFIGS[workloadType];

  // Randomly select payload size
  const targetSizeKB = config.sizes_kb[
    Math.floor(Math.random() * config.sizes_kb.length)
  ];

  // Generate payload to match target size
  const payload = {
    request_id: generateRequestId(),
    workload_type: workloadType,
    timestamp: new Date().toISOString(),
    target_size_kb: targetSizeKB,
    time_window: getCurrentTimeWindow(),  // NEW: Track time period
    load_pattern: process.env.LOAD_PATTERN || 'unknown'  // NEW: Track load type
  };

  // Add sized content
  if (workloadType === 'lightweight_api') {
    payload.payload = {
      ...config.baseContent,
      data: generateRandomData(targetSizeKB)
    };
  } else {
    // For processing workloads, generate base64 data
    payload.payload = generateBase64Data(targetSizeKB);
  }

  return payload;
}

function getCurrentTimeWindow() {
  const hour = new Date().getHours();
  if (hour >= 2 && hour < 4) return 'early_morning';
  if (hour >= 8 && hour < 10) return 'morning_peak';
  if (hour >= 12 && hour < 14) return 'midday';
  if (hour >= 18 && hour < 20) return 'evening';
  if (hour >= 22 || hour < 1) return 'late_night';
  return 'other';
}
```

### 3.2 New Artillery Test Scenario

**File:** `ml-variance-experiment/artillery-tests/variance-test.yml`

```yaml
config:
  target: "https://jt67vt5uwj.execute-api.eu-west-1.amazonaws.com/prod"
  processor: "./variance-functions.js"
  phases:
    - duration: 3600  # 1 hour per test run
      arrivalRate: {{ arrivalRate }}  # Configured per load pattern
      name: "{{ loadPattern }}"
  variables:
    ecs_base: "http://hybrid-thesis-alb-811686247.eu-west-1.elb.amazonaws.com"
    load_pattern: "{{ loadPattern }}"

scenarios:
  # EQUAL distribution across all workload types and platforms
  # This ensures balanced data collection

  - name: "Lightweight Lambda - Varied Payloads"
    weight: 12.5
    flow:
      - function: "setLightweightVariancePayload"
      - post:
          url: "/lightweight"
          json: "{{ payload }}"
          afterResponse: "logDetailedResponse"

  - name: "Lightweight ECS - Varied Payloads"
    weight: 12.5
    flow:
      - function: "setLightweightVariancePayload"
      - post:
          url: "{{ ecs_base }}/lightweight/process"
          json: "{{ payload }}"
          afterResponse: "logDetailedResponse"

  # ... (repeat for thumbnail, medium, heavy with equal weights)
```

### 3.3 Test Execution Schedule

**Required Test Runs (Can be done on same day):**

| Run # | Time Window | Duration | Load Pattern | Arrival Rate | Expected Samples |
|-------|-------------|----------|--------------|--------------|------------------|
| 1 | 2-3 AM | 60 min | Low Load | 5 req/s | ~18,000 |
| 2 | 8-9 AM | 60 min | Ramp Load | 5→25 req/s | ~54,000 |
| 3 | 12-1 PM | 60 min | Medium Load | 15 req/s | ~54,000 |
| 4 | 6-7 PM | 60 min | Burst Load | 5/30/5 req/s | ~36,000 |
| 5 | 10-11 PM | 60 min | Low Load | 5 req/s | ~18,000 |

**Total Expected Samples:** ~180,000 requests

**Payload Distribution Per Run:**
- Each workload type: ~25% of traffic
- Each payload size: ~20-25% within workload
- Platform split: 50/50 Lambda vs ECS

**Expected Label Variance:**
- `lightweight_api`: 40-60% Lambda optimal (varies with payload size)
- `thumbnail_processing`: 30-70% Lambda optimal (varies with cold starts)
- `medium_processing`: 20-80% Lambda optimal (varies with load)
- `heavy_processing`: 10-90% Lambda optimal (varies with payload size)

---

## 4. Data Collection & Preprocessing

### 4.1 Enhanced Logging

**Add to log entries:**
```json
{
  "request_id": "req_123",
  "platform": "lambda",
  "workload_type": "thumbnail_processing",
  "timestamp": "2025-11-18T14:23:45.123Z",
  "target_payload_size_kb": 500,
  "actual_payload_size_kb": 512.34,
  "time_window": "midday",
  "load_pattern": "medium_load",
  "http_status": 200,
  "response_time_ms": 245.67,
  "metrics": {
    "execution_time_ms": 234.12,
    "memory_used_mb": 256,
    "cold_start": false,
    "cpu_percent": 45.2
  }
}
```

### 4.2 Preprocessing Changes

**File:** `ml-variance-experiment/preprocessing/variance_preprocessing.py`

**Key Differences from v6:**

1. **Keep only predictive features (NO actual performance metrics):**
   ```python
   FEATURE_COLUMNS = [
       'workload_type_encoded',     # 0-3
       'payload_size_kb',            # NOW VARIES! (0.5 to 10,240)
       'hour_of_day',                # 0-23
       'time_window_encoded',        # NEW: early_morning=0, morning_peak=1, etc.
       'load_pattern_encoded',       # NEW: low=0, medium=1, burst=2, ramp=3
       'is_weekend',                 # 0/1
       'lambda_memory_limit_mb',     # Config-based

       # Interaction features
       'payload_workload_interaction',
       'payload_squared',
       'payload_log',
       'payload_hour_interaction',
       'workload_time_window_interaction',
       'payload_load_pattern_interaction'
   ]
   ```

2. **REMOVE these features (they reveal the answer):**
   ```python
   # ❌ DO NOT INCLUDE:
   # - lambda_latency_ms
   # - lambda_cost_usd
   # - ecs_latency_ms
   # - ecs_cost_usd
   # - lambda_cold_start (actual event)
   ```

3. **Label creation uses actual performance:**
   ```python
   # Labels still compare actual performance
   # But models will NOT see these metrics
   balanced_optimal = 1 if (
       lambda_latency < ecs_latency * 0.9 AND
       lambda_cost < ecs_cost * 0.9
   ) else 0
   ```

### 4.3 Expected Feature Distributions

**After variance introduction:**

```
payload_size_kb:
  Min: 0.5 KB
  Max: 10,240 KB
  Unique values: ~20 (5 per workload × 4 workloads)
  Distribution: Uniform within each workload

time_window:
  Unique values: 5
  Distribution: Depends on test schedule

load_pattern:
  Unique values: 4
  Distribution: 20% each (5 test runs)

balanced_optimal (labels):
  Expected: 30-70% Lambda optimal (VARIES by context!)
```

---

## 5. Model Training Strategy

### 5.1 Train/Test Split

**Temporal split recommended:**
- Training: Runs 1-3 (early morning, morning peak, midday)
- Validation: Run 4 (evening burst)
- Test: Run 5 (late night)

**Why temporal?**
- Tests model's ability to generalize to new time periods
- More realistic than random split
- Prevents data leakage from same time window

### 5.2 Model Architectures to Test

1. **Random Forest** (Baseline)
   - Should achieve 70-85% accuracy
   - Good for understanding feature importance

2. **Gradient Boosting (XGBoost/LightGBM)**
   - Expected: 75-90% accuracy
   - Handles interaction effects well

3. **Neural Network (2-3 hidden layers)**
   - Expected: 75-88% accuracy
   - Can learn complex non-linear patterns

4. **Ensemble (Voting Classifier)**
   - Expected: 80-92% accuracy
   - Combines strengths of all models

### 5.3 Success Criteria

**Model is considered successful if:**
- ✅ Accuracy: 75-90% (NOT 95-100% which suggests overfitting)
- ✅ Feature importance: Multiple features contribute (not just workload_type)
- ✅ Decision boundaries: Non-trivial (verified with SHAP analysis)
- ✅ Generalization: <5% gap between train/test accuracy
- ✅ Balanced: Similar precision/recall for both classes

**Model is considered FAILED if:**
- ❌ Accuracy > 95% (likely still a lookup table)
- ❌ Single feature dominates (>70% importance)
- ❌ Perfect separation by workload_type
- ❌ Train/test gap > 10%

---

## 6. Expected Outcomes

### 6.1 Feature Importance (Target)

```
Expected feature importance distribution:
  workload_type_encoded:       25-35%  (Important but not dominant)
  payload_size_kb:             20-30%  (Strong predictor)
  time_window_encoded:         10-15%  (Cold start proxy)
  load_pattern_encoded:        8-12%   (Scaling behavior)
  payload_workload_interaction: 8-12%  (Key interaction)
  hour_of_day:                 5-10%   (Context)
  Other features:              10-15%  (Remaining)
```

### 6.2 Learning Curves

**Expected patterns:**
- Training accuracy: Start at 60%, plateau at 78-82%
- Validation accuracy: Follow training closely (gap <3%)
- Loss: Smooth decrease, no overfitting

### 6.3 Decision Boundaries

**Example learned patterns:**
```
IF workload_type == lightweight_api:
    IF payload_size_kb > 7 KB:
        → Predict ECS (60% confidence)
    ELSE:
        → Predict Lambda (70% confidence)

IF workload_type == thumbnail_processing:
    IF time_window == early_morning AND payload_size_kb > 300:
        → Predict ECS (cold starts hurt Lambda)
    ELSE IF payload_size_kb < 150:
        → Predict Lambda (fast execution)
    ELSE:
        → Depends on load_pattern...
```

---

## 7. Deployment to AWS

### 7.1 Model Selection

After training, select the best model based on:
1. Test accuracy (target: 80-85%)
2. Feature usage diversity (no single feature >40% importance)
3. Inference speed (<10ms)
4. Model size (<50MB)

### 7.2 Deployment Architecture

**Option 1: Lambda-based Inference (Recommended)**
```
API Gateway → Lambda (Model Inference) → DynamoDB (Decision Log)
                ↓
        Route to Lambda/ECS based on prediction
```

**Option 2: ECS-based Inference (High Volume)**
```
ALB → ECS (Model Service) → Redis (Prediction Cache)
        ↓
  Route requests accordingly
```

### 7.3 A/B Testing

**Test the ML model against baseline:**
- Control: Static workload-based routing (current approach)
- Treatment: ML-based routing
- Metrics: Compare actual cost and latency over 1 week

---

## 8. Project Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Setup** | 1 day | Updated Artillery scripts, variance functions |
| **Data Collection** | 1 day | 5 test runs at different times, ~180K samples |
| **Preprocessing** | 2 hours | Cleaned dataset with variance features |
| **Model Training** | 4 hours | 4 models trained, evaluated, compared |
| **Analysis** | 2 hours | Feature importance, SHAP analysis, documentation |
| **Deployment** | 1 day | Package model, deploy to AWS, A/B test setup |
| **Validation** | 1 week | Real-world testing, performance comparison |

**Total:** ~3 days active work + 1 week validation

---

## 9. Key Differences from Previous Attempt

| Aspect | Previous (v6) | New (Variance) |
|--------|---------------|----------------|
| **Payload Sizes** | Fixed per workload | 4-5 sizes per workload |
| **Time Variance** | Random times | Controlled time windows |
| **Load Patterns** | Constant | 4 distinct patterns |
| **Label Distribution** | 100% deterministic | 30-70% varies by context |
| **Features Used** | Included performance metrics | Only predictive features |
| **Expected Accuracy** | 100% (trivial) | 75-90% (meaningful) |
| **Learning Task** | Lookup table | Complex decision boundaries |

---

## 10. References

**Existing Codebase:**
- `artillery-tests/dual-platform-test.yml` - Current test configuration
- `artillery-tests/functions.js.backup` - Current payload generation
- `ecs-app/app.py` - ECS workload implementation
- `lambda-functions/app.py` - Lambda workload implementation
- `ml-notebooks/01_data_preprocessing_v6_percentile_labeling.py` - Current preprocessing

**New Files to Create:**
- `ml-variance-experiment/artillery-tests/variance-test.yml`
- `ml-variance-experiment/artillery-tests/variance-functions.js`
- `ml-variance-experiment/preprocessing/variance_preprocessing.py`
- `ml-variance-experiment/training/train_models.py`
- `ml-variance-experiment/deployment/model_service.py`

---

**Next Steps:** See `NEW_PROJECT_REQUIREMENTS.md` for technical implementation details.
