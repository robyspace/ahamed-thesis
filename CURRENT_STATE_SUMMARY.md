# Current State Summary: ML Training Issues & Root Cause Analysis

**Document Version:** 1.0
**Date:** 2025-11-18
**Status:** Identified fundamental data collection flaw requiring redesign

---

## Executive Summary

The ML model training achieved **100% prediction accuracy** on both training and test sets, which initially appeared successful but was actually a **critical failure**. The model learned a trivial lookup table based on `workload_type` alone, rather than complex decision boundaries involving multiple features.

**Root Cause:** Experimental data lacks meaningful variance. Labels are perfectly deterministic within each workload type due to fixed payload sizes and consistent performance characteristics.

**Resolution:** Requires complete redesign of data collection with payload variance, time-of-day variance, and load pattern variance (see `ML_VARIANCE_EXPERIMENT_PLAN.md`).

---

## 1. Initial Findings (What Seemed Wrong)

### 1.1 Perfect Predictions Across All Models

| Model | Train Accuracy | Test Accuracy | Train/Test Gap |
|-------|----------------|---------------|----------------|
| Random Forest | 100.00% | 100.00% | 0.00% |
| Gradient Boosting | 100.00% | 100.00% | 0.00% |
| Neural Network | 100.00% | 100.00% | 0.00% |
| Logistic Regression | 100.00% | 100.00% | 0.00% |

**Why this is suspicious:**
- Real-world ML problems rarely achieve 100% accuracy
- No difference between simple (Logistic Regression) and complex (Neural Network) models
- Zero overfitting suggests the task is trivial

### 1.2 Identical Feature Importance

**Random Forest Feature Importance:**
```
workload_type_encoded:       49.8%
payload_size_kb:             18.2%
lambda_memory_limit_mb:      12.1%
hour_of_day:                  8.4%
payload_squared:              6.3%
...
```

**Gradient Boosting Feature Importance:**
```
workload_type_encoded:       51.2%
payload_size_kb:             17.9%
lambda_memory_limit_mb:      11.8%
hour_of_day:                  8.1%
payload_squared:              6.5%
...
```

**Observation:** Identical patterns across all models suggest they're all learning the same trivial rule.

---

## 2. Investigation Process

### 2.1 Initial Hypothesis: Data Leakage

**Suspected culprit:** Including actual performance metrics as features

**Features that revealed the answer:**
- `lambda_latency_ms` - Actual Lambda execution time
- `lambda_cost_usd` - Actual Lambda cost
- `ecs_latency_ms` - Actual ECS execution time
- `ecs_cost_usd` - Actual ECS cost

**Labels created from:**
```python
balanced_optimal = 1 if (
    lambda_latency < ecs_latency * 0.9 AND
    lambda_cost < ecs_cost * 0.9
) else 0
```

**Fix attempted:** Removed performance metrics, kept only predictive features.

**Result:** Still 100% accuracy! The problem was deeper.

### 2.2 Second Hypothesis: Workload Type Correlation

**Theory:** Maybe payload size perfectly correlates with workload type?

**Data analysis revealed:**
```
lightweight_api:
  Unique payload values: 1
  Actual values: [1.05 KB]

thumbnail_processing:
  Unique payload values: 2
  Actual values: [200.00 KB, 200.002 KB]  # Essentially identical

medium_processing:
  Unique payload values: 2
  Actual values: [2730.67 KB, 2730.67 KB]  # Identical

heavy_processing:
  Unique payload values: 2
  Actual values: [5461.34 KB, 5461.34 KB]  # Identical
```

**Conclusion:** Payload sizes are effectively CONSTANT within each workload type.

### 2.3 Third Discovery: Perfect Label Separation

**Label distribution by workload:**
```
lightweight_api:      100% ECS    (95,393 samples, 0 variance)
thumbnail_processing: 100% Lambda (72,453 samples, 0 variance)
medium_processing:    100% Lambda (36,478 samples, 0 variance)
heavy_processing:     100% Lambda (7,709 samples, 0 variance)
```

**Translation:** The model learned this simple rule:
```python
def predict_optimal_platform(workload_type_encoded):
    if workload_type_encoded == 0:  # lightweight_api
        return 0  # ECS
    else:
        return 1  # Lambda
```

---

## 3. Root Cause Analysis

### 3.1 The Fundamental Problem

**The ML task was not actually a machine learning problem.**

It was a **deterministic lookup** disguised as a classification task:
- Input: `workload_type` (with optional noise from other features)
- Output: Platform (100% determined by `workload_type`)
- "Learning": Memorize 4 rules

### 3.2 Why Variance Matters

**Machine learning requires:**
1. **Multiple features contributing to the decision** ✗ Failed
2. **Non-deterministic relationships** ✗ Failed
3. **Overlapping decision boundaries** ✗ Failed
4. **Context-dependent outcomes** ✗ Failed

**What we had instead:**
1. Single feature (`workload_type`) determines everything
2. 100% deterministic within each type
3. Perfect separation (no overlap)
4. Context (time, load, payload) has zero effect

### 3.3 Contributing Factors

**1. Fixed Payload Sizes**

Artillery test configuration:
```javascript
// artillery-tests/functions.js.backup
case 'lightweight_api':
    payload.payload = { user: 'test', action: 'validate' };  // Always ~1KB
    break;
case 'thumbnail_processing':
    payload.payload = Buffer.from('x'.repeat(150 * 1024)).toString('base64');  // Always 150KB → ~200KB base64
    break;
```

**Impact:** No variance in the primary driver of performance differences.

**2. Limited Time Variance**

Data collection:
```
hour_of_day:     Only 7 values (0, 1, 19, 20, 21, 22, 23)
day_of_week:     Only 2 values (Monday=0, Sunday=6)
lambda_cold_start: Only 116 cold starts out of 212,033 (0.05%!)
```

**Impact:** Cold start probability (key Lambda penalty) barely varied.

**3. Consistent Load Patterns**

Artillery configuration:
```yaml
phases:
  - duration: 21600  # 6 hours
    arrivalRate: 10  # Constant
    name: "Sustained load"
```

**Impact:** No variation in scaling behavior, concurrency, or resource contention.

**4. Deterministic Performance**

Within each workload type:
- Lambda performance: Nearly identical across all requests
- ECS performance: Nearly identical across all requests
- Result: One platform consistently better than the other

---

## 4. Detailed Data Analysis

### 4.1 Dataset Statistics

**Total Samples:** 212,033

**Workload Distribution:**
```
lightweight_api:       95,393 (45.0%)
thumbnail_processing:  72,453 (34.2%)
medium_processing:     36,478 (17.2%)
heavy_processing:       7,709 (3.6%)
```

### 4.2 Feature Variance Analysis

| Feature | Unique Values | Variance | Useful? |
|---------|---------------|----------|---------|
| `workload_type_encoded` | 4 | High | ✅ (but TOO dominant) |
| `payload_size_kb` | ~8 | **Extremely Low** | ❌ (fixed per workload) |
| `hour_of_day` | 7 | Low | ⚠️ (limited coverage) |
| `day_of_week` | 2 | Very Low | ❌ (weekend vs Monday only) |
| `is_weekend` | 2 | Low | ⚠️ (binary, limited impact) |
| `lambda_memory_limit_mb` | 4 | None | ❌ (same as workload_type) |
| `lambda_cold_start` | 2 | **Extremely Low** | ❌ (0.05% occurrence) |
| `payload_squared` | ~8 | Extremely Low | ❌ (derived from fixed payload) |
| `payload_log` | ~8 | Extremely Low | ❌ (derived from fixed payload) |

**Conclusion:** Only `workload_type_encoded` has meaningful predictive power.

### 4.3 Label Distribution Issues

**Overall:**
```
ECS (0):    95,393 (45.0%)
Lambda (1): 116,640 (55.0%)
```

**Appears balanced**, but this is misleading because it's just reflecting workload type distribution!

**Actual variance within workload types:**
```
lightweight_api:
  ECS:    100.0% (95,393 samples)
  Lambda:   0.0% (0 samples)
  VARIANCE: 0.00

thumbnail_processing:
  ECS:      0.0% (0 samples)
  Lambda: 100.0% (72,453 samples)
  VARIANCE: 0.00

medium_processing:
  ECS:      0.0% (0 samples)
  Lambda: 100.0% (36,478 samples)
  VARIANCE: 0.00

heavy_processing:
  ECS:      0.0% (0 samples)
  Lambda: 100.0% (7,709 samples)
  VARIANCE: 0.00
```

**Expected for ML:**
Each workload should have 20-80% variance in labels depending on context.

---

## 5. What Doesn't Work (Attempted Fixes)

### 5.1 Removing Performance Metrics ❌

**Attempted:** Removed `lambda_latency_ms`, `lambda_cost_usd`, `ecs_latency_ms`, `ecs_cost_usd`

**Result:** Still 100% accuracy

**Why it failed:** The underlying data still has zero variance in labels within workload types.

### 5.2 Adding Interaction Features ❌

**Attempted:** Created features like:
- `workload_payload_interaction`
- `payload_memory_interaction`
- `workload_hour_interaction`

**Result:** Still 100% accuracy

**Why it failed:** Interactions of constant features are still constant!
```
workload_payload_interaction = workload_type × constant_payload
                             = still perfectly correlated with workload_type
```

### 5.3 Percentile-Based Labeling (v6) ❌

**Attempted:** Compare Lambda p25 vs ECS p75, use decision margins

**Result:** Still 100% accuracy

**Why it failed:** Percentile comparisons still deterministic when performance is consistent:
- Lambda always performs in tight range → percentiles very close
- ECS always performs in tight range → percentiles very close
- Comparison result: Always the same within workload type

### 5.4 Feature Engineering ❌

**Attempted:** Added 11 advanced features including:
- `payload_category` (quartile-based)
- `hour_period` (time of day categories)
- `payload_log`, `payload_squared`

**Result:** Still 100% accuracy

**Why it failed:** Transformations of constant values are still constant!

---

## 6. Preprocessing Code Analysis

### 6.1 Current Implementation

**File:** `ml-notebooks/01_data_preprocessing_v6_percentile_labeling.py`

**Issue 1: Fixed payload generation** (lines 110-116)
```python
workload_encoding = {
    'lightweight_api': 0,
    'thumbnail_api': 1,
    'medium_processing': 2,
    'heavy_processing': 3
}
df['workload_type_encoded'] = df['workload_type'].map(workload_encoding)
```

Combined with artillery's fixed payloads → perfect correlation.

**Issue 2: Label creation reveals determinism** (lines 266-272)
```python
# Apply decision margins (Lambda must be SIGNIFICANTLY better)
lambda_faster = actual_lat < (compare_ecs_lat * (1 - LATENCY_MARGIN))
lambda_cheaper = actual_cost < (compare_ecs_cost * (1 - COST_MARGIN))

balanced_optimal = 1 if (lambda_faster and lambda_cheaper) else 0
```

This comparison would work IF there was variance in `actual_lat` and `actual_cost` within workload types. But there isn't!

### 6.2 Data Flow

```
Artillery (fixed payloads)
    ↓
AWS Execution (deterministic performance per workload)
    ↓
JSONL Logs (constant metrics per workload)
    ↓
Preprocessing (labels become deterministic)
    ↓
ML Training (learns lookup table)
    ↓
100% Accuracy (trivial task)
```

---

## 7. Key Learnings

### 7.1 For ML to Work, You Need:

**✅ Variance in Labels**
- Same input conditions should sometimes favor Lambda, sometimes ECS
- Current: 0% variance within workload types
- Target: 30-70% variance

**✅ Variance in Features**
- Payload sizes must vary within each workload type
- Current: 1 unique value per workload
- Target: 4-5 unique values per workload

**✅ Context Matters**
- Time of day should affect cold starts
- Current: 0.05% cold starts (too rare)
- Target: 5-25% cold starts depending on time

**✅ Multiple Features Contribute**
- No single feature should dominate (>40%)
- Current: `workload_type` = 50% importance
- Target: Top feature <30% importance

### 7.2 What Good ML Data Looks Like

**Example: Thumbnail Processing**

| Payload | Time | Load | Cold Start? | Actual Optimal | Why |
|---------|------|------|-------------|----------------|-----|
| 50 KB | 2 AM | Low | Yes (Cold) | **ECS** | Cold start penalty too high |
| 50 KB | 2 PM | High | No (Warm) | **Lambda** | Fast execution, low cost |
| 500 KB | 2 AM | Low | Yes (Cold) | **ECS** | Large payload + cold start |
| 500 KB | 2 PM | High | No (Warm) | **Lambda** | Warm Lambda efficient |
| 1 MB | Any | Any | No | **ECS** | Payload too large for Lambda |

**Variance:** 60% Lambda, 40% ECS (meaningful learning problem!)

### 7.3 Statistical Requirements

For a meaningful ML problem:

**Minimum variance needed:**
- Each workload type: At least 20-80% label split (not 0-100%)
- Payload sizes: At least 4-5 distinct sizes per workload
- Time variance: At least 5 different time windows with different cold start rates
- Load variance: At least 3 different load patterns

**Our current data:**
- Label split: 0-100% or 100-0% ❌
- Payload sizes: 1 per workload ❌
- Time variance: 7 hours, but cold starts still 0.05% ❌
- Load variance: 1 constant pattern ❌

---

## 8. Why This Matters for Thesis

### 8.1 Academic Rigor

**Current state:**
- Model achieves 100% accuracy
- Reviewer asks: "How did you validate the model isn't just memorizing?"
- Answer: "It is memorizing, and that's the problem"

**After variance redesign:**
- Model achieves 80-85% accuracy
- Reviewer asks: "Why not higher?"
- Answer: "Because the task has inherent uncertainty - same workload can favor different platforms depending on context"

### 8.2 Real-World Applicability

**Current model:**
```python
if workload == lightweight_api:
    use ECS
else:
    use Lambda
```
**Value:** Could have been hardcoded. No ML needed.

**Variance-trained model:**
```python
if workload == thumbnail AND payload > 300KB:
    use ECS
elif workload == thumbnail AND time == early_morning:
    use ECS  # cold starts likely
elif workload == thumbnail AND load == burst:
    use Lambda  # handles spikes well
else:
    use Lambda
```
**Value:** Learns nuanced decision boundaries that adapt to context.

### 8.3 Research Contribution

**Current:** "We built a classifier that perfectly predicts... by using workload type"
- Contribution: Minimal (could use if-else statement)

**After variance:** "We built a classifier that learns context-dependent platform selection, achieving 83% accuracy by considering payload size, time-of-day patterns, and load characteristics"
- Contribution: Demonstrates ML can learn complex hybrid cloud optimization

---

## 9. Files & Artifacts

### 9.1 Current Experimental Data

**Location:** `/home/user/ahamed-thesis/ml-notebooks/processed-data/ml_training_data.csv`

**Size:** 212,033 samples

**Status:** ⚠️ Not suitable for ML training (zero variance in labels per workload)

**Recommendation:** Archive but do not use for model training

### 9.2 Analysis Notebooks

**Attempted training notebooks:**
- Original training notebook (100% accuracy, investigated issue)
- Multiple preprocessing versions (v4, v5, v6) - all had same underlying problem

**Status:** Useful for understanding what NOT to do

### 9.3 Preprocessing Scripts

**File:** `ml-notebooks/01_data_preprocessing_v6_percentile_labeling.py`

**Key sections:**
- Lines 195-377: `create_percentile_labels()` - demonstrates the determinism
- Lines 104-124: `engineer_basic_features()` - shows fixed encoding
- Lines 167-193: `add_cost_calculations()` - correct, but data has no variance

---

## 10. Next Steps

### 10.1 Immediate Actions

1. ✅ **Document findings** (this document)
2. ✅ **Create variance experiment plan** (see `ML_VARIANCE_EXPERIMENT_PLAN.md`)
3. ⏳ **Design new artillery tests** with payload variance
4. ⏳ **Collect new data** with 4-5 payload sizes per workload
5. ⏳ **Retrain models** on variance-rich dataset

### 10.2 Success Criteria for New Experiment

**Data collection:**
- [ ] Each workload has 4-5 distinct payload sizes
- [ ] Data collected across 5 different time windows
- [ ] 4 different load patterns tested
- [ ] Cold start rate varies 5-25% across time windows

**Preprocessing:**
- [ ] Labels show 20-80% variance within each workload type
- [ ] No single feature has >40% importance
- [ ] At least 5 features contribute meaningfully

**Model training:**
- [ ] Accuracy: 75-90% (not 95%+)
- [ ] Train/test gap: <5%
- [ ] Feature importance: Distributed across multiple features
- [ ] SHAP analysis: Shows interaction effects

### 10.3 Timeline

See `ML_VARIANCE_EXPERIMENT_PLAN.md` Section 8 for detailed timeline.

**Estimated:** 3 days active work + 1 week validation

---

## 11. Conclusion

The 100% accuracy was a **symptom of a perfectly deterministic dataset**, not a successful ML model. The root cause is **insufficient variance in experimental design**, particularly:

1. Fixed payload sizes per workload type
2. Minimal time-of-day variance
3. Constant load patterns
4. Resulting in 100% deterministic labels within each workload type

**This is actually a valuable finding** - it demonstrates the importance of experimental design in ML. The solution requires collecting new data with intentional variance, which will create a real machine learning problem where models must learn context-dependent decision boundaries.

**Status:** Ready to proceed with variance-focused redesign (see `ML_VARIANCE_EXPERIMENT_PLAN.md`).

---

**Document prepared for:** Starting fresh ML experiment in new thread
**Reference when:** Setting up new project, explaining to thesis advisor, designing future experiments
**Key takeaway:** ML needs variance; perfect accuracy often means trivial task
