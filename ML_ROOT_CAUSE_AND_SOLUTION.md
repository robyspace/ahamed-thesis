# ML Model Training - ROOT CAUSE ANALYSIS & FINAL SOLUTION

## 🔴 THE REAL PROBLEM DISCOVERED

After extensive investigation, we discovered the ROOT CAUSE of why all three ML models were producing 100% identical predictions.

### Problem Evolution

**Stage 1: Initial Issue**
- All models produced identical predictions
- `workload_type_encoded` dominated with 35.63% importance in XGBoost
- Models learned simple rule: `lightweight_api → ECS, others → Lambda`

**Stage 2: First Fix Attempt** ❌ FAILED
- Removed `workload_type_encoded` and its direct interactions
- Result: **STILL 100% identical predictions!**
- Models had same accuracy (74.75%) with same predictions on every sample

**Stage 3: Root Cause Discovery** ✅ FOUND IT!
- **`lambda_memory_limit_mb` is a PERFECT PROXY for `workload_type`!**
- When we removed `workload_type_encoded`, models just switched to `lambda_memory_limit_mb`

## The Hidden Proxy

Looking at the preprocessing script (`01_data_preprocessing_v6_percentile_labeling.py`):

```python
# Lines 41-46
LAMBDA_MEMORY_CONFIGS = {
    'lightweight_api': 128,           # Always 128 MB
    'thumbnail_processing': 512,       # Always 512 MB
    'medium_processing': 1024,         # Always 1024 MB
    'heavy_processing': 2048          # Always 2048 MB
}

# Line 129
df['lambda_memory_limit_mb'] = df['workload_type'].map(LAMBDA_MEMORY_CONFIGS)
```

**This creates a PERFECT 1-to-1 mapping:**
- 128 MB = lightweight_api (100% correlation)
- 512 MB = thumbnail_processing (100% correlation)
- 1024 MB = medium_processing (100% correlation)
- 2048 MB = heavy_processing (100% correlation)

### Why First Fix Failed

When we removed `workload_type_encoded` but kept `lambda_memory_limit_mb`:
- Models just learned to use memory config instead
- Since memory perfectly predicts workload type, result was identical
- Features derived from memory also became proxies:
  - `memory_payload_interaction` = lambda_memory_limit_mb × payload
  - `payload_memory_ratio` = payload / lambda_memory_limit_mb

## FINAL SOLUTION

### All Workload-Derived Features Removed (7 total)

1. ❌ **`workload_type_encoded`** - Direct encoding of workload type
2. ❌ **`lambda_memory_limit_mb`** ← **THE HIDDEN CULPRIT!**
3. ❌ **`workload_payload_interaction`** - workload × payload
4. ❌ **`workload_memory_interaction`** - workload × memory
5. ❌ **`workload_hour_interaction`** - workload × hour
6. ❌ **`memory_payload_interaction`** - Depends on #2
7. ❌ **`payload_memory_ratio`** - Depends on #2

### Final Feature Set (8 PURE features)

**Payload Features (4)**:
- ✅ `payload_size_kb` - Actual payload size
- ✅ `payload_squared` - Non-linear transformation
- ✅ `payload_log` - Log transformation
- ✅ `payload_category` - Binned categories (0-3)

**Time Features (4)**:
- ✅ `hour_of_day` - Hour (0-23)
- ✅ `day_of_week` - Day (0-6)
- ✅ `is_weekend` - Weekend flag
- ✅ `hour_period` - Time period (0-3)

### Model-Specific Subsets

**Random Forest (6 features)** - Payload-focused:
```python
['payload_log', 'payload_category', 'payload_size_kb',
 'payload_squared', 'hour_of_day', 'hour_period']
```

**XGBoost (6 features)** - Non-linear focused:
```python
['payload_squared', 'payload_log', 'payload_category',
 'hour_period', 'day_of_week', 'is_weekend']
```

**Neural Network (8 features)** - All features:
```python
['payload_size_kb', 'payload_squared', 'payload_log', 'payload_category',
 'hour_of_day', 'day_of_week', 'is_weekend', 'hour_period']
```

## Expected Results After Fix

### Before (ALL Attempts)
```
✗ RF == XGBoost: 31,805 (100.0%)
✗ RF == NN:      31,805 (100.0%)
✗ All models: EXACTLY same accuracy (0.7475)
✗ Predictions by workload:
  - lightweight_api:      0% Lambda (always ECS)
  - thumbnail_processing: 100% Lambda
  - medium_processing:    100% Lambda
  - heavy_processing:     100% Lambda
```

### After (Expected)
```
✓ RF == XGBoost: ~24,000-27,000 (75-85%)
✓ RF == NN:      ~22,000-26,000 (70-82%)
✓ Models have DIFFERENT accuracies
✓ Predictions by workload show VARIANCE:
  - lightweight_api:      20-60% Lambda (varies!)
  - thumbnail_processing: 70-95% Lambda (varies!)
  - medium_processing:    60-85% Lambda (varies!)
  - heavy_processing:     55-80% Lambda (varies!)
✓ Feature importance balanced (no feature >30%)
```

## Why This Fix Will Work

1. **No Workload Proxy**: Zero features perfectly correlate with workload_type
2. **Pure Request Characteristics**: Only payload size and timing remain
3. **Forces Real Learning**: Models MUST learn from actual request patterns
4. **Within-Workload Variance**: Same workload can have different predictions based on payload/time
5. **Model Diversity**: Different models will learn different patterns

## Verification Checklist

After training with FINAL notebook, verify:

- [ ] Models produce DIFFERENT predictions (70-90% agreement, not 100%)
- [ ] Within each workload, predictions vary (not all 0% or 100%)
- [ ] Feature importance is balanced (no single feature >30%)
- [ ] Models respond to payload size changes
- [ ] Models respond to time-of-day patterns
- [ ] Test accuracy may drop initially (this is EXPECTED and GOOD!)

## If Models STILL Produce Identical Results

If models are still 100% identical after this fix, it means:

**The labels themselves are deterministic** based on payload+time combination.

This would indicate a **DATA PROBLEM**, not a model problem. In that case:

1. The labeling logic in preprocessing script needs fundamental redesign
2. Labels may be too simplistic (e.g., all requests >1000KB → Lambda)
3. Need to add randomness or more nuanced decision boundaries
4. Consider using ACTUAL performance data instead of estimated thresholds

## Files

| File | Status | Description |
|------|--------|-------------|
| `02_ml_model_training_colab_new.ipynb` | ❌ Original | All models identical (workload_type_encoded) |
| `02_ml_model_training_colab_fixed.ipynb` | ❌ First Fix | Still identical (lambda_memory_limit_mb proxy) |
| `02_ml_model_training_colab_FINAL.ipynb` | ✅ **USE THIS** | All workload proxies removed |

## Usage

1. **Upload to Colab**:
   ```
   File: ml-notebooks/02_ml_model_training_colab_FINAL.ipynb
   ```

2. **Upload Training Data**:
   ```
   File: ml-notebooks/processed-data/ml_training_data_v6_percentile.csv
   ```

3. **Run All Cells**

4. **Verify Results** (Cell 32):
   - Check prediction agreement (should be 70-90%, NOT 100%)
   - Check within-workload variance (should vary, not 0% or 100%)
   - Check feature importance (should be balanced, no >30%)

## Success Criteria

✅ **Models are DIFFERENT**:
- RF and XGBoost disagree on 10-30% of samples
- RF and NN disagree on 15-30% of samples

✅ **Predictions VARY within workloads**:
- Same workload type shows variance (e.g., 60-80% Lambda)
- Not deterministic (not 0% or 100%)

✅ **Feature importance BALANCED**:
- No single feature >30% importance
- Top features are payload-related (payload_log, payload_category)

✅ **Models respond to changes**:
- Changing payload size affects predictions
- Changing time affects predictions
- Same workload + different payload = different prediction

## Next Steps

1. **Train with FINAL notebook**
2. **Verify models are different** (check Cell 32 output)
3. **If models still identical** → Data labeling issue, need to redesign labels
4. **If models differ** → Proceed to hyperparameter tuning for higher accuracy
5. **Document findings** in thesis

## Root Cause Timeline

1. **Day 1**: Found workload_type_encoded dominating (35.63%)
2. **Day 1**: Removed workload_type_encoded → Still 100% identical!
3. **Day 2**: DISCOVERED lambda_memory_limit_mb is perfect proxy
4. **Day 2**: Created FINAL fix removing ALL workload-derived features
5. **Day 2**: Ready for final training validation

## Lessons Learned

1. **Hidden Correlations**: Features can be proxies even if not obvious
2. **Check Feature Engineering**: Derived features can leak target information
3. **Verify Data Pipeline**: Preprocessing can create deterministic patterns
4. **100% Agreement = Red Flag**: Models should disagree if learning is real
5. **Feature Importance**: Use to detect proxy features early

---

**Status**: FINAL solution ready for validation
**File to use**: `ml-notebooks/02_ml_model_training_colab_FINAL.ipynb`
**Confidence**: High - all workload proxies removed, only pure features remain
