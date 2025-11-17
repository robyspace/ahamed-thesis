# ML Model Training Issue: All Models Produce Identical Results

## Executive Summary

**CRITICAL ISSUE FOUND**: All three ML models (Random Forest, XGBoost, Neural Network) are producing **EXACTLY the same predictions** on every single test sample, despite using:
- Different algorithms
- Different feature subsets
- Different hyperparameters

## Evidence

### 1. Identical Metrics (Cell 29)
```
Model              Test_Accuracy  Test_Precision  Test_Recall  Test_F1
Random Forest      0.7475         0.8449          0.7371       0.7874
XGBoost            0.7475         0.8449          0.7371       0.7874
Neural Network     0.7475         0.8449          0.7371       0.7874
```
**All metrics identical to 4 decimal places!**

### 2. 100% Prediction Agreement (Cell 32)
```
Total test samples: 31,805
RF == XGBoost: 31,805 (100.0%)
RF == NN:      31,805 (100.0%)
XGB == NN:     31,805 (100.0%)
```
**Every single prediction is identical across all models!**

### 3. Trivial Decision Rule (Cell 32)
```
PREDICTIONS BY WORKLOAD:
- thumbnail_processing: All models → 100% Lambda
- lightweight_api:      All models →   0% Lambda (100% ECS)
- medium_processing:    All models → 100% Lambda
- heavy_processing:     All models → 100% Lambda
```

## Root Cause

### The Models Learned a Trivial Rule

All three models learned the exact same simple decision:

```python
if workload_type == 'lightweight_api':
    return ECS (0)
else:
    return Lambda (1)
```

**ALL other features are being ignored**: payload size, time of day, memory configuration, cold start, etc.

### Why This Happened

The `workload_type_encoded` feature is **completely dominating** the decision, for the following reasons:

1. **Labeling Method Bias**: The percentile-based labeling in `01_data_preprocessing_v6_percentile_labeling.py` (lines 195-377) compares Lambda vs ECS percentiles **within each workload type**. This creates patterns that are primarily workload-dependent.

2. **Feature Importance**: Looking at the feature importance outputs:
   - **Random Forest** (Cell 21): `workload_type_encoded` has 0.1095 importance (3rd highest)
   - **XGBoost** (Cell 23): `workload_type_encoded` has **0.3563 importance** (highest by far!)
   - The next highest XGBoost feature is only 0.2489

3. **Lack of Within-Workload Variance**: The actual label distributions show variance by workload:
   - `thumbnail_processing`: 89.6% Lambda
   - `lightweight_api`: 37.3% Lambda
   - `medium_processing`: 77.4% Lambda
   - `heavy_processing`: 70.9% Lambda

   But the models just learned to predict the **majority class** for each workload, ignoring the nuances.

4. **Feature Engineering Doesn't Help**: Even though the notebook uses:
   - Different feature subsets for each model (RF: 10 features, XGB: 10 different features, NN: 15 features)
   - Advanced features like `payload_squared`, `workload_payload_interaction`, etc.

   **None of this matters** because `workload_type_encoded` dominates everything.

## Impact

- **No real ML learning**: The models are essentially a 4-way lookup table based on workload type
- **Wasted compute**: Training Random Forest, XGBoost, and Neural Network is pointless when they all learn the same trivial rule
- **Poor real-world performance**: The model won't adapt to:
  - Large vs small payloads within the same workload
  - Different times of day
  - Memory configurations
  - Cold start penalties
- **Cannot achieve 85% accuracy goal**: Current accuracy is 74.75%, and it won't improve without addressing the root cause

## Recommended Fixes

### Option 1: Remove `workload_type_encoded` from Training (Quick Fix)
Remove `workload_type_encoded` from the feature set entirely during training. This will force the models to learn from other features like:
- `payload_size_kb`
- `lambda_memory_limit_mb`
- `hour_of_day`
- Cold start status
- Feature interactions

**Pros**: Easy to implement, forces feature diversity
**Cons**: Workload type IS important for the decision, so removing it might hurt accuracy

### Option 2: Improve Label Creation (Proper Fix)
The issue is in the data preprocessing. The `create_percentile_labels()` function (lines 195-377 in `01_data_preprocessing_v6_percentile_labeling.py`) needs to create labels that depend MORE on other features.

**Current logic**:
```python
# Lines 266-272
lambda_faster = actual_lat < (compare_ecs_lat * (1 - LATENCY_MARGIN))
lambda_cheaper = actual_cost < (compare_ecs_cost * (1 - COST_MARGIN))
balanced_optimal = 1 if (lambda_faster and lambda_cheaper) else 0
```

**Problem**: The percentile comparisons (p25, p50, p75) are calculated **per workload**, making the labels heavily workload-dependent.

**Fix**: Create labels that account for cross-workload patterns:
1. Compare requests across ALL workloads, not just within workload
2. Weight other features (payload size, memory) more heavily in the decision
3. Use actual Lambda vs ECS metrics from the same request (if available) instead of percentile estimates

### Option 3: Use Stratified Sampling with Feature Constraints
During training, ensure each batch has diversity in:
- Workload types
- Payload sizes
- Memory configurations

This prevents the model from converging on workload-only patterns.

### Option 4: Add Regularization to Penalize Workload-Only Decisions
For tree-based models:
- Limit `max_depth` more aggressively
- Increase `min_samples_split` and `min_samples_leaf`
- Use `max_features` to prevent single-feature dominance

For Neural Network:
- Add stronger L1/L2 regularization
- Use dropout on the input layer to force robustness

## Verification Steps

After implementing fixes, verify that models are learning different patterns:

1. **Check Prediction Diversity**:
   ```python
   same_predictions = (y_pred_rf == y_pred_xgb).sum() / len(y_pred_rf)
   print(f"Prediction agreement: {same_predictions * 100:.1f}%")
   # Should be 70-90%, NOT 100%!
   ```

2. **Check Within-Workload Variance**:
   ```python
   for workload in workloads:
       preds = y_pred[X['workload_type'] == workload]
       print(f"{workload}: {preds.mean() * 100:.1f}% Lambda")
   # Should NOT be 0% or 100%!
   ```

3. **Check Feature Importance Diversity**:
   ```python
   # XGBoost feature importance
   # workload_type_encoded should be < 30% importance
   # Other features should contribute meaningfully
   ```

4. **Test Edge Cases**:
   ```python
   # Same workload, different payload sizes
   test_cases = [
       {'workload': 'heavy_processing', 'payload_kb': 1, 'memory_mb': 2048},
       {'workload': 'heavy_processing', 'payload_kb': 5000, 'memory_mb': 2048}
   ]
   # Predictions should be DIFFERENT!
   ```

## Next Steps

1. **Immediate**: Document this issue in the thesis
2. **Short-term**: Implement Option 1 (remove workload_type_encoded) and retrain
3. **Medium-term**: Implement Option 2 (fix labeling logic) for proper solution
4. **Long-term**: Add automated tests to detect this issue in future training runs

## Files to Modify

1. `ml-notebooks/02_ml_model_training_colab_new.ipynb` - Cell 13 (feature selection)
2. `ml-notebooks/01_data_preprocessing_v6_percentile_labeling.py` - Lines 195-377 (labeling function)
3. Create new file: `ml-notebooks/model_validation_tests.py` - Add verification tests

## References

- Notebook: `ml-notebooks/02_ml_model_training_colab_new.ipynb`
- Preprocessing: `ml-notebooks/01_data_preprocessing_v6_percentile_labeling.py`
- Evidence cells: 21, 23, 25, 26, 27, 29, 32
