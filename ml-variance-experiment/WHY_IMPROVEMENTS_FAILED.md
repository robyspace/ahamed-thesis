# Why ML Improvements Failed - Root Cause Analysis
**Date:** 2025-11-24
**Critical Finding:** Standard ML improvement techniques decreased accuracy by 3.53%

---

## 📊 Performance Comparison

| Model | Previous | Improved | Change |
|-------|----------|----------|--------|
| **Neural Network** | **69.59%** | **66.06%** | **-3.53%** ❌ |
| Random Forest | 69.48% | 65.15% | -4.33% ❌ |
| XGBoost | 69.48% | 60.30% | -9.18% ❌ |
| LightGBM | 69.48% | 65.16% | -4.32% ❌ |
| Ensemble | N/A | 62.85% | New (poor) ❌ |

**All models performed worse after "improvements"**

---

## 🔍 Root Cause Analysis

### **1. SMOTE Balancing Introduced Noise** ❌

**What we did:**
- Applied SMOTE to balance classes from 65-35 to 50-50
- Generated synthetic minority samples

**Why it failed:**
```
Real-world distribution: 65% Lambda, 35% ECS
SMOTE distribution:      50% Lambda, 50% ECS

Problem: Synthetic samples DON'T reflect real patterns!
```

**Explanation:**
- **SMOTE creates synthetic samples** by interpolating between minority class neighbors
- These synthetic samples may not represent real workload behaviors
- The model learned patterns from **artificial data** that don't exist in reality
- When tested on real data, the model performs worse

**Key Insight:**
The 65-35 imbalance is **NOT a bug** - it reflects reality! Lambda IS optimal more often in practice. Balancing to 50-50 made the model learn an incorrect prior distribution.

**XGBoost particularly affected:**
- Previous: 69.48%
- Improved: 60.30% (-9.18% - worst decline)
- XGBoost is sensitive to sample weights and synthetic data
- Struggled with artificially balanced distribution

---

### **2. Too Many Features Diluted Signal** ❌

**What we did:**
- Increased features from 15 to 25 (+10 new features)
- Added polynomial, interaction, binned features

**Why it failed:**
```
Original: 15 carefully selected features
Improved: 25 features (many redundant/noisy)

Problem: Curse of dimensionality + signal dilution
```

**Feature Redundancy:**
```python
# We added:
payload_squared     # Redundant with payload_size_kb
payload_cubed       # Even more redundant
payload_sqrt        # Yet another payload transform
payload_inv         # Another payload transform

# These are all highly correlated!
Correlation matrix shows:
- payload_size_kb ↔ payload_squared: 0.95+
- payload_size_kb ↔ payload_log: 0.90+
- payload_size_kb ↔ payload_cubed: 0.92+
```

**Impact:**
- Model had to learn from 25 features instead of 15
- Many features highly correlated (multicollinearity)
- Signal-to-noise ratio decreased
- Overfitting to training data increased

**Evidence:**
- Train-test gaps remained small (good generalization)
- But overall accuracy dropped (worse decision boundaries)
- Models couldn't identify which features were truly predictive

---

### **3. Hyperparameter Tuning Overfit to CV** ❌

**What we did:**
- GridSearchCV with 108 parameter combinations
- 5-fold cross-validation
- Selected "best" parameters based on CV score

**Why it failed:**
```
Cross-validation score: Optimized on balanced synthetic data
Test performance: Evaluated on real imbalanced data

Problem: Optimization target mismatched reality
```

**Explanation:**
- Hyperparameters were tuned on **SMOTE-balanced data** (50-50)
- Test set has **real distribution** (65-35)
- "Best" parameters for synthetic data ≠ best for real data
- Model optimized for wrong distribution

**Neural Network example:**
- GridSearch selected: hidden_layers=(128,64,32), lr=0.01, alpha=0.001
- These parameters work well for balanced data
- But perform worse on imbalanced real data
- Previous default parameters were actually better!

---

### **4. Ensemble Failed Due to Poor Components** ❌

**What we did:**
- Created VotingClassifier with all 4 models
- Soft voting (probability averaging)

**Why it failed:**
```
Ensemble accuracy: 62.85%
Individual best: 66.06%

Problem: Ensemble WORSE than best individual!
```

**Explanation:**
- Ensemble only works if individual models are diverse AND good
- All tree models perform poorly (60-65%)
- Averaging poor predictions doesn't help
- Neural Network (66%) was dragged down by poor tree models

**Classic ensemble failure mode:**
- Garbage in → Garbage out
- Can't rescue bad models by combining them

---

## 💡 Key Lessons Learned

### **1. Real-World Imbalance is NOT a Problem to Fix**

The 65-35 class distribution reflects reality:
- Lambda IS optimal more often in real workloads
- Balancing to 50-50 is artificial and harmful
- Model should learn the true prior distribution

**Correct Approach:**
- Keep natural 65-35 distribution
- Use class weights if needed (not SMOTE)
- Let model learn real-world patterns

---

### **2. More Features ≠ Better Performance**

Feature engineering requires:
- **Domain knowledge** over blind additions
- **Removing redundancy** not adding more
- **Signal enhancement** not noise introduction

**Our mistake:**
- Added 10 features blindly
- Many highly correlated
- Diluted existing signal

**Correct Approach:**
- Use fewer, more meaningful features
- Remove redundant features
- Focus on features that capture real decision factors

---

### **3. Standard ML Practices Don't Always Apply**

Common ML best practices that **FAILED** here:
- ❌ Balance classes with SMOTE
- ❌ Add more features for more signal
- ❌ Tune hyperparameters extensively
- ❌ Create ensembles for better performance

**Why they failed:**
- Our problem has unique characteristics:
  - Limited data variance (by design)
  - Natural class imbalance that's meaningful
  - Strong dominant features (workload_type)
  - Small dataset (88K samples)

**Lesson:** Context matters more than cookbook approaches

---

### **4. The Original Model Was Already Good**

**Previous model (69.59%):**
- ✅ Used natural class distribution
- ✅ Had focused, meaningful features
- ✅ Simple hyperparameters that worked
- ✅ No artificial data

**"Improved" model (66.06%):**
- ❌ Artificial class balance
- ❌ Feature bloat
- ❌ Over-tuned parameters
- ❌ Synthetic data contamination

**Verdict:** Sometimes simpler is better!

---

## 🎯 What Actually Limits Accuracy?

After this experiment, we know the true limitations:

### **Not Fixable with ML Techniques:**

1. **Data Determinism**
   - medium_processing: 96.1% Lambda optimal
   - Can't learn variance where it doesn't exist
   - Would need NEW data collection

2. **Feature Limitations**
   - Missing: cold start probability, network latency, memory pressure
   - Can't predict what we can't measure
   - Would need instrumentation changes

3. **Label Noise**
   - lightweight_api: 47.2% Lambda (near random)
   - Borderline cases are inherently ambiguous
   - Perfect accuracy impossible

4. **Fundamental Complexity**
   - Real-world platform selection is complex
   - 69.6% may be near the theoretical maximum for this feature set
   - More sophisticated features needed, not more ML tricks

---

## 📋 Recommendations

### **1. Keep the Original Model** ✅

**Rationale:**
- 69.59% is better than 66.06%
- Simpler and more interpretable
- Uses natural data distribution
- Already validated and ready

**Action:**
- Use the original Neural Network (69.59%)
- Deploy this model to AWS Lambda
- Don't use the "improved" version

---

### **2. If You Must Improve (75%+ goal):**

#### **Option A: Selective Feature Engineering** 🎯

**Add ONLY these specific features:**
```python
# Cold start indicators
'is_cold_start_likely' = f(time_since_last_request, time_window)

# Memory pressure
'memory_utilization_ratio' = payload_size / lambda_memory_limit

# Network indicators
'is_vpc_attached' = boolean
'expected_network_latency' = f(region, vpc)

# Concurrent load
'concurrent_request_estimate' = f(load_pattern, time_window)
```

**DON'T add:**
- Redundant payload transforms
- Unnecessary polynomial features
- Random interaction terms

**Expected gain:** +2-3% → 71-73%

---

#### **Option B: Collect More Data** 📊

**Focus on:**
- medium_processing with more payload variance
- More time windows (expand from 5 to 10)
- More load patterns (expand from 4 to 8)
- Edge cases where model is uncertain

**Target:** 200K samples (up from 88K)
**Expected gain:** +3-5% → 73-75%
**Time required:** 2-3 days

---

#### **Option C: Hybrid Approach** 🔄

**Combine:**
1. Keep original 15 features
2. Add 3-5 carefully selected new features (Option A)
3. Collect 50K more samples for problem workloads
4. NO SMOTE, NO hyperparameter tuning, NO ensemble

**Expected gain:** +4-6% → 73-75%
**Time required:** 3-4 days

---

### **3. For Thesis: Be Honest About Results** 📝

**What to report:**
```markdown
### Improvement Attempts

We attempted to improve the 69.59% baseline through standard ML
techniques including SMOTE class balancing, feature engineering,
hyperparameter tuning, and ensemble methods.

**Results:** These techniques decreased accuracy to 66.06% (-3.53%).

**Analysis:** The performance degradation revealed important insights:

1. **Class imbalance reflects reality:** The 65-35 Lambda-ECS distribution
   is not a data quality issue but reflects real-world optimal patterns.
   Artificial balancing introduced harmful bias.

2. **Feature quality over quantity:** Adding 10 additional features diluted
   the signal rather than enhancing it, demonstrating the importance of
   domain-informed feature selection.

3. **Simplicity is valuable:** The original model's simpler architecture
   and natural data distribution proved more effective than complex
   improvements, challenging the "more is better" assumption.

**Conclusion:** The 69.59% baseline represents a reasonable ceiling given
current feature constraints and data characteristics. Further improvement
requires expanded data collection or instrumentation for additional features,
not more sophisticated ML techniques.
```

**This is STRONG thesis content:**
- Shows critical thinking
- Demonstrates scientific rigor
- Validates that you tried to improve
- Explains why improvements failed
- Provides valuable lessons

---

## ✅ Final Recommendation

### **Deploy the Original Model (69.59%)**

**Reasoning:**
1. Better performance than "improved" version
2. Simpler and more maintainable
3. Uses real data distribution
4. Already validated
5. Ready for deployment

**Path Forward:**
1. ✅ Use ML_Variance_Training_Complete.ipynb results (69.59%)
2. ✅ Deploy variance_model_best.pkl to AWS Lambda
3. ✅ Run comparative evaluation
4. ✅ Document honestly in thesis (including failed improvements)
5. ✅ Suggest future work (more data, better features)

**Do NOT:**
- ❌ Use the "improved" 66.06% model
- ❌ Try more ML tricks without new data
- ❌ Hide the failed improvement attempt

---

## 🎓 Thesis Contribution

This "failure" is actually **valuable research**:

**Novel Contribution:**
"We demonstrate that standard ML improvement techniques (SMOTE, feature expansion, hyperparameter tuning) can degrade performance when applied to problems with:
- Meaningful class imbalance
- Limited but intentional data variance
- Strong categorical features
- Domain-specific constraints

Our results emphasize the importance of problem context over standard ML recipes."

**This makes your thesis STRONGER, not weaker!**

---

**Last Updated:** 2025-11-24 09:15:00
**Status:** Analysis complete, recommendation clear
**Action:** Deploy original 69.59% model
