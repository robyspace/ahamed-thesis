# 🤖 ML Model Development Guide - Google Colab

## Phase 3: ML Model Training for Intelligent Router

This directory contains all the necessary scripts and notebooks for training machine learning models to predict the optimal platform (AWS Lambda vs ECS Fargate) for different workload types.

---

## 📁 Files Overview

```
ml-notebooks/
├── 01_data_preprocessing.py          # Data preprocessing with cost calculation
├── 02_ml_model_training_colab.ipynb  # Complete ML training notebook for Colab
├── README.md                         # This file
└── processed-data/                   # Output directory (generated after running preprocessing)
    ├── full_processed_data.csv
    ├── ml_training_data.csv
    └── preprocessing_summary.json
```

---

## 🚀 Quick Start Guide

### Step 1: Run Data Preprocessing Locally

Before uploading to Google Colab, you need to preprocess your collected metrics data locally.

```bash
# Navigate to your thesis directory
cd /path/to/ahamed-thesis

# Install required packages (if not already installed)
pip install pandas numpy

# Run preprocessing script
python ml-notebooks/01_data_preprocessing.py
```

**Expected Output:**
```
================================================================================
HYBRID SERVERLESS-CONTAINER THESIS
Data Preprocessing Pipeline with Cost Calculation
================================================================================

STEP 1: LOADING DATA
================================================================================
📁 Found XX JSONL files
...

✅ Loaded 212,033 total requests

STEP 2: NORMALIZING METRICS
...

STEP 3: FEATURE ENGINEERING
...

STEP 4: COST CALCULATION
💰 Calculating costs...
   💵 Calculated Lambda costs: 105,795 requests
      Average: $0.0000003xxx per request
      Total: $0.0xxx
   💵 Calculated ECS costs: 106,238 requests
      Average: $0.0000001xxx per request
      Total: $0.0xxx

STEP 5: GROUND TRUTH LABELING
🏷️  Creating ground truth labels...
...

STEP 6: SAVING PROCESSED DATA
💾 Saved full data: ml-notebooks/processed-data/full_processed_data.csv
💾 Saved ML training data: ml-notebooks/processed-data/ml_training_data.csv
💾 Saved summary: ml-notebooks/processed-data/preprocessing_summary.json

✅ PREPROCESSING COMPLETE!
```

**What this script does:**
- ✅ Loads all JSONL files from `data-output/` directory
- ✅ Filters data from Nov 16-17 (your 6-hour collection period)
- ✅ Calculates per-request costs using AWS pricing formulas
- ✅ Creates feature engineering (temporal, categorical)
- ✅ Generates ground truth labels (optimal platform based on cost + latency)
- ✅ Exports clean CSV files ready for ML training

---

### Step 2: Upload to Google Colab

1. **Open Google Colab:** https://colab.research.google.com/
2. **Upload the notebook:**
   - Click "File" → "Upload notebook"
   - Upload `ml-notebooks/02_ml_model_training_colab.ipynb`
3. **Upload the preprocessed data:**
   - When prompted in the notebook, upload `ml-notebooks/processed-data/ml_training_data.csv`

---

### Step 3: Run ML Training in Colab

The Colab notebook will guide you through:

1. **📦 Install packages** (automatically handled)
2. **📂 Load preprocessed data**
3. **📊 Exploratory Data Analysis (EDA)**
   - Visualize workload distributions
   - Cost comparisons
   - Latency patterns
4. **🎯 Feature engineering and train-test split**
5. **Train 3 ML models:**
   - 🌲 **Random Forest** (baseline, interpretable)
   - 🚀 **XGBoost** (target: 85%+ accuracy)
   - 🧠 **Neural Network** (deep learning comparison)
6. **📊 Model evaluation and comparison**
7. **💾 Export the best model** for AWS deployment

---

## 📊 Expected Results

Based on your **212,033 requests** collected:

### Data Distribution

| Platform | Workload | Count | % |
|----------|----------|-------|---|
| Lambda | Lightweight | 46,820 | 22.1% |
| Lambda | Thumbnail | 36,177 | 17.1% |
| Lambda | Medium | 18,830 | 8.9% |
| Lambda | Heavy | 3,968 | 1.9% |
| ECS | Lightweight | 48,573 | 22.9% |
| ECS | Thumbnail | 36,276 | 17.1% |
| ECS | Medium | 17,648 | 8.3% |
| ECS | Heavy | 3,741 | 1.8% |

### Model Performance Targets

| Model | Target Accuracy | Expected Use Case |
|-------|----------------|-------------------|
| Random Forest | 75-80% | Baseline, interpretable |
| **XGBoost** | **85%+** | **Production deployment** |
| Neural Network | 80-85% | Deep learning comparison |

---

## 💰 Cost Calculation Details

### Lambda Cost Formula

```python
# AWS Lambda Pricing (eu-west-1)
REQUEST_COST = $0.20 / 1,000,000 = $0.0000002 per request
DURATION_COST = $0.0000166667 per GB-second

# Per-request calculation:
lambda_cost = (memory_mb / 1024) * (execution_time_ms / 1000) * 0.0000166667 + 0.0000002
```

**Example:**
- Memory: 128 MB
- Execution: 116.52 ms
- Cost: `(128/1024) * (116.52/1000) * 0.0000166667 + 0.0000002 = $0.000000443`

### ECS Fargate Cost Formula

```python
# AWS ECS Fargate Pricing (eu-west-1)
VCPU_COST = $0.04656 per vCPU-hour
MEMORY_COST = $0.00511 per GB-hour

# Task configurations (from task definitions):
lightweight_api: 0.25 vCPU, 0.5 GB
thumbnail: 0.5 vCPU, 1.0 GB
medium: 1.0 vCPU, 2.0 GB
heavy: 1.0 vCPU, 2.0 GB

# Per-request calculation (amortized):
ecs_cost = (vcpu * 0.04656 + memory_gb * 0.00511) / 3600 * (execution_time_ms / 1000)
```

**Example (Lightweight):**
- Task: 0.25 vCPU, 0.5 GB
- Execution: 52.38 ms
- Cost: `(0.25 * 0.04656 + 0.5 * 0.00511) / 3600 * (52.38/1000) = $0.000000206`

---

## 🏷️ Ground Truth Labeling Logic

The optimal platform is determined using **both** cost and latency:

```python
def label_optimal_platform(lambda_data, ecs_data):
    """
    Returns 1 for Lambda, 0 for ECS

    Lambda is optimal if BOTH:
    1. Latency: lambda_latency <= ecs_latency × 1.1 (10% tolerance)
    2. Cost: lambda_cost <= ecs_cost × 1.1 (10% tolerance)
    """
    latency_ok = lambda_latency <= ecs_latency * 1.1
    cost_ok = lambda_cost <= ecs_cost * 1.1

    return 1 if (latency_ok and cost_ok) else 0
```

**Rationale:**
- If Lambda is faster AND cheaper (or within 10% tolerance) → Lambda wins
- Otherwise → ECS wins
- This ensures we don't sacrifice too much on one metric for gains in the other

---

## 🎯 Feature Selection

The ML models will use these features (available at runtime for prediction):

| Feature | Type | Description |
|---------|------|-------------|
| `workload_type_encoded` | Categorical (0-3) | 0=lightweight, 1=thumbnail, 2=medium, 3=heavy |
| `payload_size_kb` | Continuous | Size of request payload in KB |
| `hour_of_day` | Categorical (0-23) | Hour when request was made |
| `day_of_week` | Categorical (0-6) | Day of week (0=Monday, 6=Sunday) |
| `is_weekend` | Binary (0/1) | Weekend flag |

**Why these features?**
- ✅ Available at request time (no need for execution first)
- ✅ Strong correlation with performance patterns
- ✅ Captures workload type, size, and temporal patterns
- ✅ Minimal overhead for prediction

---

## 📦 Model Export and Deployment

After training, the notebook will export:

1. **`best_model_xgboost.pkl`** (or `random_forest.pkl`, `neural_network.h5`)
   - Trained model ready for inference

2. **`scaler.pkl`**
   - Feature scaler (needed for Neural Network)

3. **`feature_columns.json`**
   - List of feature names in correct order

4. **`model_metadata.json`**
   - Model performance metrics
   - Training configuration
   - Feature importance

**Deployment workflow:**
```
Google Colab Training
    ↓
Download model files
    ↓
Upload to AWS S3
    ↓
Create Lambda function (Intelligent Router)
    ↓
Load model and serve predictions
    ↓
Route requests to optimal platform
```

---

## 🔍 Troubleshooting

### Issue: "No such file or directory: ml_training_data.csv"

**Solution:** Make sure you've run the preprocessing script first:
```bash
python ml-notebooks/01_data_preprocessing.py
```

### Issue: "Memory error when loading data"

**Solution:**
- Colab has 12-25 GB RAM depending on runtime
- If needed, reduce data size in preprocessing by adjusting `DATE_FILTER`
- Use batch processing for very large datasets

### Issue: "Low model accuracy (<70%)"

**Possible causes:**
1. **Imbalanced labels:** Check label distribution
2. **Insufficient data:** Collect more requests for underrepresented workloads
3. **Feature engineering:** Add more relevant features
4. **Hyperparameter tuning:** Adjust model parameters

**Solutions:**
- Use SMOTE for class balancing
- Adjust `latency_tolerance` and `cost_tolerance` in labeling
- Add features like: concurrent requests, system load, request patterns

---

## 📊 Evaluation Metrics

### Primary Metric: **Accuracy**
- Target: **85%+** overall accuracy
- Measures: % of correct platform predictions

### Secondary Metrics:

1. **Precision** (How many predicted Lambda/ECS are actually optimal?)
   - `Precision = TP / (TP + FP)`
   - Important for minimizing wrong routing

2. **Recall** (How many actual optimal Lambda/ECS were correctly predicted?)
   - `Recall = TP / (TP + FN)`
   - Important for maximizing correct routing

3. **F1-Score** (Harmonic mean of precision and recall)
   - `F1 = 2 * (Precision * Recall) / (Precision + Recall)`
   - Balanced metric

4. **Confusion Matrix**
   - Shows: True Positives, False Positives, True Negatives, False Negatives
   - Helps identify where model makes mistakes

---

## 🚀 Next Steps After Training

1. **✅ Download trained model files** from Colab
2. **📤 Upload to your repository:**
   ```bash
   mkdir -p ml-models/trained
   cp best_model_*.pkl ml-models/trained/
   cp scaler.pkl ml-models/trained/
   cp *.json ml-models/trained/
   ```

3. **🔧 Create Intelligent Router API** (Phase 4)
   - Lambda function to host the model
   - Accepts: workload type, payload size, timestamp
   - Returns: optimal platform (Lambda/ECS)

4. **📊 Evaluate Hybrid System** (Phase 5)
   - Compare against Lambda-only baseline
   - Compare against ECS-only baseline
   - Measure: cost savings, latency improvements, accuracy

5. **📝 Document Results** (Phase 6)
   - Thesis documentation
   - Performance visualizations
   - Cost-benefit analysis

---

## 📚 Additional Resources

### AWS Pricing References
- [Lambda Pricing](https://aws.amazon.com/lambda/pricing/)
- [ECS Fargate Pricing](https://aws.amazon.com/fargate/pricing/)

### ML Model Documentation
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Keras/TensorFlow](https://www.tensorflow.org/guide/keras)

### Google Colab Tips
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [GPU/TPU acceleration](https://colab.research.google.com/notebooks/gpu.ipynb)

---

## 💡 Tips for Best Results

1. **Data Quality:**
   - Ensure all workload types have sufficient samples (1000+ each)
   - Remove outliers if needed
   - Check for data collection errors

2. **Feature Engineering:**
   - Add domain-specific features if available
   - Consider interaction features (e.g., workload_type × payload_size)
   - Normalize/scale features appropriately

3. **Model Tuning:**
   - Use GridSearchCV for hyperparameter optimization
   - Monitor for overfitting (train vs. validation accuracy)
   - Try ensemble methods if single models underperform

4. **Evaluation:**
   - Use stratified k-fold cross-validation
   - Test on truly unseen data
   - Consider per-workload accuracy (not just overall)

---

## ✅ Checklist

Before proceeding to deployment, ensure:

- [ ] Preprocessing script runs successfully
- [ ] All 4 workload types have data for both platforms
- [ ] ML training notebook completes without errors
- [ ] Test accuracy >= 85% (target)
- [ ] Model files exported successfully
- [ ] Feature names and metadata saved
- [ ] Performance metrics documented

---

## 🤝 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review error messages in Colab output
3. Verify data quality and preprocessing steps
4. Adjust model hyperparameters if needed

---

**Good luck with your ML model training! 🚀**
