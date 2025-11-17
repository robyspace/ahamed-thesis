# 🚀 Quick Start: ML Model Training on Google Colab

## One-Command Solution

```bash
cd /home/user/ahamed-thesis
bash ml-notebooks/run_preprocessing_and_package.sh
```

This script will:
1. ✅ Check your collected data files
2. ✅ Run data preprocessing with cost calculation
3. ✅ Create training-ready CSV file
4. ✅ Package everything for Google Colab upload
5. ✅ Generate step-by-step upload instructions

---

## Manual Steps (if needed)

### Step 1: Preprocess Data

```bash
python3 ml-notebooks/01_data_preprocessing.py
```

**Output:**
- `ml-notebooks/processed-data/ml_training_data.csv` (upload this to Colab)
- `ml-notebooks/processed-data/full_processed_data.csv`
- `ml-notebooks/processed-data/preprocessing_summary.json`

### Step 2: Upload to Google Colab

1. Go to https://colab.research.google.com/
2. Upload `ml-notebooks/02_ml_model_training_colab.ipynb`
3. When prompted, upload `ml-training_data.csv`
4. Click "Runtime" → "Run all"

### Step 3: Download Trained Models

After training completes (~15-30 min):
- `best_model_xgboost.pkl`
- `scaler.pkl`
- `feature_columns.json`
- `model_metadata.json`

---

## 📊 Your Data Summary

**Total Collected: 212,033 requests**

| Platform | Lightweight | Thumbnail | Medium | Heavy | Total |
|----------|------------|-----------|--------|-------|-------|
| Lambda | 46,820 | 36,177 | 18,830 | 3,968 | 105,795 |
| ECS | 48,573 | 36,276 | 17,648 | 3,741 | 106,238 |

This is **4.2x more** than your 50K target! 🎉

---

## 🎯 Expected Results

### Model Performance Targets

| Model | Expected Accuracy |
|-------|------------------|
| Random Forest | 75-80% |
| **XGBoost** | **85%+** ✅ |
| Neural Network | 80-85% |

### Training Time

- **Google Colab (Free):** 15-30 minutes
- **Google Colab Pro:** 10-15 minutes

---

## 💡 What the Models Learn

The ML models predict the optimal platform (Lambda vs ECS) based on:

**Input Features (at request time):**
1. Workload type (lightweight/thumbnail/medium/heavy)
2. Payload size (KB)
3. Hour of day
4. Day of week
5. Weekend flag

**Output:**
- `1` = Lambda is optimal (faster + cheaper)
- `0` = ECS is optimal (faster + cheaper)

**Ground Truth:** Based on actual latency + cost from your collected data

---

## 📁 File Structure After Preprocessing

```
ahamed-thesis/
├── data-output/                    # Your collected metrics (212K requests)
│   ├── lambda_lightweight_api_2025-11-16.jsonl
│   ├── lambda_thumbnail_processing_2025-11-16.jsonl
│   ├── lambda_medium_processing_2025-11-16.jsonl
│   ├── lambda_heavy_processing_2025-11-16.jsonl
│   ├── ecs_lightweight_api_2025-11-16.jsonl
│   └── ... (similar for Nov 17)
│
├── ml-notebooks/
│   ├── 01_data_preprocessing.py         # Preprocessing script
│   ├── 02_ml_model_training_colab.ipynb # Colab notebook
│   ├── README.md                         # Detailed guide
│   ├── QUICK_START.md                    # This file
│   ├── run_preprocessing_and_package.sh  # Automation script
│   │
│   ├── processed-data/                   # Generated after preprocessing
│   │   ├── ml_training_data.csv          # 📤 Upload this to Colab
│   │   ├── full_processed_data.csv
│   │   └── preprocessing_summary.json
│   │
│   └── colab-upload/                     # Ready-to-upload package
│       ├── ml_training_data.csv
│       ├── 02_ml_model_training_colab.ipynb
│       ├── README.md
│       └── UPLOAD_INSTRUCTIONS.txt
│
└── ml-models/                           # Create this for trained models
    └── trained/                         # Save downloaded models here
        ├── best_model_xgboost.pkl
        ├── scaler.pkl
        ├── feature_columns.json
        └── model_metadata.json
```

---

## ⚠️ Troubleshooting

### "FileNotFoundError: ml_training_data.csv"
**Solution:** Run preprocessing first:
```bash
python3 ml-notebooks/01_data_preprocessing.py
```

### "Memory Error in Colab"
**Solutions:**
- Use Google Colab Pro (more RAM)
- Reduce data size by editing DATE_FILTER in preprocessing script
- Use batch processing

### "Low Model Accuracy (<70%)"
**Possible Issues:**
1. Imbalanced labels → Check `preprocessing_summary.json`
2. Insufficient data for some workloads → Collect more
3. Poor feature selection → Add more features in preprocessing

**Solutions:**
- Adjust `latency_tolerance` and `cost_tolerance` in preprocessing
- Use SMOTE for class balancing
- Add more domain-specific features

### "Cannot upload file to Colab"
**Solutions:**
- Check file size (Colab limit: 100 MB for uploads)
- Use Google Drive mount instead:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  df = pd.read_csv('/content/drive/MyDrive/ml_training_data.csv')
  ```

---

## 🎓 Learning Resources

### Understanding the Models

**Random Forest:**
- Ensemble of decision trees
- Votes on best platform
- Interpretable (can see feature importance)

**XGBoost:**
- Gradient boosting (trees learn from previous mistakes)
- Usually highest accuracy
- Industry standard for tabular data

**Neural Network:**
- Deep learning approach
- Can learn complex patterns
- Requires more data and tuning

### Key Metrics

**Accuracy:** % of correct predictions
- `Accuracy = (TP + TN) / (TP + TN + FP + FN)`

**Precision:** Of predicted Lambda, how many are actually optimal?
- `Precision = TP / (TP + FP)`

**Recall:** Of actual optimal Lambda, how many did we predict?
- `Recall = TP / (TP + FN)`

**F1-Score:** Balance between precision and recall
- `F1 = 2 × (Precision × Recall) / (Precision + Recall)`

---

## ✅ Checklist

Before starting ML training:

- [ ] Collected 212,033 requests (✅ Done!)
- [ ] All 4 workload types have data for Lambda and ECS
- [ ] Preprocessing script runs successfully
- [ ] `ml_training_data.csv` generated
- [ ] File size is reasonable (<100 MB for Colab upload)
- [ ] Google Colab account ready

After ML training:

- [ ] Test accuracy >= 85%
- [ ] Model files downloaded
- [ ] Saved to `ml-models/trained/` directory
- [ ] Ready for Phase 4: Intelligent Router deployment

---

## 🚀 Next Phases

**Phase 4: Intelligent Router (After ML Training)**
- Create AWS Lambda function to host the model
- Implement routing API
- Load and serve predictions

**Phase 5: Evaluation**
- Compare ML-hybrid vs Lambda-only vs ECS-only
- Measure cost savings and latency improvements
- Validate accuracy in production

**Phase 6: Documentation**
- Thesis write-up
- Visualizations and charts
- Final presentation

---

## 📞 Need Help?

1. **Detailed Guide:** Read `ml-notebooks/README.md`
2. **Colab Notebook:** Check cell outputs for error messages
3. **Preprocessing:** Review `preprocessing_summary.json` for data quality
4. **Model Performance:** Check confusion matrices and classification reports

---

**Ready to train your models? Let's go! 🎉**

```bash
bash ml-notebooks/run_preprocessing_and_package.sh
```
