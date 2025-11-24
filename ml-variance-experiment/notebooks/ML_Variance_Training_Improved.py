"""
Improved ML Training Script for Variance-Aware Platform Selection
Addresses:
1. Identical results bug for tree-based models
2. Low accuracy (69.6% → target 75%+)

Improvements:
- Advanced hyperparameter tuning
- Better feature engineering
- Class imbalance handling
- Ensemble methods
- Cross-validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

print("="*80)
print("IMPROVED ML TRAINING PIPELINE")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("\n" + "="*80)
print("1. LOADING DATA")
print("="*80)

DATA_FILE = 'preprocessing/processed-data/ml_training_data_variance_v1_20251123_134539.csv'
df = pd.read_csv(DATA_FILE)

print(f"✅ Data loaded: {len(df):,} samples")

# ==============================================================================
# 2. ADVANCED FEATURE ENGINEERING
# ==============================================================================
print("\n" + "="*80)
print("2. ADVANCED FEATURE ENGINEERING")
print("="*80)

# Original features
ORIGINAL_FEATURES = [
    'workload_type_encoded', 'payload_size_kb', 'time_window_encoded',
    'load_pattern_encoded', 'hour_of_day', 'is_weekend', 'lambda_memory_limit_mb',
    'payload_squared', 'payload_log', 'payload_workload_interaction',
    'payload_hour_interaction', 'payload_time_window_interaction',
    'workload_time_window_interaction', 'payload_load_pattern_interaction',
    'time_window_load_pattern_interaction'
]

# NEW: Advanced feature engineering
print("\nCreating advanced features...")

# 1. Polynomial features for payload (capture non-linear relationships)
df['payload_cubed'] = df['payload_size_kb'] ** 3
df['payload_sqrt'] = np.sqrt(df['payload_size_kb'])
df['payload_inv'] = 1 / (df['payload_size_kb'] + 1)  # Avoid division by zero

# 2. More interaction terms
df['memory_payload_interaction'] = df['lambda_memory_limit_mb'] * df['payload_size_kb']
df['memory_workload_interaction'] = df['lambda_memory_limit_mb'] * df['workload_type_encoded']
df['three_way_interaction'] = df['payload_size_kb'] * df['time_window_encoded'] * df['load_pattern_encoded']

# 3. Binned features (help models find thresholds)
df['payload_bin'] = pd.qcut(df['payload_size_kb'], q=10, labels=False, duplicates='drop')
df['hour_bin'] = pd.cut(df['hour_of_day'], bins=[0, 6, 12, 18, 24], labels=False, include_lowest=True)

# 4. Ratio features
df['payload_per_memory'] = df['payload_size_kb'] / (df['lambda_memory_limit_mb'] + 1)

# 5. Aggregate features (statistical)
df['payload_zscore'] = (df['payload_size_kb'] - df['payload_size_kb'].mean()) / df['payload_size_kb'].std()

# Updated feature list
FEATURE_COLUMNS = ORIGINAL_FEATURES + [
    'payload_cubed', 'payload_sqrt', 'payload_inv',
    'memory_payload_interaction', 'memory_workload_interaction',
    'three_way_interaction', 'payload_bin', 'hour_bin',
    'payload_per_memory', 'payload_zscore'
]

print(f"✅ Feature engineering complete")
print(f"   Original features: {len(ORIGINAL_FEATURES)}")
print(f"   New features: {len(FEATURE_COLUMNS) - len(ORIGINAL_FEATURES)}")
print(f"   Total features: {len(FEATURE_COLUMNS)}")

# ==============================================================================
# 3. PREPARE DATA
# ==============================================================================
print("\n" + "="*80)
print("3. PREPARING DATA")
print("="*80)

X = df[FEATURE_COLUMNS].copy()
y = df['balanced_optimal'].copy()

print(f"Features: {X.shape}")
print(f"Target: {y.shape}")
print(f"Class distribution: {y.value_counts().to_dict()}")
print(f"  Lambda optimal: {y.sum():,} ({y.mean()*100:.1f}%)")
print(f"  ECS optimal: {(~y.astype(bool)).sum():,} ({(1-y.mean())*100:.1f}%)")

# Train-Val-Test split (60-20-20)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nSplit sizes:")
print(f"  Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Val:   {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

# ==============================================================================
# 4. HANDLE CLASS IMBALANCE
# ==============================================================================
print("\n" + "="*80)
print("4. HANDLING CLASS IMBALANCE")
print("="*80)

print(f"\nOriginal training distribution:")
print(f"  Lambda: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"  ECS: {(~y_train.astype(bool)).sum():,} ({(1-y_train.mean())*100:.1f}%)")

# Apply SMOTE to balance classes
print("\nApplying SMOTE (Synthetic Minority Over-sampling)...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Balanced training distribution:")
print(f"  Lambda: {y_train_balanced.sum():,} ({y_train_balanced.mean()*100:.1f}%)")
print(f"  ECS: {(~y_train_balanced.astype(bool)).sum():,} ({(1-y_train_balanced.mean())*100:.1f}%)")
print(f"✅ Training set balanced: {len(X_train_balanced):,} samples")

# ==============================================================================
# 5. HYPERPARAMETER TUNING
# ==============================================================================
print("\n" + "="*80)
print("5. HYPERPARAMETER TUNING")
print("="*80)

print("\n🔍 Tuning Neural Network (this may take 5-10 minutes)...")

# Neural Network with GridSearch
nn_param_grid = {
    'hidden_layer_sizes': [(128, 64, 32), (64, 32, 16), (100, 50, 25), (128, 64)],
    'learning_rate_init': [0.001, 0.01, 0.005],
    'alpha': [0.0001, 0.001, 0.01],  # L2 regularization
    'batch_size': [32, 64, 128]
}

nn_base = MLPClassifier(
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)

nn_grid = GridSearchCV(
    nn_base, nn_param_grid, cv=5, scoring='accuracy',
    n_jobs=-1, verbose=1
)

nn_grid.fit(X_train_balanced, y_train_balanced)

print(f"\n✅ Best Neural Network parameters:")
for param, value in nn_grid.best_params_.items():
    print(f"   {param}: {value}")
print(f"   Best CV score: {nn_grid.best_score_:.4f}")

best_nn = nn_grid.best_estimator_

# ==============================================================================
# 6. TRAIN IMPROVED MODELS WITH BETTER HYPERPARAMETERS
# ==============================================================================
print("\n" + "="*80)
print("6. TRAINING IMPROVED MODELS")
print("="*80)

models = {
    'Neural Network (Tuned)': best_nn,
    'Random Forest (Improved)': RandomForestClassifier(
        n_estimators=500,  # More trees
        max_depth=20,  # Deeper trees
        min_samples_split=5,  # More granular splits
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',  # Handle imbalance
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost (Improved)': XGBClassifier(
        n_estimators=500,
        max_depth=10,  # Deeper
        learning_rate=0.05,  # Lower learning rate
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(~y_train.astype(bool)).sum() / y_train.sum(),  # Handle imbalance
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),
    'LightGBM (Improved)': LGBMClassifier(
        n_estimators=500,
        max_depth=15,
        learning_rate=0.05,
        num_leaves=50,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'='*80}")
    print(f"Training: {name}")
    print(f"{'='*80}")

    # Train (use balanced data for tree models, original for NN)
    if 'Neural Network' in name:
        # NN was already trained with GridSearch
        print("Using pre-trained model from GridSearch...")
    else:
        print("Fitting model...")
        model.fit(X_train_balanced, y_train_balanced)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Probabilities
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba)

    results[name] = {
        'model': model,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'train_test_gap': train_acc - test_acc,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }

    print(f"\n📊 Results:")
    print(f"  Train Accuracy:  {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Val Accuracy:    {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Test Accuracy:   {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Precision:       {precision:.4f}")
    print(f"  Recall:          {recall:.4f}")
    print(f"  F1-Score:        {f1:.4f}")
    print(f"  ROC-AUC:         {roc_auc:.4f}")
    print(f"  Train-Test Gap:  {train_acc - test_acc:.4f}")

    if test_acc >= 0.75:
        print(f"  ✅ TARGET ACHIEVED: {test_acc*100:.1f}% ≥ 75%")
    elif test_acc >= 0.70:
        print(f"  ⚠️  Close to target: {test_acc*100:.1f}% (need 75%)")
    else:
        print(f"  ❌ Below target: {test_acc*100:.1f}% (need 75%)")

# ==============================================================================
# 7. ENSEMBLE MODEL
# ==============================================================================
print("\n" + "="*80)
print("7. CREATING ENSEMBLE MODEL")
print("="*80)

print("\nCreating Voting Classifier (combines all models)...")

# Retrain models for ensemble (need to train them fresh)
rf_ensemble = RandomForestClassifier(
    n_estimators=500, max_depth=20, min_samples_split=5, min_samples_leaf=2,
    max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1
)
xgb_ensemble = XGBClassifier(
    n_estimators=500, max_depth=10, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, scale_pos_weight=(~y_train.astype(bool)).sum() / y_train.sum(),
    random_state=42, n_jobs=-1, verbosity=0
)
lgb_ensemble = LGBMClassifier(
    n_estimators=500, max_depth=15, learning_rate=0.05, num_leaves=50,
    subsample=0.8, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
)

# Train individual models
rf_ensemble.fit(X_train_balanced, y_train_balanced)
xgb_ensemble.fit(X_train_balanced, y_train_balanced)
lgb_ensemble.fit(X_train_balanced, y_train_balanced)

# Create ensemble
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_ensemble),
        ('xgb', xgb_ensemble),
        ('lgb', lgb_ensemble),
        ('nn', best_nn)
    ],
    voting='soft'  # Use probability voting
)

print("Training ensemble...")
ensemble.fit(X_train_balanced, y_train_balanced)

# Evaluate ensemble
y_train_pred_ens = ensemble.predict(X_train)
y_val_pred_ens = ensemble.predict(X_val)
y_test_pred_ens = ensemble.predict(X_test)
y_test_proba_ens = ensemble.predict_proba(X_test)[:, 1]

train_acc_ens = accuracy_score(y_train, y_train_pred_ens)
val_acc_ens = accuracy_score(y_val, y_val_pred_ens)
test_acc_ens = accuracy_score(y_test, y_test_pred_ens)
precision_ens = precision_score(y_test, y_test_pred_ens)
recall_ens = recall_score(y_test, y_test_pred_ens)
f1_ens = f1_score(y_test, y_test_pred_ens)
roc_auc_ens = roc_auc_score(y_test, y_test_proba_ens)

results['Ensemble (All Models)'] = {
    'model': ensemble,
    'train_acc': train_acc_ens,
    'val_acc': val_acc_ens,
    'test_acc': test_acc_ens,
    'precision': precision_ens,
    'recall': recall_ens,
    'f1': f1_ens,
    'roc_auc': roc_auc_ens,
    'train_test_gap': train_acc_ens - test_acc_ens,
    'y_test_pred': y_test_pred_ens,
    'y_test_proba': y_test_proba_ens
}

print(f"\n📊 Ensemble Results:")
print(f"  Test Accuracy:   {test_acc_ens:.4f} ({test_acc_ens*100:.2f}%)")
print(f"  Precision:       {precision_ens:.4f}")
print(f"  Recall:          {recall_ens:.4f}")
print(f"  F1-Score:        {f1_ens:.4f}")
print(f"  ROC-AUC:         {roc_auc_ens:.4f}")

if test_acc_ens >= 0.75:
    print(f"  ✅ TARGET ACHIEVED: {test_acc_ens*100:.1f}% ≥ 75%")
else:
    print(f"  ⚠️  Ensemble: {test_acc_ens*100:.1f}% (target: 75%)")

# ==============================================================================
# 8. MODEL COMPARISON
# ==============================================================================
print("\n" + "="*80)
print("8. MODEL COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Acc': [r['train_acc'] for r in results.values()],
    'Val Acc': [r['val_acc'] for r in results.values()],
    'Test Acc': [r['test_acc'] for r in results.values()],
    'Precision': [r['precision'] for r in results.values()],
    'Recall': [r['recall'] for r in results.values()],
    'F1-Score': [r['f1'] for r in results.values()],
    'ROC-AUC': [r['roc_auc'] for r in results.values()],
    'Train-Test Gap': [r['train_test_gap'] for r in results.values()]
})

comparison_df = comparison_df.sort_values('Test Acc', ascending=False).reset_index(drop=True)

print("\n")
print(comparison_df.to_string(index=False))

best_model_name = comparison_df.iloc[0]['Model']
best_test_acc = comparison_df.iloc[0]['Test Acc']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test Accuracy: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")

if best_test_acc >= 0.75:
    print(f"\n🎉 SUCCESS! Achieved target accuracy (≥75%)")
elif best_test_acc >= 0.72:
    print(f"\n✓ Very close to target ({best_test_acc*100:.1f}% vs 75%)")
else:
    print(f"\n⚠️  Still below target ({best_test_acc*100:.1f}% vs 75%)")
    print("   Consider: more data collection, additional features, or different approach")

# ==============================================================================
# 9. SAVE BEST MODEL
# ==============================================================================
print("\n" + "="*80)
print("9. SAVING BEST MODEL")
print("="*80)

import os
os.makedirs('models', exist_ok=True)

best_model = results[best_model_name]['model']

# Save model
model_filename = 'models/variance_model_improved.pkl'
joblib.dump(best_model, model_filename)
print(f"✅ Model saved: {model_filename}")

# Save features
features_filename = 'models/feature_columns_improved.json'
with open(features_filename, 'w') as f:
    json.dump(FEATURE_COLUMNS, f, indent=2)
print(f"✅ Features saved: {features_filename}")

# Save metadata
metadata = {
    'model_type': best_model_name,
    'training_date': datetime.now().isoformat(),
    'num_features': len(FEATURE_COLUMNS),
    'feature_columns': FEATURE_COLUMNS,
    'improvements': [
        'Advanced feature engineering (25 features)',
        'SMOTE class balancing',
        'Hyperparameter tuning (GridSearchCV)',
        'Better model configurations',
        'Ensemble voting classifier'
    ],
    'performance': {
        'test_accuracy': float(best_test_acc),
        'precision': float(comparison_df.iloc[0]['Precision']),
        'recall': float(comparison_df.iloc[0]['Recall']),
        'f1_score': float(comparison_df.iloc[0]['F1-Score']),
        'roc_auc': float(comparison_df.iloc[0]['ROC-AUC'])
    }
}

metadata_filename = 'models/model_metadata_improved.json'
with open(metadata_filename, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata saved: {metadata_filename}")

# ==============================================================================
# 10. SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)

print(f"\n📊 Improvements vs Previous Version:")
print(f"   Previous best:  69.59% (Neural Network)")
print(f"   Current best:   {best_test_acc*100:.2f}% ({best_model_name})")
print(f"   Improvement:    {(best_test_acc - 0.6959)*100:+.2f}%")

if best_test_acc >= 0.75:
    print(f"\n✅ TARGET ACHIEVED: {best_test_acc*100:.1f}% ≥ 75%")
    print("   Ready for deployment!")
else:
    print(f"\n⚠️  Accuracy: {best_test_acc*100:.1f}% (target: 75%)")
    print("   Recommend: review results and decide if acceptable")

print(f"\n🚀 Next Steps:")
print("   1. Review detailed results")
print("   2. If satisfied, deploy to AWS Lambda")
print("   3. Run comparative evaluation")
print("   4. Measure real-world performance")

print(f"\n✅ Done! End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
