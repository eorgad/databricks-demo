# Databricks notebook source
# MAGIC %md
# MAGIC # 🧬 Drug Discovery Demo - Part 3: Toxicity Model Training
# MAGIC
# MAGIC ## Overview
# MAGIC Train machine learning models to predict molecular toxicity:
# MAGIC - Prepare training data with engineered features
# MAGIC - Train multiple classification models
# MAGIC - Hyperparameter tuning with Hyperopt
# MAGIC - MLflow experiment tracking
# MAGIC - Model evaluation and comparison
# MAGIC - Register best model to MLflow Model Registry
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Build ML pipelines for molecular data
# MAGIC - Use MLflow for experiment tracking
# MAGIC - Perform distributed hyperparameter tuning
# MAGIC - Evaluate classification models
# MAGIC - Deploy models to production

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score
import xgboost as xgb

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature

# Hyperopt for tuning
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, space_eval

# Spark
from pyspark.sql.functions import col
import warnings
warnings.filterwarnings('ignore')

print("✓ Libraries imported successfully")

# COMMAND ----------

# Configuration
CATALOG = "main"
SCHEMA = "drug_discovery"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.molecular_features"

# MLflow experiment
EXPERIMENT_NAME = "/Users/your-email@company.com/drug_discovery_toxicity"
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"Configuration:")
print(f"  Feature table: {FEATURE_TABLE}")
print(f"  MLflow experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load and Prepare Data

# COMMAND ----------

# DBTITLE 1,Load Feature Tables
# Load train, validation, and test sets
train_df = spark.table(f"{FEATURE_TABLE}").toPandas()
val_df = spark.table(f"{FEATURE_TABLE}_val").toPandas()
test_df = spark.table(f"{FEATURE_TABLE}_test").toPandas()

print(f"✓ Data loaded:")
print(f"  Training: {len(train_df):,} molecules")
print(f"  Validation: {len(val_df):,} molecules")
print(f"  Test: {len(test_df):,} molecules")

# COMMAND ----------

# DBTITLE 1,Prepare Features and Target
def prepare_ml_data(df, include_fingerprints=True):
    """
    Prepare features and target for ML
    
    Args:
        df: Input dataframe
        include_fingerprints: Whether to include fingerprint features
    
    Returns:
        X (features), y (target)
    """
    # Select descriptor features
    descriptor_cols = [c for c in df.columns if c.startswith('calc_')]
    
    X_descriptors = df[descriptor_cols].fillna(0)
    
    if include_fingerprints:
        # Expand fingerprint array into individual columns
        fp_array = np.array(df['morgan_fp'].tolist())
        fp_cols = [f'fp_{i}' for i in range(fp_array.shape[1])]
        X_fp = pd.DataFrame(fp_array, columns=fp_cols, index=df.index)
        
        # Combine descriptors and fingerprints
        X = pd.concat([X_descriptors, X_fp], axis=1)
    else:
        X = X_descriptors
    
    # Target variable
    y = df['is_toxic'].astype(int)
    
    return X, y

# Prepare training data
print("Preparing features...")
X_train, y_train = prepare_ml_data(train_df, include_fingerprints=True)
X_val, y_val = prepare_ml_data(val_df, include_fingerprints=True)
X_test, y_test = prepare_ml_data(test_df, include_fingerprints=True)

print(f"✓ Features prepared:")
print(f"  Feature dimensions: {X_train.shape}")
print(f"  Target distribution: {y_train.value_counts().to_dict()}")
print(f"  Class balance: {y_train.mean()*100:.1f}% toxic")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Baseline Model Training

# COMMAND ----------

# DBTITLE 1,Train Baseline Random Forest
print("Training baseline Random Forest model...")

with mlflow.start_run(run_name="Baseline_RandomForest") as run:
    
    # Model parameters
    params = {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Log parameters
    mlflow.log_params(params)
    
    # Train model
    rf_model = RandomForestClassifier(**params)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = rf_model.predict(X_train)
    y_val_pred = rf_model.predict(X_val)
    y_train_proba = rf_model.predict_proba(X_train)[:, 1]
    y_val_proba = rf_model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    train_metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'train_roc_auc': roc_auc_score(y_train, y_train_proba)
    }
    
    val_metrics = {
        'val_accuracy': accuracy_score(y_val, y_val_pred),
        'val_precision': precision_score(y_val, y_val_pred),
        'val_recall': recall_score(y_val, y_val_pred),
        'val_f1': f1_score(y_val, y_val_pred),
        'val_roc_auc': roc_auc_score(y_val, y_val_proba)
    }
    
    # Log metrics
    mlflow.log_metrics({**train_metrics, **val_metrics})
    
    # Log model
    signature = infer_signature(X_train, y_train_pred)
    mlflow.sklearn.log_model(rf_model, "model", signature=signature)
    
    # Log feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    mlflow.log_table(feature_importance.head(20), "feature_importance.json")
    
    print("\n✓ Baseline model trained!")
    print(f"\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")

# COMMAND ----------

# DBTITLE 1,Visualize Baseline Results
# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Validation confusion matrix
cm_val = confusion_matrix(y_val, y_val_pred)
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Validation Confusion Matrix', fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_xticklabels(['Non-Toxic', 'Toxic'])
axes[0].set_yticklabels(['Non-Toxic', 'Toxic'])

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
axes[1].plot(fpr, tpr, label=f'ROC-AUC = {val_metrics["val_roc_auc"]:.3f}', linewidth=2)
axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve - Validation Set', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Top Features by Importance
# Plot top 20 features
top_features = feature_importance.head(20)

plt.figure(figsize=(10, 8))
plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Top 20 Most Important Features', fontweight='bold', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Hyperparameter Tuning with Hyperopt

# COMMAND ----------

# DBTITLE 1,Define Search Space
# Define hyperparameter search space for Random Forest
search_space = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200]),
    'max_depth': hp.choice('max_depth', [10, 15, 20, 25, 30]),
    'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
    'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4]),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', None])
}

print("Search space defined:")
for param, space in search_space.items():
    print(f"  {param}: {space}")

# COMMAND ----------

# DBTITLE 1,Define Objective Function
def train_and_evaluate(params):
    """
    Objective function for hyperparameter tuning
    
    Args:
        params: Hyperparameter dict
    
    Returns:
        Loss dict for Hyperopt
    """
    with mlflow.start_run(nested=True):
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(
            **params,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_val_proba = model.predict_proba(X_val)[:, 1]
        val_roc_auc = roc_auc_score(y_val, y_val_proba)
        
        # Log metric
        mlflow.log_metric("val_roc_auc", val_roc_auc)
        
        # Return negative AUC (Hyperopt minimizes)
        return {'loss': -val_roc_auc, 'status': STATUS_OK}

print("✓ Objective function defined")

# COMMAND ----------

# DBTITLE 1,Run Hyperparameter Optimization
print("Starting hyperparameter optimization...")
print("This will run 20 trials in parallel...\n")

with mlflow.start_run(run_name="Hyperopt_Tuning"):
    
    # Use SparkTrials for parallel tuning
    spark_trials = SparkTrials(parallelism=4)
    
    # Run optimization
    best_params = fmin(
        fn=train_and_evaluate,
        space=search_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=spark_trials
    )
    
    # Convert index choices back to actual values
    best_params_actual = space_eval(search_space, best_params)
    
    print("\n✓ Hyperparameter optimization complete!")
    print(f"\nBest parameters:")
    for param, value in best_params_actual.items():
        print(f"  {param}: {value}")
    
    # Log best parameters
    mlflow.log_params(best_params_actual)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train Final Model with Best Parameters

# COMMAND ----------

# DBTITLE 1,Train Optimized Model
print("Training final model with optimized parameters...")

with mlflow.start_run(run_name="Optimized_RandomForest") as run:
    
    # Add fixed parameters
    final_params = {
        **best_params_actual,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Log parameters
    mlflow.log_params(final_params)
    
    # Train model
    final_model = RandomForestClassifier(**final_params)
    final_model.fit(X_train, y_train)
    
    # Predictions on all sets
    y_train_pred = final_model.predict(X_train)
    y_val_pred = final_model.predict(X_val)
    y_test_pred = final_model.predict(X_test)
    
    y_train_proba = final_model.predict_proba(X_train)[:, 1]
    y_val_proba = final_model.predict_proba(X_val)[:, 1]
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    
    # Calculate comprehensive metrics
    def calculate_metrics(y_true, y_pred, y_proba, prefix):
        return {
            f'{prefix}_accuracy': accuracy_score(y_true, y_pred),
            f'{prefix}_precision': precision_score(y_true, y_pred),
            f'{prefix}_recall': recall_score(y_true, y_pred),
            f'{prefix}_f1': f1_score(y_true, y_pred),
            f'{prefix}_roc_auc': roc_auc_score(y_true, y_proba)
        }
    
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba, 'train')
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba, 'val')
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba, 'test')
    
    # Log all metrics
    mlflow.log_metrics({**train_metrics, **val_metrics, **test_metrics})
    
    # Log model with signature
    signature = infer_signature(X_train, y_train_pred)
    mlflow.sklearn.log_model(
        final_model, 
        "model",
        signature=signature,
        registered_model_name="toxicity_predictor"
    )
    
    # Save run ID for later use
    run_id = run.info.run_id
    
    print("✓ Final model trained and logged!")
    print(f"\nTest Set Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

# COMMAND ----------

# DBTITLE 1,Comprehensive Evaluation
# Print classification report
print("Classification Report (Test Set):")
print("="*60)
print(classification_report(y_test, y_test_pred, target_names=['Non-Toxic', 'Toxic']))

# Confusion matrix
cm_test = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix (Test Set):")
print(cm_test)
print(f"\nTrue Negatives: {cm_test[0, 0]}")
print(f"False Positives: {cm_test[0, 1]}")
print(f"False Negatives: {cm_test[1, 0]}")
print(f"True Positives: {cm_test[1, 1]}")

# COMMAND ----------

# DBTITLE 1,Final Model Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Test Set Confusion Matrix', fontweight='bold')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xticklabels(['Non-Toxic', 'Toxic'])
axes[0, 0].set_yticklabels(['Non-Toxic', 'Toxic'])

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
axes[0, 1].plot(fpr, tpr, label=f'ROC-AUC = {test_metrics["test_roc_auc"]:.3f}', linewidth=2)
axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve (Test Set)', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Prediction Distribution
axes[1, 0].hist([y_test_proba[y_test == 0], y_test_proba[y_test == 1]], 
               bins=50, label=['Non-Toxic', 'Toxic'], 
               color=['green', 'red'], alpha=0.6, edgecolor='black')
axes[1, 0].set_xlabel('Predicted Toxicity Probability')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Prediction Distribution by True Label', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Feature Importance (Top 15)
feature_imp = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

axes[1, 1].barh(range(len(feature_imp)), feature_imp['importance'], color='steelblue')
axes[1, 1].set_yticks(range(len(feature_imp)))
axes[1, 1].set_yticklabels(feature_imp['feature'])
axes[1, 1].set_xlabel('Importance')
axes[1, 1].set_title('Top 15 Features', fontweight='bold')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Compare Multiple Algorithms

# COMMAND ----------

# DBTITLE 1,Train XGBoost Model
print("Training XGBoost model for comparison...")

with mlflow.start_run(run_name="XGBoost_Classifier"):
    
    # Calculate scale_pos_weight for imbalanced classes
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    params_xgb = {
        'n_estimators': 150,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'auc'
    }
    
    mlflow.log_params(params_xgb)
    
    # Train
    xgb_model = xgb.XGBClassifier(**params_xgb)
    xgb_model.fit(X_train, y_train)
    
    # Evaluate
    y_test_pred_xgb = xgb_model.predict(X_test)
    y_test_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    test_metrics_xgb = calculate_metrics(y_test, y_test_pred_xgb, y_test_proba_xgb, 'test')
    mlflow.log_metrics(test_metrics_xgb)
    
    # Log model
    signature = infer_signature(X_train, y_test_pred_xgb)
    mlflow.xgboost.log_model(xgb_model, "model", signature=signature)
    
    print("✓ XGBoost model trained!")
    print(f"\nTest Set Performance:")
    for metric, value in test_metrics_xgb.items():
        print(f"  {metric}: {value:.4f}")

# COMMAND ----------

# DBTITLE 1,Model Comparison
# Compare all models
comparison_df = pd.DataFrame({
    'Model': ['Random Forest (Baseline)', 'Random Forest (Optimized)', 'XGBoost'],
    'Accuracy': [
        val_metrics['val_accuracy'],  # baseline
        test_metrics['test_accuracy'],  # optimized
        test_metrics_xgb['test_accuracy']  # xgboost
    ],
    'Precision': [
        val_metrics['val_precision'],
        test_metrics['test_precision'],
        test_metrics_xgb['test_precision']
    ],
    'Recall': [
        val_metrics['val_recall'],
        test_metrics['test_recall'],
        test_metrics_xgb['test_recall']
    ],
    'F1-Score': [
        val_metrics['val_f1'],
        test_metrics['test_f1'],
        test_metrics_xgb['test_f1']
    ],
    'ROC-AUC': [
        val_metrics['val_roc_auc'],
        test_metrics['test_roc_auc'],
        test_metrics_xgb['test_roc_auc']
    ]
})

print("\nModel Comparison:")
print("="*80)
display(comparison_df)

# Visualize comparison
comparison_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].plot(
    kind='bar', figsize=(12, 6), rot=15
)
plt.title('Model Performance Comparison', fontweight='bold', fontsize=14)
plt.ylabel('Score')
plt.ylim(0.5, 1.0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary

# COMMAND ----------

print("="*70)
print("TOXICITY MODEL TRAINING COMPLETE")
print("="*70)

print(f"\n🎯 Best Model: Random Forest (Optimized)")
print(f"\n📊 Test Set Performance:")
print(f"  • Accuracy:  {test_metrics['test_accuracy']:.4f}")
print(f"  • Precision: {test_metrics['test_precision']:.4f}")
print(f"  • Recall:    {test_metrics['test_recall']:.4f}")
print(f"  • F1-Score:  {test_metrics['test_f1']:.4f}")
print(f"  • ROC-AUC:   {test_metrics['test_roc_auc']:.4f}")

print(f"\n💾 Model Registry:")
print(f"  • Model name: toxicity_predictor")
print(f"  • MLflow run ID: {run_id}")
print(f"  • Experiment: {EXPERIMENT_NAME}")

print(f"\n🔬 Model Interpretation:")
print(f"  • Top feature: {feature_imp.iloc[0]['feature']}")
print(f"  • Feature count: {len(X_train.columns)}")
print(f"  • Training time: ~1-2 minutes")

print(f"\n➡️  Next Steps:")
print(f"  1. Run Notebook 04: Solubility Model Training")
print(f"  2. Run Notebook 05: Batch Inference")
print(f"  3. Deploy model to production")
print(f"  4. Monitor model performance")

print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **End of Notebook 03: Toxicity Model Training**
# MAGIC
# MAGIC Continue to [Notebook 04: Solubility Model Training](...)
