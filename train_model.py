"""
=============================================================================
train_model.py
=============================================================================
Trains a Supervised Machine Learning model (Random Forest Classifier) to
predict student academic performance categories.

Educational Objectives for B.Tech AI Lab:
1. Data Ingestion: Load structured tabular data using Pandas.
2. Feature Separation: Distinguish input features (X) from the target label (y).
3. Train-Test Split: 80% training set, 20% testing set with stratified sampling.
4. Model Training: Train a Random Forest Classifier.
5. Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
6. Baseline Comparison: Compare with Logistic Regression for educational analysis.
7. Model Persistence: Save the trained model with Joblib for production inference.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def train_and_evaluate():
    print("=" * 70)
    print("AI-Based Student Performance Prediction System: Model Training")
    print("=" * 70)
    
    # 1. Verify and Load Dataset
    data_path = os.path.join("data", "student_performance.csv")
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at '{data_path}'.")
        print("Please run 'python generate_dataset.py' first.")
        sys.exit(1)
        
    print(f"[STEP 1] Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records with columns: {list(df.columns)}")
    
    # 2. Define Features (X) and Target (y)
    feature_cols = [
        "attendance",
        "internal_marks",
        "assignment_completion",
        "previous_semester_percentage",
        "study_hours"
    ]
    target_col = "performance_category"
    
    X = df[feature_cols]
    y = df[target_col]
    
    print("\n[STEP 2] Features and Target defined:")
    print(f" - Input Features (X): {feature_cols}")
    print(f" - Target Label (y): '{target_col}' with classes: {sorted(y.unique())}")
    
    # 3. Train-Test Split (80% Train, 20% Test)
    # stratify=y guarantees identical class proportions in both sets
    # random_state ensures reproducible results
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[STEP 3] Splitting data: 80% Train ({len(X_train)} samples), 20% Test ({len(X_test)} samples)")
    
    # 4. Train Primary Model: Random Forest Classifier
    # Why Random Forest?
    # - Ensemble of decision trees using bagging (bootstrap aggregation)
    # - Resilient to overfitting compared to individual decision trees
    # - Invariant to feature scaling (no need for min-max or z-score normalization)
    # - Handles non-linear interactions among study habits, attendance, and exam scores
    print("\n[STEP 4] Training Primary Model: Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,     # 100 decision trees in the forest
        max_depth=6,          # Restrict depth to prevent overfitting
        min_samples_split=4,  # Minimum samples required to split an internal node
        random_state=42       # Reproducibility
    )
    rf_model.fit(X_train, y_train)
    
    # 5. Evaluate Random Forest
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    
    print("\n" + "=" * 70)
    print("PRIMARY MODEL EVALUATION: RANDOM FOREST")
    print("=" * 70)
    print(f"Accuracy Score: {rf_accuracy:.4f} ({rf_accuracy * 100:.2f}%)")
    
    # Target class order for consistent confusion matrix and report
    class_labels = ["At Risk", "Average", "Good"]
    
    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred_rf, target_names=class_labels, digits=4))
    
    cm_rf = confusion_matrix(y_test, y_pred_rf, labels=class_labels)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm_rf, index=[f"Actual {c}" for c in class_labels], columns=[f"Pred {c}" for c in class_labels]))
    
    # Feature Importance Analysis
    importances = rf_model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    print("\nFeature Importance Rankings:")
    for feature, imp in feat_imp.items():
        print(f" - {feature:<30}: {imp:.4f} ({imp * 100:.1f}%)")
        
    # 6. Educational Comparison: Train Logistic Regression Baseline
    print("\n" + "-" * 70)
    print("[EDUCATIONAL COMPARISON] Baseline Model: Logistic Regression")
    print("-" * 70)
    lr_pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f} ({lr_accuracy * 100:.2f}%)")
    print(f"Random Forest Accuracy:       {rf_accuracy:.4f} ({rf_accuracy * 100:.2f}%)")
    if rf_accuracy >= lr_accuracy:
        print(">> Random Forest achieved higher or comparable performance due to its ability")
        print("   to capture non-linear student behavior patterns.")
    print("-" * 70)
    
    # 7. Generate and Save Confusion Matrix Visualization
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    cm_plot_path = os.path.join(model_dir, "confusion_matrix.png")
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm_rf,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=True
    )
    plt.title("Confusion Matrix - Student Performance Random Forest", fontsize=12, pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("Actual Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"\n[STEP 5] Confusion matrix plot saved to '{cm_plot_path}'")
    
    # 8. Save Trained Model to Disk using Joblib
    model_path = os.path.join(model_dir, "student_performance_model.pkl")
    # Save a dictionary containing the model, feature names, and metadata
    model_payload = {
        "model": rf_model,
        "feature_names": feature_cols,
        "target_classes": class_labels,
        "model_name": "RandomForestClassifier",
        "accuracy": rf_accuracy
    }
    joblib.dump(model_payload, model_path)
    print(f"[STEP 6] Trained model successfully saved to '{model_path}'")
    print("=" * 70)
    print("Model Training & Evaluation Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    train_and_evaluate()
