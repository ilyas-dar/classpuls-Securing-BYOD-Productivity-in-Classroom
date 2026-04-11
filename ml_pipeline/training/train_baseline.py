# ml_pipeline/training/train_baseline.py
# ============================================================
# Train XGBoost baseline engagement model with proper evaluation.
# Usage: python -m ml_pipeline.training.train_baseline
# ============================================================

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from backend.models.baseline_model import BaselineEngagementModel

DATA_DIR  = "backend/data"
MODEL_OUT = "ml_pipeline/models/saved_models/baseline_model.pkl"


def build_features(df: pd.DataFrame) -> tuple:
    feats = pd.DataFrame()

    # Time features (cyclical)
    feats["hour_sin"]     = np.sin(2 * np.pi * df["hour"] / 24)
    feats["hour_cos"]     = np.cos(2 * np.pi * df["hour"] / 24)
    feats["day_sin"]      = np.sin(2 * np.pi * df["day_of_week"] / 5)
    feats["day_cos"]      = np.cos(2 * np.pi * df["day_of_week"] / 5)

    # Student features
    feats["gpa_norm"]        = df["gpa"] / 4.0
    feats["prev_score_norm"] = df["prev_score"] / 100.0
    feats["grade_level"]     = df["grade_level"]
    feats["learning_pace"]   = df["learning_pace"].map({"slow": 0, "medium": 1, "fast": 2})

    # Subject one-hot
    for subj in ["Math", "Science", "English", "History", "CS"]:
        feats[f"subject_{subj}"] = (df["subject"] == subj).astype(int)

    # Activity one-hot
    for act in ["lecture", "group_work", "individual_task", "quiz"]:
        feats[f"activity_{act}"] = (df["activity_type"] == act).astype(int)

    # Archetype one-hot (available in synthetic data — not at inference time,
    # but helps the model learn patterns; at inference we drop it)
    if "archetype" in df.columns:
        for arch in ["high_achiever", "bright_but_bored", "steady_worker",
                     "struggling_but_trying", "disengaged", "anxious_high_performer"]:
            feats[f"arch_{arch}"] = (df["archetype"] == arch).astype(int)

    return feats.fillna(0), df["engagement_score"]


def evaluate(y_true, y_pred, split_name: str):
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    # Treat engagement < 0.5 as "disengaged" — compute classification metrics
    y_true_bin = (y_true < 0.5).astype(int)
    y_pred_bin = (y_pred < 0.5).astype(int)
    tp = ((y_pred_bin == 1) & (y_true_bin == 1)).sum()
    fp = ((y_pred_bin == 1) & (y_true_bin == 0)).sum()
    fn = ((y_pred_bin == 0) & (y_true_bin == 1)).sum()
    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)

    print(f"\n  [{split_name}]")
    print(f"    RMSE      = {np.sqrt(mse):.4f}")
    print(f"    MAE       = {mae:.4f}")
    print(f"    R²        = {r2:.4f}")
    print(f"    Precision = {precision:.4f}  (detecting disengaged students)")
    print(f"    Recall    = {recall:.4f}  (catching disengaged students)")
    print(f"    F1        = {f1:.4f}")
    return r2


def main():
    print("=" * 55)
    print("  Training XGBoost baseline engagement model")
    print("=" * 55)

    sessions_path = os.path.join(DATA_DIR, "synthetic_classroom_data.csv")
    if not os.path.exists(sessions_path):
        print(f"\nERROR: {sessions_path} not found.")
        print("Run: python -m backend.data.synthetic_data_generator\n")
        sys.exit(1)

    df = pd.read_csv(sessions_path)
    print(f"Loaded {len(df)} records | {df['student_id'].nunique()} students | {df['class_id'].nunique()} sessions")
    print(f"Engagement — mean: {df['engagement_score'].mean():.3f}  std: {df['engagement_score'].std():.3f}")

    X, y = build_features(df)
    print(f"Feature matrix: {X.shape}")

    # Train / validation / test split (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    print(f"Split — train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}")

    # Train
    print("\nTraining...")
    model = BaselineEngagementModel()
    model.train(X_train, y_train)

    # Evaluate on all splits
    print("\nEvaluation results:")
    evaluate(y_train, model.predict(X_train), "Train")
    evaluate(y_val,   model.predict(X_val),   "Validation")
    r2_test = evaluate(y_test, model.predict(X_test), "Test  (held out)")

    # Feature importance
    importance = model.feature_importance()
    print("\nTop 10 most important features:")
    for feat, score in list(importance.items())[:10]:
        bar = "█" * int(score * 40)
        print(f"  {feat:<35} {bar} {score:.4f}")

    # 5-fold cross-validation on full dataset
    print("\n5-fold cross-validation (full dataset)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        m = BaselineEngagementModel()
        m.train(X.iloc[tr_idx], y.iloc[tr_idx])
        r2 = r2_score(y.iloc[val_idx], m.predict(X.iloc[val_idx]))
        cv_r2.append(r2)
        print(f"  Fold {fold+1}: R² = {r2:.4f}")
    print(f"  Mean R² = {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")

    # Save
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    print(f"\nModel saved → {MODEL_OUT}")
    print(f"Model type  : {model.model_type}")


if __name__ == "__main__":
    main()
