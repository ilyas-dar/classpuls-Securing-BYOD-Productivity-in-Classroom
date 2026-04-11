# ml_pipeline/training/train_anomaly.py
# ============================================================
# Train the Autoencoder anomaly detector with proper evaluation.
# Usage: python -m ml_pipeline.training.train_anomaly
# ============================================================

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from backend.models.anomaly_detector import AnomalyDetector

DATA_PATH  = "backend/data/synthetic_classroom_data.csv"
MODEL_OUT  = "ml_pipeline/models/saved_models/anomaly_detector"


def main():
    print("=" * 55)
    print("  Training Autoencoder anomaly detector")
    print("=" * 55)

    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: {DATA_PATH} not found.")
        print("Run: python -m backend.data.synthetic_data_generator\n")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records")

    # Label: engagement < 0.4 = anomalous
    df["is_anomaly"] = (df["engagement_score"] < 0.4).astype(int)
    anomaly_rate = df["is_anomaly"].mean()
    print(f"Anomaly rate in dataset: {anomaly_rate:.1%}")
    print(f"  Normal records  : {(df['is_anomaly']==0).sum()}")
    print(f"  Anomaly records : {(df['is_anomaly']==1).sum()}")

    # Train ONLY on normal records — this is the key insight:
    # the autoencoder learns what normal looks like,
    # anything it can't reconstruct well = anomaly
    normal_df = df[df["is_anomaly"] == 0].copy()
    print(f"\nTraining on {len(normal_df)} normal records only...")

    detector = AnomalyDetector()
    detector.train(normal_df, epochs=40, validation_split=0.15)

    # ── Evaluation ──────────────────────────────────────────
    print("\nEvaluating on full dataset...")
    X_all = AnomalyDetector.extract_features(df)
    X_scaled = detector.scaler.transform(X_all)
    preds = detector.model.predict(X_scaled, verbose=0)
    mse = np.mean(np.square(preds - X_scaled), axis=1)
    anomaly_scores = mse / (detector.threshold + 1e-10)

    # Binary prediction at threshold = 1.0
    y_pred = (anomaly_scores > 1.0).astype(int)
    y_true = df["is_anomaly"].values

    print("\nClassification Report (detecting disengaged students):")
    print(classification_report(y_true, y_pred,
          target_names=["Normal", "Anomaly"], digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  True Negatives  (correctly normal)  : {tn}")
    print(f"  False Positives (false alarms)       : {fp}")
    print(f"  False Negatives (missed anomalies)   : {fn}")
    print(f"  True Positives  (caught anomalies)   : {tp}")

    # Score distribution
    print(f"\nAnomaly score distribution:")
    print(f"  Normal   — mean: {anomaly_scores[y_true==0].mean():.3f}  max: {anomaly_scores[y_true==0].max():.3f}")
    print(f"  Anomaly  — mean: {anomaly_scores[y_true==1].mean():.3f}  max: {anomaly_scores[y_true==1].max():.3f}")

    # Per-archetype breakdown
    if "archetype" in df.columns:
        print(f"\nAnomaly detection rate by student archetype:")
        df["predicted_anomaly"] = y_pred
        for arch in df["archetype"].unique():
            mask = df["archetype"] == arch
            detected = df.loc[mask & (df["is_anomaly"]==1), "predicted_anomaly"].mean()
            false_alarm = df.loc[mask & (df["is_anomaly"]==0), "predicted_anomaly"].mean()
            print(f"  {arch:<30} recall: {detected:.1%}  false-alarm: {false_alarm:.1%}")

    # Save
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    detector.save(MODEL_OUT)
    print(f"\nModel saved → {MODEL_OUT}_model.h5 / _config.pkl")


if __name__ == "__main__":
    main()
