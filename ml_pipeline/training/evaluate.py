import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score, classification_report

def evaluate_baseline(model_path='ml_pipeline/models/saved_models/baseline_model.pkl'):
    data = joblib.load(model_path)
    print("Baseline model loaded.")
    print(f"Features: {data['feature_names']}")

def evaluate_anomaly(model_path='ml_pipeline/models/saved_models/anomaly_detector_config.pkl'):
    data = joblib.load(model_path)
    print("Anomaly detector loaded.")
    # threshold tells you how sensitive the detector is
    print(f"Threshold: {data['threshold']:.4f}")
    print(f"Features: {data['feature_names']}")

if __name__ == '__main__':
    evaluate_baseline()
    evaluate_anomaly()