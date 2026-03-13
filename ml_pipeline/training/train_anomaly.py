import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class AnomalyDetector:
    def __init__(self):
        # isolation forest works well here,no need for heavy autoencoders
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,  # expecting roughly 5% anomalies in real data
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.threshold = None
        self.feature_names = []

    def prepare_features(self, df):
        features = pd.DataFrame()
        # normalize everything to roughly 0-1 range
        features['app_switches_rate']       = df['app_switches'] / 10
        features['keystroke_intensity_norm'] = df['keystroke_intensity'] / 100
        features['inactivity_rate']         = df['inactivity_periods'] / 10
        features['poll_participation']      = df['poll_participation']
        features['collaboration_rate']      = df['collaboration_actions'] / 20
        # ratio features catch patterns that individual cols miss
        features['switch_inactivity_ratio'] = (
            features['app_switches_rate'] / (features['inactivity_rate'] + 0.01)
        )
        features['keystroke_collab_ratio']  = (
            features['keystroke_intensity_norm'] / (features['collaboration_rate'] + 0.01)
        )
        return features

    def train(self, df):
        print("Preparing anomaly features...")
        X = self.prepare_features(df)
        self.feature_names = X.columns.tolist()

        print("Fitting scaler...")
        X_scaled = self.scaler.fit_transform(X)

        print("Training IsolationForest anomaly detector...")
        self.model.fit(X_scaled)

        # flip sign so higher score = more anomalous,easier to reason about
        scores = -self.model.decision_function(X_scaled)
        # threshold at 95th percentile so only worst cases get flagged
        self.threshold = float(np.percentile(scores, 95))

        n_anomalies = int((scores > self.threshold).sum())
        print(f"Training complete.")
        print(f"Anomaly threshold (95th pct): {self.threshold:.4f}")
        print(f"Anomalies detected in training set: {n_anomalies} / {len(df)}")
        return scores

    def detect(self, df):
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        scores = -self.model.decision_function(X_scaled)
        is_anomaly = scores > self.threshold
        # normalise so 1.0 = threshold,anything above is anomalous
        anomaly_score = scores / (self.threshold + 1e-9)
        return is_anomaly, anomaly_score

    def save(self, path='ml_pipeline/models/saved_models'):
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            'model':         self.model,
            'scaler':        self.scaler,
            'threshold':     self.threshold,
            'feature_names': self.feature_names,
        }, os.path.join(path, 'anomaly_detector_config.pkl'))
        print(f"Anomaly detector saved to {path}/anomaly_detector_config.pkl")


if __name__ == '__main__':
    sessions_df = pd.read_csv('backend/data/synthetic_classroom_data.csv')
    detector = AnomalyDetector()
    detector.train(sessions_df)
    detector.save()