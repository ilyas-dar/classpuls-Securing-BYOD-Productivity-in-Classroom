# backend/models/anomaly_detector.py

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class AnomalyDetector:
    INPUT_DIM = 7

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.threshold: float = 0.01

    def _build_model(self):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is not installed.")
        inp = layers.Input(shape=(self.INPUT_DIM,))
        x = layers.Dense(16, activation="relu")(inp)
        x = layers.Dense(8, activation="relu")(x)
        x = layers.Dense(4, activation="relu")(x)
        x = layers.Dense(8, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
        out = layers.Dense(self.INPUT_DIM, activation="linear")(x)
        model = keras.Model(inp, out)
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    @staticmethod
    def extract_features(df) -> np.ndarray:
        import pandas as pd
        features = pd.DataFrame()
        features["app_switches_rate"]        = df["app_switches"] / 10
        features["keystroke_intensity_norm"] = df["keystroke_intensity"] / 100
        features["inactivity_rate"]          = df["inactivity_periods"] / 10
        features["poll_participation"]       = df["poll_participation"]
        features["collaboration_rate"]       = df["collaboration_actions"] / 20
        features["switch_inactivity_ratio"]  = (
            features["app_switches_rate"] / (features["inactivity_rate"] + 0.01)
        )
        features["keystroke_collab_ratio"]   = (
            features["keystroke_intensity_norm"] / (features["collaboration_rate"] + 0.01)
        )
        return features.values

    def train(self, df, epochs: int = 40, validation_split: float = 0.15):
        X = self.extract_features(df)
        X_scaled = self.scaler.fit_transform(X)
        self.model = self._build_model()
        self.model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            shuffle=True,
            verbose=1,
        )
        preds = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.square(preds - X_scaled), axis=1)
        self.threshold = float(np.percentile(mse, 95))
        print(f"Anomaly threshold set to: {self.threshold:.6f}")

    def score(self, activity_dict: dict) -> float:
        import pandas as pd
        row = pd.DataFrame([activity_dict])
        X = self.extract_features(row)
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled, verbose=0)
        mse = float(np.mean(np.square(pred - X_scaled)))
        return mse / (self.threshold + 1e-10)

    def save(self, path_prefix: str):
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        # New .keras format — works with Keras 3 / TF 2.17
        self.model.save(f"{path_prefix}_model.keras")
        joblib.dump(
            {"scaler": self.scaler, "threshold": self.threshold},
            f"{path_prefix}_config.pkl"
        )
        print(f"Anomaly detector saved → {path_prefix}_model.keras / _config.pkl")

    @classmethod
    def load(cls, path_prefix: str) -> "AnomalyDetector":
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is not installed.")
        instance = cls()
        # Try new .keras format first, fall back to old .h5
        keras_path = f"{path_prefix}_model.keras"
        h5_path    = f"{path_prefix}_model.h5"
        if os.path.exists(keras_path):
            instance.model = keras.models.load_model(keras_path)
        elif os.path.exists(h5_path):
            instance.model = keras.models.load_model(h5_path)
        else:
            raise FileNotFoundError(f"No model found at {keras_path} or {h5_path}")
        config = joblib.load(f"{path_prefix}_config.pkl")
        instance.scaler    = config["scaler"]
        instance.threshold = config["threshold"]
        return instance