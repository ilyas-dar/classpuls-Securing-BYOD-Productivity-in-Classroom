# backend/models/baseline_model.py
# XGBoost baseline engagement model

import numpy as np
import joblib
import os

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestRegressor
    XGBOOST_AVAILABLE = False

from sklearn.preprocessing import StandardScaler


class BaselineEngagementModel:
    """
    Predicts expected (baseline) engagement for a student given context.
    Uses XGBoost if available, falls back to Random Forest.
    """

    def __init__(self):
        if XGBOOST_AVAILABLE:
            self.model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
        else:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

        self.scaler = StandardScaler()
        self.feature_names: list = []
        self.model_type = "XGBoost" if XGBOOST_AVAILABLE else "RandomForest"

    def train(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return np.clip(self.model.predict(X_scaled), 0.0, 1.0)

    def feature_importance(self) -> dict:
        if not self.feature_names:
            return {}
        if XGBOOST_AVAILABLE:
            scores = self.model.feature_importances_
        else:
            scores = self.model.feature_importances_
        return dict(sorted(zip(self.feature_names, scores), key=lambda x: x[1], reverse=True))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler,
                     "feature_names": self.feature_names, "model_type": self.model_type}, path)
        print(f"{self.model_type} baseline model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "BaselineEngagementModel":
        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.feature_names = data.get("feature_names", [])
        instance.model_type = data.get("model_type", "unknown")
        return instance
