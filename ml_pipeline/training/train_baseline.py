import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

class BaselineEngagementModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_columns = []
        
    def prepare_features(self, df, students_df):
        # merge session data with student profiles on student_id
        merged = df.merge(students_df, on='student_id', how='left')
        
        features = pd.DataFrame()
        
        # sin cos encoding so model knows hour is cyclical not linear
        features['hour_sin'] = np.sin(2 * np.pi * merged['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * merged['hour'] / 24)
        
        features['grade_level'] = merged['grade_level']
        features['gpa_normalized'] = merged['gpa'] / 4.0
        features['learning_pace_encoded'] = merged['learning_pace'].map(
            {'slow': 0, 'medium': 1, 'fast': 2})
        
        # one hot encode subject
        subject_dummies = pd.get_dummies(merged['subject'], prefix='subject')
        features = pd.concat([features, subject_dummies], axis=1)
        
        # prev_scores comes as dict string from csv,parse it manually
        def parse_prev_scores(x):
            if isinstance(x, dict):
                return np.mean(list(x.values())) if x else 0.5
            try:
                import ast
                d = ast.literal_eval(str(x))
                return np.mean(list(d.values())) if isinstance(d, dict) else 0.5
            except Exception:
                # fallback if parsing fails for whatever reason
                return 0.5
        features['avg_prev_score'] = merged['prev_scores'].apply(parse_prev_scores)
        
        # one hot encode activity type
        activity_dummies = pd.get_dummies(merged['activity_type'], prefix='activity')
        features = pd.concat([features, activity_dummies], axis=1)
        
        return features, merged['engagement_score']

    def print_metrics(self, name, y_test, y_pred, cv_scores):
        # print all metrics for a model in a clean block
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        print(f"\n{name}")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}")
        print(f"  CV:   {cv_scores.mean():.4f} +/- {cv_scores.std() * 2:.4f}")
        return r2

    def train(self, df, students_df):
        print("preparing features...")
        X, y = self.prepare_features(df, students_df)
        
        # store column names before scaling,xgboost doesnt have feature_names_in_
        self.feature_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        # train random forest
        print("training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_scaled, y_train)
        rf_cv = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='r2')
        rf_r2 = self.print_metrics("Random Forest", y_test, rf.predict(X_test_scaled), rf_cv)

        # train xgboost,usually beats rf on tabular data
        print("\ntraining XGBoost...")
        xgb = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb.fit(X_train_scaled, y_train)
        xgb_cv = cross_val_score(xgb, X_train_scaled, y_train, cv=5, scoring='r2')
        xgb_r2 = self.print_metrics("XGBoost", y_test, xgb.predict(X_test_scaled), xgb_cv)

        # pick the better model based on r2
        if xgb_r2 >= rf_r2:
            print("\nXGBoost won, using it as final model")
            self.model = xgb
        else:
            print("\nRandom Forest won, using it as final model")
            self.model = rf

        # feature importance of winning model
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\ntop 10 features:")
        print(feature_importance.head(10))

        return self.model
    
    def save_model(self, path='ml_pipeline/models/saved_models/baseline_model.pkl'):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            # using feature_columns instead of feature_names_in_ since xgboost doesnt support it
            'feature_names': self.feature_columns
        }, path)
        print(f"\nmodel saved to {path}")

if __name__ == "__main__":
    sessions_df = pd.read_csv('backend/data/synthetic_classroom_data.csv')
    students_df = pd.read_csv('backend/data/student_profiles.csv')
    
    baseline_model = BaselineEngagementModel()
    baseline_model.train(sessions_df, students_df)
    baseline_model.save_model()