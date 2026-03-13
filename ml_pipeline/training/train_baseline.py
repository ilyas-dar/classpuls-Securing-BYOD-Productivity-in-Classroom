import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class BaselineEngagementModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = []
        
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
    
    def train(self, df, students_df):
        print("Preparing features...")
        X, y = self.prepare_features(df, students_df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # random forest,100 trees depth 10 works well enough here
        print("Training Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # cv to make sure its not just overfitting to the test split
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return self.model
    
    def save_model(self, path='ml_pipeline/models/saved_models/baseline_model.pkl'):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            # saving feature names so inference knows what columns to expect
            'feature_names': self.model.feature_names_in_.tolist()
        }, path)
        print(f"Model saved to {path}")

if __name__ == "__main__":
    sessions_df = pd.read_csv('backend/data/synthetic_classroom_data.csv')
    students_df = pd.read_csv('backend/data/student_profiles.csv')
    
    baseline_model = BaselineEngagementModel()
    baseline_model.train(sessions_df, students_df)
    baseline_model.save_model()