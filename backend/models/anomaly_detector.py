import joblib
import numpy as np
import os

class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.threshold = None
        self.feature_names = []

    def prepare_features(self, activity_dict):
        # pull out the fields we need,same ones used during training
        app_switches       = activity_dict.get('app_switches', 0)
        keystroke          = activity_dict.get('keystroke_intensity', 50)
        inactivity         = activity_dict.get('inactivity_periods', 0)
        poll               = activity_dict.get('poll_participation', 0)
        collaboration      = activity_dict.get('collaboration_actions', 0)

        # normalize same way as training script
        app_rate           = app_switches / 10
        keystroke_norm     = keystroke / 100
        inactivity_rate    = inactivity / 10
        collab_rate        = collaboration / 20

        # ratio features,same ones from train_anomaly.py
        switch_inactivity  = app_rate / (inactivity_rate + 0.01)
        keystroke_collab   = keystroke_norm / (collab_rate + 0.01)

        return np.array([[
            app_rate, keystroke_norm, inactivity_rate,
            poll, collab_rate,
            switch_inactivity, keystroke_collab
        ]])
        
        
# rest code will be done tommorow!
