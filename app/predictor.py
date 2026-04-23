import joblib
import pandas as pd
from pathlib import Path

from feature_engineering import engineer_feature_dict

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "live_model.pkl"
LEGACY_MODEL_PATH = BASE_DIR / "scripts" / "model" / "live_model.pkl"

class Predictor:
    def __init__(self):
        model_path = MODEL_PATH if MODEL_PATH.exists() else LEGACY_MODEL_PATH
        print(f"[MODEL PATH] {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model nenalezen: {model_path}")
        self.model = joblib.load(model_path)
        print(f"[MODEL CLASSES] {self.model.classes_}")

    def predict(self, features: dict):
        df = pd.DataFrame([engineer_feature_dict(features)])
        proba = self.model.predict_proba(df)[0]
        class_to_proba = dict(zip(self.model.classes_, proba))

        return {
            "blue_win_prob": float(class_to_proba.get(1, 0.0)),
            "red_win_prob": float(class_to_proba.get(0, 0.0)),
        }
