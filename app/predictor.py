import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "scripts" / "model" / "live_model.pkl"

class Predictor:
    def __init__(self):
        print(f"[MODEL PATH] {MODEL_PATH}")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model nenalezen: {MODEL_PATH}")
        self.model = joblib.load(MODEL_PATH)
        print(f"[MODEL CLASSES] {self.model.classes_}")

    def predict(self, features: dict):
        df = pd.DataFrame([features])
        proba = self.model.predict_proba(df)[0]
        class_to_proba = dict(zip(self.model.classes_, proba))

        return {
            "blue_win_prob": float(class_to_proba.get(1, 0.0)),
            "red_win_prob": float(class_to_proba.get(0, 0.0)),
        }