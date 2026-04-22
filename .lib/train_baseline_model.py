from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

DATA_PATH = Path("../scripts/data/processed/dataset_train_ready.csv")
MODEL_DIR = Path("../scripts/model")
MODEL_PATH = MODEL_DIR / "baseline_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURE_CONFIG_PATH = MODEL_DIR / "feature_config.json"

TARGET_COLUMN = "blue_win"

CATEGORICAL_FEATURES = [
    "blue_top_champion_name",
    "blue_jungle_champion_name",
    "blue_mid_champion_name",
    "blue_adc_champion_name",
    "blue_support_champion_name",
    "red_top_champion_name",
    "red_jungle_champion_name",
    "red_mid_champion_name",
    "red_adc_champion_name",
    "red_support_champion_name",
]

NUMERIC_FEATURES = [
    "blue_avg_rank",
    "red_avg_rank",
    "blue_avg_recent_wr",
    "red_avg_recent_wr",
    "blue_avg_mastery",
    "red_avg_mastery",
]


def build_preprocessor() -> ColumnTransformer:
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer(transformers=[
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ("num", numeric_transformer, NUMERIC_FEATURES),
    ])


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    result = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    return result


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Soubor neexistuje: {DATA_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    required = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Chybí sloupce: {missing}")

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor()

    logistic_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    rf_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("Trénuji Logistic Regression...")
    logistic_model.fit(X_train, y_train)
    logistic_metrics = evaluate_model(logistic_model, X_test, y_test)

    print("Trénuji Random Forest...")
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)

    print("\n=== VÝSLEDKY ===")
    print(f"Logistic Regression accuracy: {logistic_metrics['accuracy']:.4f}, F1: {logistic_metrics['f1']:.4f}")
    print(f"Random Forest accuracy:      {rf_metrics['accuracy']:.4f}, F1: {rf_metrics['f1']:.4f}")

    best_model_name = "logistic_regression"
    best_model = logistic_model
    best_metrics = logistic_metrics

    if rf_metrics["f1"] > logistic_metrics["f1"]:
        best_model_name = "random_forest"
        best_model = rf_model
        best_metrics = rf_metrics

    joblib.dump(best_model, MODEL_PATH)

    all_metrics = {
        "best_model": best_model_name,
        "logistic_regression": logistic_metrics,
        "random_forest": rf_metrics,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "dataset_size": int(len(df)),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    feature_config = {
        "target_column": TARGET_COLUMN,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "all_features": CATEGORICAL_FEATURES + NUMERIC_FEATURES,
    }

    with open(FEATURE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_config, f, ensure_ascii=False, indent=2)

    print(f"\nNejlepší model: {best_model_name}")
    print(f"Model uložen do: {MODEL_PATH}")
    print(f"Metriky uloženy do: {METRICS_PATH}")
    print(f"Config uložen do: {FEATURE_CONFIG_PATH}")


if __name__ == "__main__":
    main()