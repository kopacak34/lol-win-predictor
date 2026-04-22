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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def resolve_base_dir() -> Path:
    script_path = Path(__file__).resolve()

    candidate_inner = script_path.parent.parent

    candidate_outer = script_path.parent.parent.parent

    inner_dataset = candidate_inner / "data" / "processed" / "dataset_live_ready.csv"
    outer_dataset = candidate_outer / "data" / "processed" / "dataset_live_ready.csv"

    print(f"[CHECK INNER] {inner_dataset}")
    print(f"[CHECK OUTER] {outer_dataset}")

    if inner_dataset.exists():
        return candidate_inner

    if outer_dataset.exists():
        return candidate_outer

    raise FileNotFoundError(
        "Nepodařilo se najít dataset_live_ready.csv ani ve vnitřním, ani ve vnějším rootu projektu."
    )


BASE_DIR = resolve_base_dir()

DATA_PATH = BASE_DIR / "data" / "processed" / "dataset_live_ready.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "live_model.pkl"
METRICS_PATH = MODEL_DIR / "live_model_metrics.json"
FEATURE_CONFIG_PATH = MODEL_DIR / "live_model_feature_config.json"

TARGET = "blue_win"

CATEGORICAL = [
    "blue_champ_1",
    "blue_champ_2",
    "blue_champ_3",
    "blue_champ_4",
    "blue_champ_5",
    "red_champ_1",
    "red_champ_2",
    "red_champ_3",
    "red_champ_4",
    "red_champ_5",
]

NUMERIC = [
    "blue_avg_rank",
    "red_avg_rank",
    "blue_avg_mastery",
    "red_avg_mastery",
    "blue_avg_recent_wr",
    "red_avg_recent_wr",
]


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }


def main():
    print(f"[BASE_DIR] {BASE_DIR}")
    print(f"[DATA_PATH] {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Soubor neexistuje: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = CATEGORICAL + NUMERIC + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"V datasetu chybí sloupce: {missing}")

    X = df[CATEGORICAL + NUMERIC].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("cat", cat_pipe, CATEGORICAL),
        ("num", num_pipe, NUMERIC),
    ])

    models = {
        "logreg": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),
        "random_forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            ))
        ]),
    }

    best_name = None
    best_model = None
    best_f1 = -1.0
    results = {}

    for name, model in models.items():
        print(f"Trénuji: {name}")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics

        print(
            f"{name}: accuracy={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = name
            best_model = model

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    metrics_output = {
        "best_model": best_name,
        "dataset_size": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "results": results,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, ensure_ascii=False, indent=2)

    feature_config = {
        "target_column": TARGET,
        "categorical_features": CATEGORICAL,
        "numeric_features": NUMERIC,
        "all_features": CATEGORICAL + NUMERIC,
    }

    with open(FEATURE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_config, f, ensure_ascii=False, indent=2)

    print("\n=== HOTOVO ===")
    print(f"Nejlepší model: {best_name}")
    print(f"Model uložen do: {MODEL_PATH}")
    print(f"Metriky uloženy do: {METRICS_PATH}")
    print(f"Feature config uložen do: {FEATURE_CONFIG_PATH}")


if __name__ == "__main__":
    main()