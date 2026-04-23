from pathlib import Path
import json
import sys

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.feature_engineering import CATEGORICAL, NUMERIC, engineer_dataframe


def resolve_base_dir() -> Path:
    script_path = Path(__file__).resolve()
    candidate_inner = script_path.parent.parent
    candidate_outer = script_path.parent.parent.parent

    for candidate in (candidate_inner, candidate_outer):
        dataset = candidate / "data" / "processed" / "dataset_live_ready.csv"
        if dataset.exists():
            return candidate

    raise FileNotFoundError(
        "Nepodarilo se najit data/processed/dataset_live_ready.csv v rootu projektu."
    )


BASE_DIR = resolve_base_dir()

DATA_PATH = BASE_DIR / "data" / "processed" / "dataset_live_ready.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "live_model.pkl"
LEGACY_MODEL_DIR = BASE_DIR / "scripts" / "model"
LEGACY_MODEL_PATH = LEGACY_MODEL_DIR / "live_model.pkl"
METRICS_PATH = MODEL_DIR / "live_model_metrics.json"
FEATURE_CONFIG_PATH = MODEL_DIR / "live_model_feature_config.json"

TARGET = "blue_win"
RANDOM_STATE = 42


def make_preprocessor():
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer([
        ("cat", cat_pipe, CATEGORICAL),
        ("num", num_pipe, NUMERIC),
    ])


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }


def candidate_models():
    return {
        "logreg": (
            Pipeline([
                ("preprocessor", make_preprocessor()),
                ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
            ]),
            {
                "classifier__C": [0.1, 0.3, 1.0, 3.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            Pipeline([
                ("preprocessor", make_preprocessor()),
                ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
            ]),
            {
                "classifier__n_estimators": [300, 700],
                "classifier__max_depth": [None, 8, 14],
                "classifier__min_samples_leaf": [1, 3, 6],
                "classifier__class_weight": [None, "balanced_subsample"],
            },
        ),
        "extra_trees": (
            Pipeline([
                ("preprocessor", make_preprocessor()),
                ("classifier", ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1)),
            ]),
            {
                "classifier__n_estimators": [500, 900],
                "classifier__max_depth": [None, 8, 14],
                "classifier__min_samples_leaf": [1, 3, 6],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
    }


def main():
    print(f"[BASE_DIR] {BASE_DIR}")
    print(f"[DATA_PATH] {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Soubor neexistuje: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = engineer_dataframe(df)

    required = CATEGORICAL + NUMERIC + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"V datasetu chybi sloupce: {missing}")

    X = df[CATEGORICAL + NUMERIC].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    best_name = None
    best_model = None
    best_accuracy = -1.0

    for name, (pipeline, param_grid) in candidate_models().items():
        print(f"Trenuji a ladim: {name}")
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        metrics = evaluate(search.best_estimator_, X_test, y_test)
        metrics["best_cv_accuracy"] = float(search.best_score_)
        metrics["best_params"] = search.best_params_
        results[name] = metrics

        print(
            f"{name}: test_accuracy={metrics['accuracy']:.4f}, "
            f"test_f1={metrics['f1']:.4f}, "
            f"cv_accuracy={metrics['best_cv_accuracy']:.4f}"
        )

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_name = name
            best_model = search.best_estimator_

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LEGACY_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(best_model, LEGACY_MODEL_PATH)

    metrics_output = {
        "best_model": best_name,
        "selection_metric": "accuracy",
        "dataset_size": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().to_dict().items()},
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
    print(f"Nejlepsi model: {best_name}")
    print(f"Model ulozen do: {MODEL_PATH}")
    print(f"Legacy kopie modelu ulozena do: {LEGACY_MODEL_PATH}")
    print(f"Metriky ulozeny do: {METRICS_PATH}")
    print(f"Feature config ulozen do: {FEATURE_CONFIG_PATH}")


if __name__ == "__main__":
    main()
