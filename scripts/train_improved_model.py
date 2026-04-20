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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

DATA_PATH = Path("data/processed/dataset_train_ready.csv")
MODEL_DIR = Path("model")

MODEL_PATH = MODEL_DIR / "improved_no_leakage_model.pkl"
METRICS_PATH = MODEL_DIR / "improved_no_leakage_metrics.json"
FEATURE_CONFIG_PATH = MODEL_DIR / "improved_no_leakage_feature_config.json"
MATCHUP_TABLE_PATH = MODEL_DIR / "improved_no_leakage_matchup_table.csv"

TARGET_COLUMN = "blue_win"
RANDOM_STATE = 42
TEST_SIZE = 0.2

LANES = ["top", "jungle", "mid", "adc", "support"]

BASE_CATEGORICAL_FEATURES = [
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

BASE_NUMERIC_FEATURES = [
    "blue_avg_rank",
    "red_avg_rank",
    "blue_avg_recent_wr",
    "red_avg_recent_wr",
    "blue_avg_mastery",
    "red_avg_mastery",
]

ENGINEERED_NUMERIC_FEATURES = [
    "top_matchup_wr",
    "jungle_matchup_wr",
    "mid_matchup_wr",
    "adc_matchup_wr",
    "support_matchup_wr",
    "avg_matchup_wr",
    "rank_diff",
    "recent_wr_diff",
    "mastery_diff",
]


def build_lane_matchup_table(train_df: pd.DataFrame, lane: str) -> pd.DataFrame:
    blue_col = f"blue_{lane}_champion_name"
    red_col = f"red_{lane}_champion_name"

    lane_df = train_df[[blue_col, red_col, TARGET_COLUMN]].copy().dropna()

    grouped = (
        lane_df.groupby([blue_col, red_col])[TARGET_COLUMN]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={
            blue_col: "blue_champion",
            red_col: "red_champion",
            "mean": "blue_wr",
            "count": "games_count",
        })
    )

    grouped["lane"] = lane
    return grouped


def build_matchup_map(train_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    tables = []
    for lane in LANES:
        lane_table = build_lane_matchup_table(train_df, lane)
        tables.append(lane_table)

    full_table = pd.concat(tables, ignore_index=True)

    matchup_map = {
        (row["lane"], row["blue_champion"], row["red_champion"]): float(row["blue_wr"])
        for _, row in full_table.iterrows()
    }

    return matchup_map, full_table


def get_matchup_wr(matchup_map: dict, lane: str, blue_champ: str, red_champ: str, default: float = 0.5) -> float:
    return matchup_map.get((lane, blue_champ, red_champ), default)


def add_engineered_features(df: pd.DataFrame, matchup_map: dict) -> pd.DataFrame:
    out = df.copy()

    for lane in LANES:
        blue_col = f"blue_{lane}_champion_name"
        red_col = f"red_{lane}_champion_name"
        out_col = f"{lane}_matchup_wr"

        out[out_col] = out.apply(
            lambda row: get_matchup_wr(
                matchup_map=matchup_map,
                lane=lane,
                blue_champ=row[blue_col],
                red_champ=row[red_col],
                default=0.5,
            ),
            axis=1
        )

    out["avg_matchup_wr"] = out[[f"{lane}_matchup_wr" for lane in LANES]].mean(axis=1)

    out["rank_diff"] = out["blue_avg_rank"] - out["red_avg_rank"]
    out["recent_wr_diff"] = out["blue_avg_recent_wr"] - out["red_avg_recent_wr"]
    out["mastery_diff"] = out["blue_avg_mastery"] - out["red_avg_mastery"]

    return out


def build_preprocessor(categorical_features: list[str], numeric_features: list[str]) -> ColumnTransformer:
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer(transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", numeric_transformer, numeric_features),
    ])


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Soubor neexistuje: {DATA_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    required = BASE_CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES + [TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Chybí sloupce: {missing}")

    df = df[required].copy()
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    X = df[BASE_CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    X_train_base, X_test_base, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df_for_matchups = X_train_base.copy()
    train_df_for_matchups[TARGET_COLUMN] = y_train.values

    matchup_map, matchup_table = build_matchup_map(train_df_for_matchups)
    matchup_table.to_csv(MATCHUP_TABLE_PATH, index=False, encoding="utf-8")

    X_train = add_engineered_features(X_train_base, matchup_map)
    X_test = add_engineered_features(X_test_base, matchup_map)

    categorical_features = BASE_CATEGORICAL_FEATURES
    numeric_features = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES

    preprocessor = build_preprocessor(categorical_features, numeric_features)

    models = {
        "logistic_regression": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),
        "random_forest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        "gradient_boosting": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(
                random_state=RANDOM_STATE
            ))
        ]),
    }

    results = {}
    best_name = None
    best_model = None
    best_f1 = -1.0

    for model_name, model in models.items():
        print(f"Trénuji: {model_name}")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[model_name] = metrics

        print(
            f"{model_name}: accuracy={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = model_name
            best_model = model

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
        "target_column": TARGET_COLUMN,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "engineered_numeric_features": ENGINEERED_NUMERIC_FEATURES,
        "all_features": categorical_features + numeric_features,
    }

    with open(FEATURE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_config, f, ensure_ascii=False, indent=2)

    print("\n=== HOTOVO ===")
    print(f"Nejlepší model: {best_name}")
    print(f"Model uložen do: {MODEL_PATH}")
    print(f"Metriky uloženy do: {METRICS_PATH}")
    print(f"Feature config uložen do: {FEATURE_CONFIG_PATH}")
    print(f"Matchup tabulka uložena do: {MATCHUP_TABLE_PATH}")


if __name__ == "__main__":
    main()