import math


BASE_CATEGORICAL = [
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

SORTED_CATEGORICAL = [
    "blue_champ_sorted_1",
    "blue_champ_sorted_2",
    "blue_champ_sorted_3",
    "blue_champ_sorted_4",
    "blue_champ_sorted_5",
    "red_champ_sorted_1",
    "red_champ_sorted_2",
    "red_champ_sorted_3",
    "red_champ_sorted_4",
    "red_champ_sorted_5",
]

CATEGORICAL = BASE_CATEGORICAL + SORTED_CATEGORICAL

BASE_NUMERIC = [
    "blue_avg_rank",
    "red_avg_rank",
    "blue_avg_mastery",
    "red_avg_mastery",
    "blue_avg_recent_wr",
    "red_avg_recent_wr",
]

DERIVED_NUMERIC = [
    "rank_diff",
    "mastery_diff",
    "wr_diff",
]

NUMERIC = BASE_NUMERIC + DERIVED_NUMERIC


def _to_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _sorted_team_champs(features, prefix):
    champs = [str(features.get(f"{prefix}_champ_{i}", "")).strip() for i in range(1, 6)]
    champs = sorted(champ for champ in champs if champ)
    return champs + ["Unknown"] * (5 - len(champs))


def engineer_feature_dict(features):
    engineered = dict(features)

    for prefix in ("blue", "red"):
        sorted_champs = _sorted_team_champs(engineered, prefix)
        for i, champ in enumerate(sorted_champs[:5], start=1):
            engineered[f"{prefix}_champ_sorted_{i}"] = champ

    blue_mastery = math.log1p(max(_to_float(engineered.get("blue_avg_mastery")), 0.0))
    red_mastery = math.log1p(max(_to_float(engineered.get("red_avg_mastery")), 0.0))

    engineered["blue_avg_mastery"] = blue_mastery
    engineered["red_avg_mastery"] = red_mastery
    engineered["rank_diff"] = _to_float(engineered.get("blue_avg_rank")) - _to_float(engineered.get("red_avg_rank"))
    engineered["mastery_diff"] = blue_mastery - red_mastery
    engineered["wr_diff"] = _to_float(engineered.get("blue_avg_recent_wr"), 0.5) - _to_float(
        engineered.get("red_avg_recent_wr"), 0.5
    )

    return engineered


def engineer_dataframe(df):
    engineered = df.copy()

    for prefix in ("blue", "red"):
        champ_cols = [f"{prefix}_champ_{i}" for i in range(1, 6)]
        sorted_values = (
            engineered[champ_cols]
            .fillna("Unknown")
            .astype(str)
            .apply(lambda row: sorted(value.strip() or "Unknown" for value in row), axis=1)
        )

        for i in range(5):
            engineered[f"{prefix}_champ_sorted_{i + 1}"] = sorted_values.apply(lambda values, idx=i: values[idx])

    for col in BASE_NUMERIC:
        engineered[col] = engineered[col].astype(float)

    engineered["blue_avg_mastery"] = engineered["blue_avg_mastery"].clip(lower=0).apply(math.log1p)
    engineered["red_avg_mastery"] = engineered["red_avg_mastery"].clip(lower=0).apply(math.log1p)
    engineered["rank_diff"] = engineered["blue_avg_rank"] - engineered["red_avg_rank"]
    engineered["mastery_diff"] = engineered["blue_avg_mastery"] - engineered["red_avg_mastery"]
    engineered["wr_diff"] = engineered["blue_avg_recent_wr"] - engineered["red_avg_recent_wr"]

    return engineered
