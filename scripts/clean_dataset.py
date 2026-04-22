from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/dataset.csv")
CLEAN_PATH = Path("data/processed/dataset_clean.csv")
TRAIN_READY_PATH = Path("data/processed/dataset_train_ready.csv")

REQUIRED_COLUMNS = [
    "match_id",
    "blue_win",

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

    "blue_avg_rank",
    "red_avg_rank",
    "blue_avg_recent_wr",
    "red_avg_recent_wr",
    "blue_avg_mastery",
    "red_avg_mastery",
]


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Soubor neexistuje: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_required:
        raise ValueError(f"V datasetu chybí povinné sloupce: {missing_required}")


    df = df[REQUIRED_COLUMNS].copy()


    df = df.drop_duplicates(subset=["match_id"]).reset_index(drop=True)


    df = df.dropna(subset=["blue_win"]).copy()


    df["blue_win"] = df["blue_win"].astype(int)


    champion_columns = [
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

    for col in champion_columns:
        df[col] = df[col].astype(str).str.strip()


    numeric_columns = [
        "blue_avg_rank",
        "red_avg_rank",
        "blue_avg_recent_wr",
        "red_avg_recent_wr",
        "blue_avg_mastery",
        "red_avg_mastery",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False, encoding="utf-8")


    df.to_csv(TRAIN_READY_PATH, index=False, encoding="utf-8")

    print("=== ČIŠTĚNÍ HOTOVO ===")
    print(f"Počet řádků: {len(df)}")
    print(f"Uloženo: {CLEAN_PATH}")
    print(f"Uloženo: {TRAIN_READY_PATH}")


if __name__ == "__main__":
    main()