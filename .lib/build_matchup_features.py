from pathlib import Path
import pandas as pd

INPUT_PATH = Path("../scripts/data/processed/dataset_train_ready.csv")
MATCHUP_TABLE_PATH = Path("../scripts/model/matchup_table.csv")
OUTPUT_PATH = Path("../scripts/data/processed/dataset_improved.csv")

LANES = ["top", "jungle", "mid", "adc", "support"]


def build_lane_matchup_table(df: pd.DataFrame, lane: str) -> pd.DataFrame:
    blue_col = f"blue_{lane}_champion_name"
    red_col = f"red_{lane}_champion_name"

    lane_df = df[[blue_col, red_col, "blue_win"]].copy()
    lane_df = lane_df.dropna()

    grouped = (
        lane_df.groupby([blue_col, red_col])["blue_win"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={
            blue_col: "blue_champion",
            red_col: "red_champion",
            "mean": "blue_wr",
            "count": "games_count"
        })
    )

    grouped["lane"] = lane
    return grouped


def get_matchup_wr(
    matchup_map: dict,
    lane: str,
    blue_champ: str,
    red_champ: str,
    default: float = 0.5
) -> float:
    return matchup_map.get((lane, blue_champ, red_champ), default)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Soubor neexistuje: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_cols = ["blue_win"]
    for lane in LANES:
        required_cols.append(f"blue_{lane}_champion_name")
        required_cols.append(f"red_{lane}_champion_name")

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Chybí sloupce: {missing}")

    matchup_tables = []
    for lane in LANES:
        lane_table = build_lane_matchup_table(df, lane)
        matchup_tables.append(lane_table)

    full_matchup_table = pd.concat(matchup_tables, ignore_index=True)
    MATCHUP_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_matchup_table.to_csv(MATCHUP_TABLE_PATH, index=False, encoding="utf-8")

    matchup_map = {
        (row["lane"], row["blue_champion"], row["red_champion"]): float(row["blue_wr"])
        for _, row in full_matchup_table.iterrows()
    }

    for lane in LANES:
        blue_col = f"blue_{lane}_champion_name"
        red_col = f"red_{lane}_champion_name"
        out_col = f"{lane}_matchup_wr"

        df[out_col] = df.apply(
            lambda row: get_matchup_wr(
                matchup_map,
                lane,
                row[blue_col],
                row[red_col],
                0.5
            ),
            axis=1
        )

    df["avg_matchup_wr"] = df[[f"{lane}_matchup_wr" for lane in LANES]].mean(axis=1)


    if "blue_avg_rank" in df.columns and "red_avg_rank" in df.columns:
        df["rank_diff"] = df["blue_avg_rank"] - df["red_avg_rank"]

    if "blue_avg_recent_wr" in df.columns and "red_avg_recent_wr" in df.columns:
        df["recent_wr_diff"] = df["blue_avg_recent_wr"] - df["red_avg_recent_wr"]

    if "blue_avg_mastery" in df.columns and "red_avg_mastery" in df.columns:
        df["mastery_diff"] = df["blue_avg_mastery"] - df["red_avg_mastery"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("=== MATCHUP FEATURE HOTOVÉ ===")
    print(f"Matchup tabulka uložena do: {MATCHUP_TABLE_PATH}")
    print(f"Vylepšený dataset uložen do: {OUTPUT_PATH}")
    print(f"Počet řádků: {len(df)}")
    print(f"Počet matchup záznamů: {len(full_matchup_table)}")


if __name__ == "__main__":
    main()