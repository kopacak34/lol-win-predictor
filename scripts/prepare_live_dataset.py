from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/dataset_train_ready.csv")
OUTPUT_PATH = Path("../data/processed/dataset_live_ready.csv")

def main():
    df = pd.read_csv(INPUT_PATH)

    out = pd.DataFrame()

    out["blue_champ_1"] = df["blue_top_champion_name"]
    out["blue_champ_2"] = df["blue_jungle_champion_name"]
    out["blue_champ_3"] = df["blue_mid_champion_name"]
    out["blue_champ_4"] = df["blue_adc_champion_name"]
    out["blue_champ_5"] = df["blue_support_champion_name"]

    out["red_champ_1"] = df["red_top_champion_name"]
    out["red_champ_2"] = df["red_jungle_champion_name"]
    out["red_champ_3"] = df["red_mid_champion_name"]
    out["red_champ_4"] = df["red_adc_champion_name"]
    out["red_champ_5"] = df["red_support_champion_name"]

    out["blue_avg_rank"] = df["blue_avg_rank"]
    out["red_avg_rank"] = df["red_avg_rank"]
    out["blue_avg_mastery"] = df["blue_avg_mastery"]
    out["red_avg_mastery"] = df["red_avg_mastery"]
    out["blue_avg_recent_wr"] = df["blue_avg_recent_wr"]
    out["red_avg_recent_wr"] = df["red_avg_recent_wr"]
    out["blue_win"] = df["blue_win"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Uloženo do {OUTPUT_PATH}")

if __name__ == "__main__":
    main()