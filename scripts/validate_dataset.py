from pathlib import Path
import json
import pandas as pd

DATASET_PATH = Path("data/processed/dataset.csv")
REPORT_PATH = Path("data/processed/dataset_validation_report.json")


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Soubor neexistuje: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    report = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "duplicate_match_id_count": 0,
        "missing_values_per_column": {},
        "target_distribution": {},
    }

    if "match_id" in df.columns:
        report["duplicate_match_id_count"] = int(df.duplicated(subset=["match_id"]).sum())

    report["missing_values_per_column"] = {
        col: int(val) for col, val in df.isna().sum().to_dict().items()
    }

    if "blue_win" in df.columns:
        target_dist = df["blue_win"].value_counts(dropna=False).to_dict()
        report["target_distribution"] = {
            str(k): int(v) for k, v in target_dist.items()
        }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== VALIDACE DATASETU ===")
    print(f"Řádků: {len(df)}")
    print(f"Sloupců: {len(df.columns)}")

    if "match_id" in df.columns:
        print(f"Duplicity match_id: {report['duplicate_match_id_count']}")

    if "blue_win" in df.columns:
        print("Rozložení targetu blue_win:")
        print(df["blue_win"].value_counts(dropna=False))

    print(f"\nReport uložen do: {REPORT_PATH}")


if __name__ == "__main__":
    main()