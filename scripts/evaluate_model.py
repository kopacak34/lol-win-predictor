from pathlib import Path
import json

METRICS_PATH = Path("model/metrics.json")


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Soubor neexistuje: {METRICS_PATH}")

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print("=== EVALUACE MODELU ===")
    print(f"Nejlepší model: {metrics['best_model']}")
    print(f"Velikost datasetu: {metrics['dataset_size']}")
    print(f"Train size: {metrics['train_size']}")
    print(f"Test size: {metrics['test_size']}")

    print("\n--- Logistic Regression ---")
    print(f"Accuracy: {metrics['logistic_regression']['accuracy']:.4f}")
    print(f"F1:       {metrics['logistic_regression']['f1']:.4f}")

    print("\n--- Random Forest ---")
    print(f"Accuracy: {metrics['random_forest']['accuracy']:.4f}")
    print(f"F1:       {metrics['random_forest']['f1']:.4f}")


if __name__ == "__main__":
    main()