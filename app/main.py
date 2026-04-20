import traceback
import os

from spectator_client import get_active_game_from_riot_id
from feature_builder import build_features
from predictor import Predictor


def main():
    print("=== LoL Live Win Predictor ===")

    riot_id = input("Zadej Riot ID (např. name#tag): ").strip()

    if "#" not in riot_id:
        print("Špatný formát Riot ID")
        return

    game_name, tag_line = riot_id.split("#", 1)

    game = get_active_game_from_riot_id(game_name, tag_line)
    if not game:
        print("Hráč není v aktivní hře nebo se nepodařilo načíst spectator data.")
        return

    print("Hra nalezena, počítám predikci...")

    features = build_features(game)

    predictor = Predictor()
    result = predictor.predict(features)

    print("\n=== VÝSLEDEK ===")
    print(f"Blue win chance: {result['blue_win_prob'] * 100:.2f}%")
    print(f"Red win chance:  {result['red_win_prob'] * 100:.2f}%")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n=== CHYBA ===")
        traceback.print_exc()
    finally:
        os.system("pause")