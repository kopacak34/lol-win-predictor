import json
import time

from app.path_utils import get_base_path
from app.spectator_client import get_player_rank, rank_to_number, get_champion_mastery,get_recent_winrate

LOCAL_CHAMPION_JSON = get_base_path() / "assets" / "champion.json"

_champion_cache = None


def load_champion_map():
    global _champion_cache

    if _champion_cache is not None:
        return _champion_cache

    if not LOCAL_CHAMPION_JSON.exists():
        raise FileNotFoundError(f"Soubor champion.json nenalezen: {LOCAL_CHAMPION_JSON}")

    with open(LOCAL_CHAMPION_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    mapping = {}
    for _, champ in data.items():
        mapping[int(champ["key"])] = champ["id"]

    _champion_cache = mapping
    return mapping


def champ_name(champion_id: int) -> str:
    mapping = load_champion_map()
    return mapping.get(int(champion_id), f"UnknownChampion_{champion_id}")


def safe_sleep():
    time.sleep(0.2)


def build_features(game_data: dict) -> dict:
    participants = game_data["participants"]

    blue = [p for p in participants if p["teamId"] == 100]
    red = [p for p in participants if p["teamId"] == 200]

    if len(blue) != 5 or len(red) != 5:
        raise ValueError("Ve hře není přesně 5 hráčů na každé straně.")

    features = {}

    blue_ranks = []
    red_ranks = []
    blue_masteries = []
    red_masteries = []
    blue_recent_wrs = []
    red_recent_wrs = []

    for i, p in enumerate(blue, start=1):
        champion_id = int(p["championId"])
        puuid = p["puuid"]

        features[f"blue_champ_{i}"] = champ_name(champion_id)

        safe_sleep()
        rank_value = rank_to_number(get_player_rank(puuid))
        blue_ranks.append(rank_value)

        safe_sleep()
        recent_wr_value = float(get_recent_winrate(puuid, sample_size=10))
        blue_recent_wrs.append(recent_wr_value)

        safe_sleep()
        mastery_value = float(get_champion_mastery(puuid, champion_id))
        blue_masteries.append(mastery_value)


    for i, p in enumerate(red, start=1):
        champion_id = int(p["championId"])
        puuid = p["puuid"]

        features[f"red_champ_{i}"] = champ_name(champion_id)

        safe_sleep()
        rank_value = rank_to_number(get_player_rank(puuid))
        red_ranks.append(rank_value)

        safe_sleep()
        recent_wr_value = float(get_recent_winrate(puuid, sample_size=10))
        red_recent_wrs.append(recent_wr_value)

        safe_sleep()
        mastery_value = float(get_champion_mastery(puuid, champion_id))
        red_masteries.append(mastery_value)

    features["blue_avg_rank"] = sum(blue_ranks) / len(blue_ranks) if blue_ranks else 0.0
    features["red_avg_rank"] = sum(red_ranks) / len(red_ranks) if red_ranks else 0.0

    features["blue_avg_mastery"] = sum(blue_masteries) / len(blue_masteries) if blue_masteries else 0.0
    features["red_avg_mastery"] = sum(red_masteries) / len(red_masteries) if red_masteries else 0.0

    features["blue_avg_recent_wr"] = sum(blue_recent_wrs) / len(blue_recent_wrs) if blue_recent_wrs else 0.5
    features["red_avg_recent_wr"] = sum(red_recent_wrs) / len(red_recent_wrs) if red_recent_wrs else 0.5

    return features