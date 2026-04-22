import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RIOT_API_KEY")
PLATFORM_REGION = os.getenv("PLATFORM_ROUTING", "eun1").strip().lower()
REGIONAL_ROUTING = os.getenv("REGIONAL_ROUTING", "europe").strip().lower()

HEADERS = {
    "X-Riot-Token": API_KEY
}


def get_account_by_riot_id(game_name: str, tag_line: str):
    url = (
        f"https://{REGIONAL_ROUTING}.api.riotgames.com/"
        f"riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    )
    r = requests.get(url, headers=HEADERS, timeout=20)

    print(f"[ACCOUNT URL] {url}")
    print(f"[ACCOUNT STATUS] {r.status_code}")

    if r.status_code != 200:
        print(f"[ACCOUNT BODY] {r.text}")
        return None

    data = r.json()
    print(f"[ACCOUNT JSON] {data}")
    return data

def get_player_rank(puuid: str):
    url = f"https://{PLATFORM_REGION}.api.riotgames.com/lol/league/v4/entries/by-puuid/{puuid}"
    r = requests.get(url, headers=HEADERS)

    if r.status_code != 200:
        return None

    data = r.json()


    for entry in data:
        if entry["queueType"] == "RANKED_SOLO_5x5":
            return entry

    return None


def rank_to_number(rank_data):
    if not rank_data:
        return 41

    tier_map = {
        "IRON": 1,
        "BRONZE": 2,
        "SILVER": 3,
        "GOLD": 4,
        "PLATINUM": 5,
        "EMERALD": 6,
        "DIAMOND": 7,
        "MASTER": 8,
        "GRANDMASTER": 9,
        "CHALLENGER": 10
    }

    tier = rank_data["tier"]
    rank = rank_data["rank"]

    division_map = {"IV": 1, "III": 2, "II": 3, "I": 4}

    return tier_map.get(tier, 0) * 10 + division_map.get(rank, 0)


def get_champion_mastery(puuid: str, champion_id: int):
    url = f"https://{PLATFORM_REGION}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/by-champion/{champion_id}"
    r = requests.get(url, headers=HEADERS)

    if r.status_code != 200:
        return 0

    return r.json().get("championPoints", 0)

def get_active_game_by_puuid(puuid: str):
    url = (
        f"https://{PLATFORM_REGION}.api.riotgames.com/"
        f"lol/spectator/v5/active-games/by-summoner/{puuid}"
    )
    r = requests.get(url, headers=HEADERS, timeout=20)

    print(f"[SPECTATOR URL] {url}")
    print(f"[SPECTATOR STATUS] {r.status_code}")

    if r.status_code == 404:
        print("[SPECTATOR] Hráč není podle spectator endpointu v aktivní hře.")
        return None

    if r.status_code != 200:
        print(f"[SPECTATOR BODY] {r.text}")
        return None

    data = r.json()
    print("[SPECTATOR] Active game nalezena.")
    return data


def get_active_game_from_riot_id(game_name: str, tag_line: str):
    account = get_account_by_riot_id(game_name, tag_line)
    if not account:
        return None

    puuid = account.get("puuid")
    if not puuid:
        print("[ERROR] account endpoint nevrátil puuid")
        return None

    return get_active_game_by_puuid(puuid)