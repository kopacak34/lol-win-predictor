import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RIOT_API_KEY")
PLATFORM_REGION = os.getenv("PLATFORM_ROUTING", "eun1").strip().lower()
REGIONAL_ROUTING = os.getenv("REGIONAL_ROUTING", "europe").strip().lower()

HEADERS = {
    "X-Riot-Token": API_KEY
}

_recent_wr_cache = {}
_match_cache = {}


def safe_get(url: str, params=None, retries: int = 5):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=20)

            if r.status_code == 200:
                return r

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait_time = int(retry_after) if retry_after and retry_after.isdigit() else 5
                print(f"[429] Rate limit hit, čekám {wait_time}s...")
                time.sleep(wait_time)
                continue

            if r.status_code in (500, 502, 503, 504):
                wait_time = 2 ** attempt
                print(f"[{r.status_code}] Riot server error, retry za {wait_time}s...")
                time.sleep(wait_time)
                continue

            if r.status_code == 404:
                return None

            print(f"[WARN] {url} -> status {r.status_code}")
            return None

        except requests.RequestException as exc:
            wait_time = 2 ** attempt
            print(f"[EXC] {exc} -> retry za {wait_time}s...")
            time.sleep(wait_time)

    return None


def get_account_by_riot_id(game_name: str, tag_line: str):
    url = (
        f"https://{REGIONAL_ROUTING}.api.riotgames.com/"
        f"riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    )
    r = safe_get(url)

    print(f"[ACCOUNT URL] {url}")
    print(f"[ACCOUNT STATUS] {r.status_code if r else 'NONE'}")

    if not r:
        return None

    data = r.json()
    print(f"[ACCOUNT JSON] {data}")
    return data


def get_active_game_by_puuid(puuid: str):
    url = (
        f"https://{PLATFORM_REGION}.api.riotgames.com/"
        f"lol/spectator/v5/active-games/by-summoner/{puuid}"
    )
    r = safe_get(url)

    print(f"[SPECTATOR URL] {url}")
    print(f"[SPECTATOR STATUS] {r.status_code if r else 'NONE'}")

    if not r:
        print("[SPECTATOR] Hráč není podle spectator endpointu v aktivní hře.")
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


def get_player_rank(puuid: str):
    url = f"https://{PLATFORM_REGION}.api.riotgames.com/lol/league/v4/entries/by-puuid/{puuid}"
    r = safe_get(url)

    if not r:
        return None

    data = r.json()

    for entry in data:
        if entry["queueType"] == "RANKED_SOLO_5x5":
            return entry

    return None


def rank_to_number(rank_data):
    if not rank_data:
        return 0.0

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

    division_map = {"IV": 1, "III": 2, "II": 3, "I": 4}

    tier_value = tier_map.get(rank_data["tier"], 0)
    division_value = division_map.get(rank_data["rank"], 0)
    lp = int(rank_data.get("leaguePoints", 0))

    if tier_value == 0 or division_value == 0:
        return 0.0

    return tier_value + (division_value / 10.0) + min(lp, 100) / 1000.0


def get_champion_mastery(puuid: str, champion_id: int):
    url = (
        f"https://{PLATFORM_REGION}.api.riotgames.com/"
        f"lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/by-champion/{champion_id}"
    )
    r = safe_get(url)

    if not r:
        return 0

    return r.json().get("championPoints", 0)


def get_match_ids_by_puuid(puuid: str, count: int = 12):
    url = f"https://{REGIONAL_ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {
        "start": 0,
        "count": count,
        "queue": 420,  # ranked solo/duo
    }
    r = safe_get(url, params=params)
    if not r:
        return []
    return r.json()


def get_match(match_id: str):
    if match_id in _match_cache:
        return _match_cache[match_id]

    url = f"https://{REGIONAL_ROUTING}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    r = safe_get(url)
    if not r:
        return None

    data = r.json()
    _match_cache[match_id] = data
    return data


def get_recent_winrate(puuid: str, sample_size: int = 10) -> float:
    cache_key = (puuid, sample_size)
    if cache_key in _recent_wr_cache:
        return _recent_wr_cache[cache_key]

    match_ids = get_match_ids_by_puuid(puuid, count=sample_size + 5)

    wins = 0
    total = 0

    for match_id in match_ids:
        match_data = get_match(match_id)
        if not match_data:
            continue

        participants = match_data.get("info", {}).get("participants", [])
        player = next((p for p in participants if p.get("puuid") == puuid), None)
        if not player:
            continue

        wins += 1 if player.get("win") else 0
        total += 1

        if total >= sample_size:
            break

    result = round(wins / total, 4) if total > 0 else 0.5
    _recent_wr_cache[cache_key] = result
    return result