import os
import time
import json
import random
from pathlib import Path
from collections import deque
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

RIOT_API_KEY = os.getenv("RIOT_API_KEY", "").strip()
PLATFORM_ROUTING = os.getenv("PLATFORM_ROUTING", "eun1").strip().lower()
REGIONAL_ROUTING = os.getenv("REGIONAL_ROUTING", "europe").strip().lower()

if not RIOT_API_KEY:
    raise ValueError("Chybí RIOT_API_KEY v .env souboru.")

HEADERS = {
    "X-Riot-Token": RIOT_API_KEY
}

# ==========================================
# NASTAVENÍ
# ==========================================
SEED_PLAYER = {
    "game_name": "PistaciovyBandit",
    "tag_line": "LIDL"
}

TARGET_QUEUE = 420              # Ranked Solo/Duo
MATCHES_PER_PLAYER = 8          # kolik matchů brát na hráče při expanzi
TARGET_MATCH_COUNT = 1500       # kolik unikátních matchů chceme nasbírat
MAX_PLAYER_EXPANSIONS = 250     # kolik hráčů max rozšířit
RECENT_FORM_MATCHES = 10        # kolik posledních her použít pro recent form
MIN_GAME_DURATION = 600         # odfiltruje remaky
RANDOM_SEED = 42
SAVE_EVERY_MATCHES = 10         # po kolika zpracovaných matchích uložit partial stav

# ==========================================
# CESTY
# ==========================================
DATA_DIR = Path("../data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

VISITED_PUUIDS_FILE = CHECKPOINT_DIR / "visited_puuids.json"
QUEUED_PUUIDS_FILE = CHECKPOINT_DIR / "queued_puuids.json"
PLAYER_QUEUE_FILE = CHECKPOINT_DIR / "player_queue.json"
MATCH_IDS_FILE = CHECKPOINT_DIR / "match_ids.json"
PROCESSED_MATCH_IDS_FILE = CHECKPOINT_DIR / "processed_match_ids.json"

PARTIAL_DATASET_FILE = PROCESSED_DIR / "dataset_partial.csv"
FINAL_DATASET_FILE = PROCESSED_DIR / "dataset.csv"

# ==========================================
# MAPOVÁNÍ RANKŮ
# ==========================================
RANK_ORDER = {
    "IRON": 1,
    "BRONZE": 2,
    "SILVER": 3,
    "GOLD": 4,
    "PLATINUM": 5,
    "EMERALD": 6,
    "DIAMOND": 7,
    "MASTER": 8,
    "GRANDMASTER": 9,
    "CHALLENGER": 10,
}

DIVISION_ORDER = {
    "IV": 1,
    "III": 2,
    "II": 3,
    "I": 4,
}

ROLE_ORDER = {
    "TOP": 1,
    "JUNGLE": 2,
    "MIDDLE": 3,
    "BOTTOM": 4,
    "UTILITY": 5
}

# ==========================================
# CACHE
# ==========================================
summoner_cache: Dict[str, Dict[str, Any]] = {}
rank_cache: Dict[str, float] = {}
mastery_cache: Dict[str, int] = {}
match_cache: Dict[str, Dict[str, Any]] = {}
recent_form_cache: Dict[str, float] = {}
account_cache: Dict[str, Dict[str, Any]] = {}

# ==========================================
# SOUBORY / JSON HELPERS
# ==========================================
def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def save_json_file(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_raw_match_json(match_id: str, match_data: Dict[str, Any]) -> None:
    path = RAW_DIR / f"{match_id}.json"
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(match_data, f, ensure_ascii=False, indent=2)


def save_collection_checkpoint(
    visited_puuids: Set[str],
    queued_puuids: Set[str],
    player_queue: deque,
    all_match_ids: Set[str],
) -> None:
    save_json_file(VISITED_PUUIDS_FILE, list(visited_puuids))
    save_json_file(QUEUED_PUUIDS_FILE, list(queued_puuids))
    save_json_file(PLAYER_QUEUE_FILE, list(player_queue))
    save_json_file(MATCH_IDS_FILE, list(all_match_ids))


def load_collection_checkpoint(seed_puuid: str):
    visited_puuids = set(load_json_file(VISITED_PUUIDS_FILE, []))
    queued_puuids = set(load_json_file(QUEUED_PUUIDS_FILE, []))
    player_queue = deque(load_json_file(PLAYER_QUEUE_FILE, []))
    all_match_ids = set(load_json_file(MATCH_IDS_FILE, []))

    if not player_queue and not visited_puuids and not all_match_ids:
        player_queue.append(seed_puuid)
        queued_puuids.add(seed_puuid)

    return visited_puuids, queued_puuids, player_queue, all_match_ids


def load_processed_match_ids() -> Set[str]:
    return set(load_json_file(PROCESSED_MATCH_IDS_FILE, []))


def save_processed_match_ids(processed_match_ids: Set[str]) -> None:
    save_json_file(PROCESSED_MATCH_IDS_FILE, list(processed_match_ids))


# ==========================================
# API HELPERS
# ==========================================
def safe_get(url: str, params: Optional[Dict[str, Any]] = None, retries: int = 5) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=20)

            if response.status_code == 200:
                return response

            if response.status_code == 401:
                print(f"[401] Neplatný nebo expirovaný Riot API key: {url}")
                return None

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_time = int(retry_after) if retry_after and retry_after.isdigit() else 5
                print(f"[429] Rate limit hit, čekám {wait_time}s...")
                time.sleep(wait_time)
                continue

            if response.status_code in (500, 502, 503, 504):
                wait_time = 2 ** attempt
                print(f"[{response.status_code}] Server error, retry za {wait_time}s...")
                time.sleep(wait_time)
                continue

            if response.status_code == 404:
                return None

            print(f"[WARN] {url} -> status {response.status_code}")
            return None

        except requests.RequestException as exc:
            wait_time = 2 ** attempt
            print(f"[EXC] {exc} -> retry za {wait_time}s...")
            time.sleep(wait_time)

    return None


def riot_get_account_by_riot_id(game_name: str, tag_line: str) -> Optional[Dict[str, Any]]:
    cache_key = f"{game_name}#{tag_line}"
    if cache_key in account_cache:
        return account_cache[cache_key]

    url = f"https://{REGIONAL_ROUTING}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    resp = safe_get(url)
    if not resp:
        return None

    data = resp.json()
    account_cache[cache_key] = data
    return data


def riot_get_summoner_by_puuid(puuid: str) -> Optional[Dict[str, Any]]:
    if puuid in summoner_cache:
        return summoner_cache[puuid]

    url = f"https://{PLATFORM_ROUTING}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}"
    resp = safe_get(url)
    if not resp:
        return None

    data = resp.json()
    summoner_cache[puuid] = data
    return data


def riot_get_rank_numeric_by_summoner_id(summoner_id: str) -> float:
    if summoner_id in rank_cache:
        return rank_cache[summoner_id]

    url = f"https://{PLATFORM_ROUTING}.api.riotgames.com/lol/league/v4/entries/by-summoner/{summoner_id}"
    resp = safe_get(url)
    if not resp:
        rank_cache[summoner_id] = 0.0
        return 0.0

    entries = resp.json()
    rank_value = 0.0

    for entry in entries:
        if entry.get("queueType") == "RANKED_SOLO_5x5":
            tier = entry.get("tier", "")
            division = entry.get("rank", "")
            lp = int(entry.get("leaguePoints", 0))

            tier_value = RANK_ORDER.get(tier, 0)
            division_value = DIVISION_ORDER.get(division, 0)

            if tier_value and division_value:
                rank_value = tier_value + (division_value / 10.0) + min(lp, 100) / 1000.0
            break

    rank_cache[summoner_id] = rank_value
    return rank_value


def riot_get_match_ids_by_puuid(puuid: str, count: int = MATCHES_PER_PLAYER) -> List[str]:
    url = f"https://{REGIONAL_ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {
        "start": 0,
        "count": count,
        "queue": TARGET_QUEUE,
    }
    resp = safe_get(url, params=params)
    return resp.json() if resp else []


def riot_get_match(match_id: str) -> Optional[Dict[str, Any]]:
    if match_id in match_cache:
        return match_cache[match_id]

    url = f"https://{REGIONAL_ROUTING}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    resp = safe_get(url)
    if not resp:
        return None

    data = resp.json()
    match_cache[match_id] = data
    return data


def riot_get_mastery_by_puuid_and_champion(puuid: str, champion_id: int) -> int:
    cache_key = f"{puuid}:{champion_id}"
    if cache_key in mastery_cache:
        return mastery_cache[cache_key]

    url = (
        f"https://{PLATFORM_ROUTING}.api.riotgames.com/"
        f"lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/by-champion/{champion_id}"
    )
    resp = safe_get(url)
    if not resp:
        mastery_cache[cache_key] = 0
        return 0

    data = resp.json()
    points = int(data.get("championPoints", 0))
    mastery_cache[cache_key] = points
    return points


# ==========================================
# FEATURE HELPERS
# ==========================================
def compute_recent_form(puuid: str, current_match_id: str) -> float:
    cache_key = f"{puuid}:{current_match_id}"
    if cache_key in recent_form_cache:
        return recent_form_cache[cache_key]

    match_ids = riot_get_match_ids_by_puuid(puuid, count=RECENT_FORM_MATCHES + 5)

    wins = 0
    total = 0

    for match_id in match_ids:
        if match_id == current_match_id:
            continue

            # unreachable? no, kept below
        match_data = riot_get_match(match_id)
        if not match_data:
            continue

        participants = match_data.get("info", {}).get("participants", [])
        player = next((p for p in participants if p.get("puuid") == puuid), None)
        if not player:
            continue

        wins += 1 if player.get("win") else 0
        total += 1

        if total >= RECENT_FORM_MATCHES:
            break

    result = round(wins / total, 4) if total > 0 else 0.0
    recent_form_cache[cache_key] = result
    return result


def extract_team(participants: List[Dict[str, Any]], team_id: int) -> List[Dict[str, Any]]:
    return [p for p in participants if p.get("teamId") == team_id]


def role_sort_key(role: str) -> int:
    return ROLE_ORDER.get(role, 99)


def normalize_team_roles(team_players: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(team_players, key=lambda p: role_sort_key(p.get("teamPosition", "")))


def add_team_features(row: Dict[str, Any], team_players: List[Dict[str, Any]], prefix: str, match_id: str) -> None:
    role_names = ["top", "jungle", "mid", "adc", "support"]

    rank_values = []
    recent_values = []
    mastery_values = []

    for i, player in enumerate(team_players):
        role_name = role_names[i] if i < len(role_names) else f"player_{i+1}"

        puuid = player.get("puuid", "")
        champion_id = int(player.get("championId", 0))
        champion_name = player.get("championName", "")
        riot_game_name = player.get("riotIdGameName", "")
        riot_tag_line = player.get("riotIdTagline", "")
        riot_id = f"{riot_game_name}#{riot_tag_line}" if riot_game_name else ""

        row[f"{prefix}_{role_name}_champion_id"] = champion_id
        row[f"{prefix}_{role_name}_champion_name"] = champion_name
        row[f"{prefix}_{role_name}_riot_id"] = riot_id
        row[f"{prefix}_{role_name}_puuid"] = puuid

        rank_numeric = 0.0
        summoner_data = riot_get_summoner_by_puuid(puuid)
        if summoner_data:
            summoner_id = summoner_data.get("id", "")
            if summoner_id:
                rank_numeric = riot_get_rank_numeric_by_summoner_id(summoner_id)

        recent_wr = compute_recent_form(puuid, match_id) if puuid else 0.0
        mastery = riot_get_mastery_by_puuid_and_champion(puuid, champion_id) if puuid and champion_id else 0

        row[f"{prefix}_{role_name}_rank_numeric"] = rank_numeric
        row[f"{prefix}_{role_name}_recent_wr"] = recent_wr
        row[f"{prefix}_{role_name}_mastery"] = mastery

        rank_values.append(rank_numeric)
        recent_values.append(recent_wr)
        mastery_values.append(mastery)

    row[f"{prefix}_avg_rank"] = round(sum(rank_values) / len(rank_values), 4) if rank_values else 0.0
    row[f"{prefix}_avg_recent_wr"] = round(sum(recent_values) / len(recent_values), 4) if recent_values else 0.0
    row[f"{prefix}_avg_mastery"] = round(sum(mastery_values) / len(mastery_values), 2) if mastery_values else 0.0


def build_row_from_match(match_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    metadata = match_data.get("metadata", {})
    info = match_data.get("info", {})

    if info.get("queueId") != TARGET_QUEUE:
        return None

    game_duration = int(info.get("gameDuration", 0))
    if game_duration < MIN_GAME_DURATION:
        return None

    participants = info.get("participants", [])
    if len(participants) != 10:
        return None

    blue_team = normalize_team_roles(extract_team(participants, 100))
    red_team = normalize_team_roles(extract_team(participants, 200))

    if len(blue_team) != 5 or len(red_team) != 5:
        return None

    match_id = metadata.get("matchId", "")
    teams = info.get("teams", [])
    blue_win = 0

    for team in teams:
        if team.get("teamId") == 100:
            blue_win = 1 if team.get("win") else 0
            break

    row: Dict[str, Any] = {
        "match_id": match_id,
        "game_creation": info.get("gameCreation"),
        "game_duration": game_duration,
        "queue_id": info.get("queueId"),
        "blue_win": blue_win,
    }

    add_team_features(row, blue_team, "blue", match_id)
    add_team_features(row, red_team, "red", match_id)

    return row


# ==========================================
# SBĚR MATCH ID
# ==========================================
def collect_match_ids_from_single_seed(seed_player: Dict[str, str]) -> Set[str]:
    random.seed(RANDOM_SEED)

    game_name = seed_player["game_name"]
    tag_line = seed_player["tag_line"]

    account = riot_get_account_by_riot_id(game_name, tag_line)
    if not account:
        raise ValueError(f"Nepodařilo se načíst seed účet {game_name}#{tag_line}")

    seed_puuid = account.get("puuid", "")
    if not seed_puuid:
        raise ValueError("Seed účet nemá puuid.")

    visited_puuids, queued_puuids, player_queue, all_match_ids = load_collection_checkpoint(seed_puuid)

    expansions = len(visited_puuids)

    while player_queue and len(all_match_ids) < TARGET_MATCH_COUNT and expansions < MAX_PLAYER_EXPANSIONS:
        puuid = player_queue.popleft()

        if puuid in visited_puuids:
            continue

        visited_puuids.add(puuid)
        expansions += 1

        match_ids = riot_get_match_ids_by_puuid(puuid, count=MATCHES_PER_PLAYER)
        random.shuffle(match_ids)

        print(
            f"[INFO] Hráč {expansions} | "
            f"matchů hráče: {len(match_ids)} | "
            f"celkem unikátních matchů: {len(all_match_ids)} | "
            f"fronta: {len(player_queue)}"
        )

        for match_id in match_ids:
            if len(all_match_ids) >= TARGET_MATCH_COUNT:
                break

            if match_id in all_match_ids:
                continue

            match_data = riot_get_match(match_id)
            if not match_data:
                continue

            info = match_data.get("info", {})
            if info.get("queueId") != TARGET_QUEUE:
                continue

            if int(info.get("gameDuration", 0)) < MIN_GAME_DURATION:
                continue

            participants = info.get("participants", [])
            if len(participants) != 10:
                continue

            all_match_ids.add(match_id)

            random.shuffle(participants)
            for participant in participants:
                participant_puuid = participant.get("puuid", "")
                if (
                    participant_puuid
                    and participant_puuid not in visited_puuids
                    and participant_puuid not in queued_puuids
                ):
                    player_queue.append(participant_puuid)
                    queued_puuids.add(participant_puuid)

        save_collection_checkpoint(visited_puuids, queued_puuids, player_queue, all_match_ids)

    print(f"[DONE] Nasbíráno unikátních matchů: {len(all_match_ids)}")
    print(f"[DONE] Navštívených hráčů: {len(visited_puuids)}")

    return all_match_ids


# ==========================================
# ZPRACOVÁNÍ DATASETU
# ==========================================
def crawl_dataset() -> pd.DataFrame:
    all_match_ids = collect_match_ids_from_single_seed(SEED_PLAYER)
    print(f"Celkem unikátních matchů ke zpracování: {len(all_match_ids)}")

    processed_match_ids = load_processed_match_ids()

    rows: List[Dict[str, Any]]
    if PARTIAL_DATASET_FILE.exists():
        df_existing = pd.read_csv(PARTIAL_DATASET_FILE)
        rows = df_existing.to_dict(orient="records")
        print(f"[RESUME] Načten partial dataset s {len(rows)} řádky")
    else:
        rows = []

    shuffled_match_ids = list(all_match_ids)
    random.shuffle(shuffled_match_ids)

    processed_since_save = 0

    for match_id in tqdm(shuffled_match_ids, desc="Zpracovávám zápasy"):
        if match_id in processed_match_ids:
            continue

        match_data = riot_get_match(match_id)
        if not match_data:
            continue

        save_raw_match_json(match_id, match_data)

        row = build_row_from_match(match_data)
        processed_match_ids.add(match_id)

        if row:
            rows.append(row)

        processed_since_save += 1

        if processed_since_save >= SAVE_EVERY_MATCHES:
            df_partial = pd.DataFrame(rows)
            if not df_partial.empty:
                df_partial = df_partial.drop_duplicates(subset=["match_id"]).reset_index(drop=True)
                df_partial.to_csv(PARTIAL_DATASET_FILE, index=False, encoding="utf-8")

            save_processed_match_ids(processed_match_ids)
            processed_since_save = 0
            print(f"[SAVE] Průběžně uloženo. Řádků: {len(df_partial) if 'df_partial' in locals() else 0}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["match_id"]).reset_index(drop=True)

    df.to_csv(FINAL_DATASET_FILE, index=False, encoding="utf-8")
    save_processed_match_ids(processed_match_ids)

    return df


# ==========================================
# MAIN
# ==========================================
def main() -> None:
    ensure_directories()

    df = crawl_dataset()

    print(f"\nHotovo. Uloženo {len(df)} řádků do: {FINAL_DATASET_FILE}")


if __name__ == "__main__":
    main()