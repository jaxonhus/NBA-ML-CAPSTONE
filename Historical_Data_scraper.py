"""
NBA Historical Data Collector (1996–2025)
==========================================
Pulls data from 11 nba_api endpoints and saves each to its own CSV.

Install:
    pip install nba_api pandas tqdm requests

Run:
    python nba_data_collector.py

Output (one CSV per endpoint):
    ./nba_data/
        01_player_career_stats.csv     — career + season-by-season totals per player
        02_player_game_logs.csv        — every player game log, every season
        03_player_awards.csv           — awards history per player
        04_league_game_logs.csv        — league-wide game logs per season
        05_league_leaders.csv          — scoring/ast/reb leaders per season
        06_league_dash_pt_stats.csv    — league dashboard passing/touch stats
        07_standings.csv               — final standings per season
        08_team_dash_lineups.csv       — team lineup stats per season
        09_draft_history.csv           — NBA draft history (all years)
        10_shot_chart_detail.csv       — shot chart data per season (sampled)
        11_game_rotation.csv           — player rotation data per game (sampled)

Checkpoint/resume: if interrupted, just re-run — it picks up where it left off.
"""

import time
import os
import requests
import pandas as pd
from tqdm import tqdm
import urllib3
import threading
import signal

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Patch requests.Session BEFORE importing nba_api endpoints ────────────────
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "DNT": "1",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

_OriginalSession = requests.Session

class PatchedSession(_OriginalSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.headers.update(HEADERS)
        adapter = requests.adapters.HTTPAdapter(
            max_retries=urllib3.Retry(
                total=3,
                backoff_factor=2,
                status_forcelist=[429, 500, 502, 503, 504],
            )
        )
        self.mount("https://", adapter)
        self.mount("http://", adapter)

requests.Session = PatchedSession

# ── Now safe to import nba_api ───────────────────────────────────────────────
from nba_api.stats.endpoints import (
    PlayerCareerStats,
    PlayerGameLog,
    PlayerAwards,
    LeagueGameLog,
    LeagueLeaders,
    LeagueDashPtStats,
    LeagueStandings,
    TeamDashLineups,
    DraftHistory,
    ShotChartDetail,
    GameRotation,
)
from nba_api.stats.static import players as nba_players, teams as nba_teams

# ── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR       = "./nba_data"
SEASONS          = [f"{y}-{str(y+1)[-2:]}" for y in range(1996, 2025)]
DELAY            = 0.8
TIMEOUT          = 60    # per-request HTTP timeout (seconds)
CALL_TIMEOUT     = 90    # wall-clock timeout per safe_get call before we give up and move on

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_get(endpoint_cls, retries=4, **kwargs):
    """Call endpoint with exponential-backoff retry AND a hard wall-clock timeout.
    If the call hangs longer than CALL_TIMEOUT seconds, it is abandoned and
    None is returned so the caller can skip to the next data point.
    """
    for attempt in range(retries):
        result_holder = [None]
        error_holder  = [None]

        def _call():
            try:
                result_holder[0] = endpoint_cls(timeout=TIMEOUT, headers=HEADERS, **kwargs)
            except KeyError as e:
                error_holder[0] = ("key", e)
            except Exception as e:
                error_holder[0] = ("other", e)

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=CALL_TIMEOUT)

        if t.is_alive():
            # Thread is still stuck — abandon it and skip this data point entirely
            print(f"    ⏱  Timed out after {CALL_TIMEOUT}s — skipping.")
            return None

        if error_holder[0]:
            etype, exc = error_holder[0]
            if etype == "key":
                print(f"    ✗ Unexpected response structure (KeyError: {exc}) — skipping.")
                return None
            wait = 5 * (2 ** attempt)
            print(f"    ⚠ Attempt {attempt+1}/{retries} failed: {exc}")
            print(f"      Waiting {wait}s...")
            time.sleep(wait)
            continue

        time.sleep(DELAY)
        return result_holder[0]

    print(f"    ✗ All retries exhausted — skipping.")
    return None


def to_df(result, idx=0):
    if result is None:
        return pd.DataFrame()
    try:
        frames = result.get_data_frames()
        if not frames or idx >= len(frames):
            return pd.DataFrame()
        return frames[idx]
    except Exception as e:
        print(f"    ✗ Failed to parse data frames: {e}")
        return pd.DataFrame()


def ckpt_path(name):
    return os.path.join(OUTPUT_DIR, f"_ckpt_{name}.csv")


def load_ckpt(name, key_col="SEASON"):
    path = ckpt_path(name)
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        offset = df[key_col].nunique() if key_col in df.columns else 0
        print(f"  ↩  Resuming '{name}' — skipping first {offset} {key_col}s")
        return df, offset
    return pd.DataFrame(), 0


def save_ckpt(name, df):
    df.to_csv(ckpt_path(name), index=False)


def finalize(label, name, df):
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    c = ckpt_path(name)
    if os.path.exists(c):
        os.remove(c)
    mb = os.path.getsize(path) / 1_000_000
    print(f"  ✅ {label}: {len(df):,} rows  ({mb:.1f} MB)  →  {path}")
    return df


def season_bar(seasons_list, desc):
    return tqdm(seasons_list, desc=f"{desc:<32}", unit="season")



# ════════════════════════════════════════════════════════════════════════════
# 02 — playergamelog.py
#      Every game a player played — one PlayerGameLog call per player/season.
#      Resumes automatically from checkpoint using (player_id, season) pairs.
# ════════════════════════════════════════════════════════════════════════════

def collect_player_game_logs():
    print("\n── 02 playergamelog ─────────────────────────────────────────────")

    all_players = nba_players.get_players()

    # Load existing checkpoint rows
    ckpt_file = ckpt_path("02_player_game_logs")
    existing_df = pd.DataFrame()
    done_pairs  = set()          # (player_id, season) already in the checkpoint

    if os.path.exists(ckpt_file):
        print(f"  ↩  Loading checkpoint ...")
        existing_df = pd.read_csv(ckpt_file, low_memory=False)
        print(f"     {len(existing_df):,} rows already collected")

        if "PLAYER_ID" in existing_df.columns and "SEASON" in existing_df.columns:
            done_pairs = set(
                zip(existing_df["PLAYER_ID"].astype(str),
                    existing_df["SEASON"].astype(str))
            )
            print(f"     {len(done_pairs):,} (player, season) pairs already done — skipping them")
    else:
        print("  No checkpoint found — starting from scratch.")

    rows = [existing_df] if not existing_df.empty else []
    ckpt_counter = 0

    for season in season_bar(SEASONS, "02 Player Game Logs"):
        for player in all_players:
            pid = str(player["id"])

            for stype in ["Regular Season", "Playoffs"]:
                if stype == "Regular Season" and (pid, season) in done_pairs:
                    continue

                result = safe_get(
                    PlayerGameLog,
                    player_id=player["id"],
                    season=season,
                    season_type_all_star=stype,
                )
                df = to_df(result)
                if not df.empty:
                    df["PLAYER_ID"]   = player["id"]
                    df["PLAYER_NAME"] = player["full_name"]
                    df["SEASON"]      = season
                    df["SEASON_TYPE"] = stype
                    rows.append(df)

            done_pairs.add((pid, season))
            ckpt_counter += 1
            if ckpt_counter % 100 == 0 and rows:
                save_ckpt("02_player_game_logs", pd.concat(rows, ignore_index=True))

        if rows:
            save_ckpt("02_player_game_logs", pd.concat(rows, ignore_index=True))
            print(f"  ✔ Season {season} done.")

    if rows:
        return finalize("Player Game Logs", "02_player_game_logs",
                        pd.concat(rows, ignore_index=True))
    return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# 03 — playerawards.py
#      All-NBA, All-Star, MVP, ROY, DPOY, etc. per player.
# ════════════════════════════════════════════════════════════════════════════
def collect_player_awards():
    print("\n── 03 playerawards ──────────────────────────────────────────────")
    all_players = nba_players.get_players()

    existing, offset = load_ckpt("03_player_awards", key_col="PLAYER_ID")
    rows = [existing] if not existing.empty else []

    remaining = all_players[offset:]
    for player in tqdm(remaining, desc="03 Player Awards        ", unit="player"):
        result = safe_get(PlayerAwards, player_id=player["id"])
        df = to_df(result)
        if not df.empty:
            df["PLAYER_ID"]   = player["id"]
            df["PLAYER_NAME"] = player["full_name"]
            rows.append(df)
            if len(rows) % 100 == 0:
                save_ckpt("03_player_awards", pd.concat(rows, ignore_index=True))

    if rows:
        return finalize("Player Awards", "03_player_awards",
                        pd.concat(rows, ignore_index=True))
    return pd.DataFrame()



# ════════════════════════════════════════════════════════════════════════════
# 05 — leagueleaders.py
#      Top stat leaders (PTS, REB, AST, STL, BLK, shooting %) per season.
# ════════════════════════════════════════════════════════════════════════════
def collect_league_leaders():
    print("\n── 05 leagueleaders ─────────────────────────────────────────────")
    existing, offset = load_ckpt("05_league_leaders")
    rows = [existing] if not existing.empty else []

    stat_categories = ["PTS", "REB", "AST", "STL", "BLK", "FG_PCT", "FT_PCT", "FG3_PCT"]
    remaining = SEASONS[offset:]

    for season in season_bar(remaining, "05 League Leaders"):
        for stat in stat_categories:
            result = safe_get(
                LeagueLeaders,
                season=season,
                season_type_all_star="Regular Season",
                stat_category_abbreviation=stat,
                league_id="00",
                per_mode48="PerGame",
            )
            df = to_df(result)
            if not df.empty:
                df["SEASON"]   = season
                df["STAT_CAT"] = stat
                rows.append(df)

        save_ckpt("05_league_leaders", pd.concat(rows, ignore_index=True))

    if rows:
        return finalize("League Leaders", "05_league_leaders",
                        pd.concat(rows, ignore_index=True))
    return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# 06 — leaguedashptstats.py
#      Passing, touches, speed/distance, drives (tracking era: 2013-14+).
# ════════════════════════════════════════════════════════════════════════════
def collect_league_dash_pt_stats():
    print("\n── 06 leaguedashptstats ─────────────────────────────────────────")
    tracking_seasons = [s for s in SEASONS if int(s[:4]) >= 2013]

    existing, offset = load_ckpt("06_league_dash_pt_stats")
    rows = [existing] if not existing.empty else []

    pt_types  = ["Passing", "Touches", "SpeedDistance", "Rebounding", "Drives"]
    remaining = tracking_seasons[offset:]

    for season in season_bar(remaining, "06 Dash PT Stats"):
        for pt_type in pt_types:
            result = safe_get(
                LeagueDashPtStats,
                season=season,
                per_mode_simple="PerGame",
                player_or_team="Player",
                pt_measure_type=pt_type,
                season_type_all_star="Regular Season",
            )
            df = to_df(result)
            if not df.empty:
                df["SEASON"]     = season
                df["PT_MEASURE"] = pt_type
                rows.append(df)

        save_ckpt("06_league_dash_pt_stats", pd.concat(rows, ignore_index=True))

    if rows:
        return finalize("League Dash PT Stats", "06_league_dash_pt_stats",
                        pd.concat(rows, ignore_index=True))
    return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# 07 — iststandings.py  (LeagueStandings)
#      Final regular-season standings every year.
# ════════════════════════════════════════════════════════════════════════════
def collect_standings():
    print("\n── 07 iststandings ──────────────────────────────────────────────")
    existing, offset = load_ckpt("07_standings")
    rows = [existing] if not existing.empty else []

    remaining = SEASONS[offset:]
    for season in season_bar(remaining, "07 Standings"):
        result = safe_get(
            LeagueStandings,
            season=season,
            season_type="Regular Season",
            league_id="00",
        )
        df = to_df(result)
        if not df.empty:
            df["SEASON"] = season
            rows.append(df)
            save_ckpt("07_standings", pd.concat(rows, ignore_index=True))

    if rows:
        return finalize("Standings", "07_standings",
                        pd.concat(rows, ignore_index=True))
    return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# 08 — teamdashlineups.py
#      5-man lineup stats per team per season.
# ════════════════════════════════════════════════════════════════════════════
def collect_team_dash_lineups():
    print("\n── 08 teamdashlineups ───────────────────────────────────────────")
    all_teams = nba_teams.get_teams()

    existing, offset = load_ckpt("08_team_dash_lineups")
    rows = [existing] if not existing.empty else []

    remaining = SEASONS[offset:]
    for season in season_bar(remaining, "08 Team Dash Lineups"):
        season_rows = []
        for team in all_teams:
            result = safe_get(
                TeamDashLineups,
                team_id=team["id"],
                season=season,
                season_type_all_star="Regular Season",
                per_mode_simple="PerGame",
                group_quantity=5,
            )
            df = to_df(result)
            if not df.empty:
                df["SEASON"]    = season
                df["TEAM_NAME"] = team["full_name"]
                df["TEAM_ID"]   = team["id"]
                season_rows.append(df)

        if season_rows:
            rows.append(pd.concat(season_rows, ignore_index=True))
            save_ckpt("08_team_dash_lineups", pd.concat(rows, ignore_index=True))

    if rows:
        return finalize("Team Dash Lineups", "08_team_dash_lineups",
                        pd.concat(rows, ignore_index=True))
    return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# 09 — drafthistory.py
#      Every NBA draft pick ever — single API call, all years.
# ════════════════════════════════════════════════════════════════════════════
def collect_draft_history():
    print("\n── 09 drafthistory ──────────────────────────────────────────────")
    path = os.path.join(OUTPUT_DIR, "09_draft_history.csv")
    if os.path.exists(path):
        print(f"  ↩  Already exists — skipping")
        return pd.read_csv(path)

    result = safe_get(DraftHistory, league_id="00")
    df = to_df(result)
    if not df.empty:
        return finalize("Draft History", "09_draft_history", df)
    return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# 10 — shotchartdetail.py
#      Shot location (x/y court coords) + make/miss for every player.
# ════════════════════════════════════════════════════════════════════════════
def collect_shot_chart_detail():
    print("\n── 10 shotchartdetail ───────────────────────────────────────────")
    all_players = nba_players.get_players()

    existing, offset = load_ckpt("10_shot_chart_detail", key_col="PLAYER_ID")
    rows = [existing] if not existing.empty else []

    remaining = all_players[offset:]
    for player in tqdm(remaining, desc="10 Shot Chart Detail    ", unit="player"):
        result = safe_get(
            ShotChartDetail,
            player_id=player["id"],
            team_id=0,
            season_nullable="",            # blank = career (all seasons)
            season_type_all_star="Regular Season",
            context_measure_simple="FGA",
            league_id="00",
        )
        df = to_df(result, idx=0)
        if not df.empty:
            df["PLAYER_NAME"] = player["full_name"]
            rows.append(df)
            if len(rows) % 100 == 0:
                save_ckpt("10_shot_chart_detail", pd.concat(rows, ignore_index=True))

    if rows:
        return finalize("Shot Chart Detail", "10_shot_chart_detail",
                        pd.concat(rows, ignore_index=True))
    return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# 11 — gamerotation.py
#      Player rotation / minutes intervals per game.
#      Derives game IDs from the league game log (run #04 first).
# ════════════════════════════════════════════════════════════════════════════
def collect_game_rotation(league_game_log_df=None):
    print("\n── 11 gamerotation ──────────────────────────────────────────────")

    game_log_path = os.path.join(OUTPUT_DIR, "04_league_game_logs.csv")
    if league_game_log_df is None or league_game_log_df.empty:
        if os.path.exists(game_log_path):
            print("  Loading game IDs from 04_league_game_logs.csv...")
            league_game_log_df = pd.read_csv(game_log_path, low_memory=False)
        else:
            print("  ⚠  No game log found — run collect_league_game_logs() first.")
            return pd.DataFrame()

    game_ids = (
        league_game_log_df[["GAME_ID", "SEASON", "GAME_DATE"]]
        .drop_duplicates("GAME_ID")
        .sort_values("GAME_DATE")
    )

    existing, offset = load_ckpt("11_game_rotation", key_col="GAME_ID")
    rows = [existing] if not existing.empty else []

    remaining = game_ids.iloc[offset:]
    print(f"  {len(remaining):,} games to fetch  ({offset:,} already done)")

    for _, row in tqdm(remaining.iterrows(), total=len(remaining),
                       desc="11 Game Rotation        ", unit="game"):
        result = safe_get(GameRotation, game_id=str(int(row["GAME_ID"])).zfill(10))
        for idx, side in [(0, "home"), (1, "away")]:
            df = to_df(result, idx=idx)
            if not df.empty:
                df["GAME_ID"] = row["GAME_ID"]
                df["SEASON"]  = row["SEASON"]
                df["SIDE"]    = side
                rows.append(df)

        if len(rows) % 200 == 0 and rows:
            save_ckpt("11_game_rotation", pd.concat(rows, ignore_index=True))

    if rows:
        return finalize("Game Rotation", "11_game_rotation",
                        pd.concat(rows, ignore_index=True))
    return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 62)
    print("  🏀  NBA Historical Data Collector  (1996–2025)")
    print("=" * 62)
    print(f"  Seasons : {SEASONS[0]} → {SEASONS[-1]}  ({len(SEASONS)} seasons)")
    print(f"  Output  : {os.path.abspath(OUTPUT_DIR)}/")
    print(f"  Tip     : Re-run anytime — checkpoints resume from last spot")
    print("=" * 62)

    results = {}
    results["player_gamelogs"] = collect_player_game_logs()      # 02
    results["player_awards"]   = collect_player_awards()         # 03
    results["league_leaders"]  = collect_league_leaders()        # 05
    results["dash_pt_stats"]   = collect_league_dash_pt_stats()  # 06
    results["standings"]       = collect_standings()             # 07
    results["team_lineups"]    = collect_team_dash_lineups()     # 08
    results["draft_history"]   = collect_draft_history()         # 09
    results["shot_charts"]     = collect_shot_chart_detail()     # 10
    results["game_rotation"]   = collect_game_rotation()         # 11 (loads 04 from file)

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  ✅  ALL DONE — OUTPUT FILES")
    print("=" * 62)
    output_files = [
        ("01_player_career_stats",  "playercareerstats.py"),
        ("02_player_game_logs",     "playergamelog.py"),
        ("03_player_awards",        "playerawards.py"),
        ("04_league_game_logs",     "leaguegamelog.py"),
        ("05_league_leaders",       "leagueleaders.py"),
        ("06_league_dash_pt_stats", "leaguedashptstats.py"),
        ("07_standings",            "iststandings.py"),
        ("08_team_dash_lineups",    "teamdashlineups.py"),
        ("09_draft_history",        "drafthistory.py"),
        ("10_shot_chart_detail",    "shotchartdetail.py"),
        ("11_game_rotation",        "gamerotation.py"),
    ]
    for fname, source in output_files:
        path = os.path.join(OUTPUT_DIR, f"{fname}.csv")
        if os.path.exists(path):
            size = os.path.getsize(path) / 1_000_000
            row_count = sum(1 for _ in open(path)) - 1
            print(f"  {fname}.csv")
            print(f"      ← {source:<28}  {row_count:>10,} rows  {size:>6.1f} MB")
        else:
            print(f"  {fname}.csv  ← {source}  [not generated]")
    print("=" * 62)