"""
Fetch FanGraphs pitcher leaderboards (incl. per-pitch-type Stuff+) via
pybaseball's normal FanGraphs function, for the external check of the
label-sensitivity result (docs/DISCOVERY_FINDINGS.md).

Polite by design: one request per season, a pause between seasons, and it
STOPS on any block (HTTP 403/429, a challenge page, or empty data). It does
not retry around or imitate a browser. If it stops, use FanGraphs' own
leaderboard CSV export instead.

Run (use the framework Python; the anaconda one has a broken pandas):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
      src/stuffplus/fetch_fangraphs_stuffplus.py 2025 2026
Writes data/fangraphs_pitching_<season>.csv
"""
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
PAUSE_S = 10


def fetch(season):
    from pybaseball import pitching_stats
    try:
        df = pitching_stats(season, qual=0)
    except Exception as e:          # HTTP errors, challenge pages that fail to parse, etc.
        sys.exit(f"{season}: FanGraphs request failed ({type(e).__name__}: {e}). "
                 "Stopping - use the leaderboard's CSV export instead.")
    if df is None or len(df) == 0:
        sys.exit(f"{season}: no rows returned. Stopping - use the leaderboard's CSV export instead.")
    stuff = [c for c in df.columns if 'stf' in c.lower() or 'stuff' in c.lower()]
    if not stuff:
        sys.exit(f"{season}: got {len(df)} rows but no Stuff+ columns "
                 f"(columns: {list(df.columns)[:15]} ...). Stopping.")
    keep = [c for c in ['IDfg', 'Name', 'Team', 'Throws', 'IP', 'Pitches'] if c in df.columns] + stuff
    out = _ROOT / 'data' / f'fangraphs_pitching_{season}.csv'
    df[keep].to_csv(out, index=False)
    print(f"{season}: {len(df)} pitchers, {len(stuff)} Stuff+ columns -> {out}")
    print(f"   e.g. {stuff[:10]}")


if __name__ == "__main__":
    seasons = [int(s) for s in sys.argv[1:]] or [2025]
    for i, s in enumerate(seasons):
        if i:
            time.sleep(PAUSE_S)
        fetch(s)
