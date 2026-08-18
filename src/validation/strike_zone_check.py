"""
Fixed vs. Per-Pitch Strike Zone Check

Follow-up to docs/METHODOLOGY_REVIEW.md's objection: notebooks/ProStuff+.ipynb
defines chase% using a fixed rectangular zone (zone_z_min, zone_z_max = 1.6, 3.5
feet) applied to every batter, instead of the actual per-pitch, batter-specific
sz_top/sz_bot columns Statcast already provides. This checks how much that
approximation actually matters, rather than just asserting it does.

Pulls one day of real Statcast data directly from the Baseball Savant CSV
export (same approach as src/tda/classify_pitches_to_csv.py -- this bypasses
a real, separately-discovered pybaseball.statcast() version-compatibility bug
in the currently installed environment: the network call succeeds but
pybaseball's own postprocessing step crashes on a duplicate-column issue,
independent of this strike-zone question but worth knowing about since
several other scripts in this repo depend on pybaseball.statcast()).

Compares chase% under the fixed zone vs. the real per-pitch zone, both at
the aggregate rate and the individual-pitch label-agreement level.

Analysis/reporting script. See docs/METHODOLOGY_REVIEW.md for write-up.
"""

import warnings
warnings.filterwarnings("ignore")

import io
import requests
import pandas as pd

# matches notebooks/ProStuff+.ipynb's chase% definition exactly
ZONE_X_MIN, ZONE_X_MAX = -8.5 / 12, 8.5 / 12
ZONE_Z_MIN, ZONE_Z_MAX = 1.6, 3.5

SWING_OUTCOMES = ['swinging_strike', 'swinging_strike_blocked', 'missed_bunt',
                  'foul', 'foul_bunt', 'foul_tip', 'bunt_foul_tip', 'hit_into_play']


def fetch_savant_csv(start_date, end_date):
    """Same direct Baseball Savant CSV export src/tda/classify_pitches_to_csv.py
    uses -- avoids the pybaseball.statcast() postprocessing bug entirely."""
    url = (
        "https://baseballsavant.mlb.com/statcast_search/csv?"
        "all=true&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&"
        "hfGT=R%7CPO%7CS%7C=&hfSea=&hfSit=&player_type=pitcher&hfOuts=&opponent=&"
        "pitcher_throws=&batter_stands=&hfSA=&game_date_gt={}&game_date_lt={}&"
        "team=&position=&hfRO=&home_road=&hfFlag=&metric_1=&hfInn=&min_pitches=0&"
        "min_results=0&group_by=name&sort_col=pitches&player_event_sort=h_launch_speed&"
        "sort_order=desc&min_abs=0&type=details&"
    ).format(start_date, end_date)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def main(start_date='2025-09-15', end_date='2025-09-15'):
    df = fetch_savant_csv(start_date, end_date)
    df = df.dropna(subset=['plate_x', 'plate_z', 'sz_top', 'sz_bot'])
    print(f"Pitches with complete plate/zone data: {len(df)}")

    fixed_in_zone = (df['plate_x'].ge(ZONE_X_MIN)) & (df['plate_x'].le(ZONE_X_MAX)) & \
                     (df['plate_z'].ge(ZONE_Z_MIN)) & (df['plate_z'].le(ZONE_Z_MAX))
    real_in_zone = (df['plate_x'].ge(ZONE_X_MIN)) & (df['plate_x'].le(ZONE_X_MAX)) & \
                    (df['plate_z'].ge(df['sz_bot'])) & (df['plate_z'].le(df['sz_top']))

    is_swing = df['description'].isin(SWING_OUTCOMES)
    chase_fixed = is_swing & ~fixed_in_zone
    chase_real = is_swing & ~real_in_zone

    print(f"\nChase rate (fixed {ZONE_Z_MIN}-{ZONE_Z_MAX}ft zone): "
          f"{chase_fixed.sum()} of {len(df)} = {100 * chase_fixed.mean():.2f}%")
    print(f"Chase rate (real per-batter zone):     "
          f"{chase_real.sum()} of {len(df)} = {100 * chase_real.mean():.2f}%")

    disagree = chase_fixed != chase_real
    print(f"\nIndividual pitches where the two definitions disagree on chase/not-chase: "
          f"{disagree.sum()} ({100 * disagree.mean():.2f}%)")

    print("\nReal per-batter strike zone top/bottom range in this sample:")
    print(df[['sz_top', 'sz_bot']].describe())


if __name__ == "__main__":
    main()
