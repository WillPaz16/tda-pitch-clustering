import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pybaseball.statcast import statcast
from pybaseball.league_pitching_stats import bwar_pitch
from stuff_plus_calculator import StuffPlusCalculator


def load_statcast_2025(start_date="2025-03-28", end_date="2025-4-04"):
    print(f"Loading Statcast data for 2025 season: {start_date} → {end_date}")
    return statcast(start_date, end_date)


def load_pitcher_war(season=2025):
    print(f"Loading WAR data from pybaseball.league_pitching_stats.bwar_pitch() for {season}")
    war_df = bwar_pitch(return_all=False)
    war_df = war_df[war_df["year_ID"] == season].copy()
    war_df = war_df[["name_common", "WAR"]].copy()
    war_df.columns = ["Name", "WAR"]
    war_df["Name_norm"] = war_df["Name"].astype(str).str.strip().str.lower()
    return war_df[["Name_norm", "WAR"]]


def build_pitcher_name_map(df):
    name_map_df = df.drop_duplicates('pitcher_id')[['pitcher_id', 'player_name']].copy()
    name_map_df = name_map_df.rename(columns={'player_name': 'pitcher'})

    def flip_name(name):
        if pd.isna(name):
            return name
        name_str = str(name).strip()
        if ',' in name_str:
            parts = name_str.split(',')
            return f"{parts[1].strip()} {parts[0].strip()}"
        return name_str

    name_map_df['pitcher_flipped'] = name_map_df['pitcher'].apply(flip_name)
    name_map_df['pitcher_norm'] = name_map_df['pitcher_flipped'].astype(str).str.strip().str.lower()

    return name_map_df[['pitcher_id', 'pitcher_norm']]


def compute_pitcher_weighted_z(per_pt, weights):
    per_pt = per_pt.copy()
    per_pt["stuff_z_pt"] = (
        weights[0] * per_pt["z_xw_pt"]
        + weights[1] * per_pt["z_miss_pt"]
        + weights[2] * per_pt["z_chase_pt"]
    )
    pitcher_z = (
        per_pt.groupby("pitcher_id")
        .apply(
            lambda g: np.average(g["stuff_z_pt"], weights=g["pitch_count_pt"])
            if g["pitch_count_pt"].sum() > 0
            else np.nan
        )
        .rename("weighted_stuff_z")
        .reset_index()
    )
    return pitcher_z


def evaluate_weights(per_pt, pitcher_name_map, war_df, weights, verbose=False):
    pitcher_z = compute_pitcher_weighted_z(per_pt, weights)
    merged = pitcher_z.merge(pitcher_name_map, on="pitcher_id", how="left")
    merged = merged.merge(war_df, left_on="pitcher_norm", right_on="Name_norm", how="left")

    merged = merged.dropna(subset=["weighted_stuff_z", "WAR"])

    if len(merged) < 10:
        return np.nan

    corr = merged["weighted_stuff_z"].corr(merged["WAR"])
    return corr


def search_best_weights_constrained(war_df, step=0.1, min_weight=0.05):
    """Constrained search with larger step size for testing"""
    best = {"weights": None, "corr": -np.inf}
    grid = np.arange(min_weight, 1.0 + 1e-9, step)

    print(f"Testing constrained weights (min_weight={min_weight}, step={step})")

    for w0 in grid:
        for w1 in grid:
            w2 = 1.0 - w0 - w1
            if w2 < min_weight or w2 > 1.0:
                continue

            # For testing, just evaluate first valid combination
            corr = 0.1  # Mock correlation for testing
            if pd.notna(corr) and corr > best["corr"]:
                best = {"weights": [w0, w1, w2], "corr": corr}
                print(f"Found valid weights: {best['weights']}")
                return best  # Return immediately for testing

    return best


# Quick test
if __name__ == "__main__":
    print("Testing constrained weight search...")
    war_df = pd.DataFrame({'Name_norm': ['test'], 'WAR': [1.0]})
    result = search_best_weights_constrained(war_df, step=0.2, min_weight=0.1)
    print("Test result:", result)