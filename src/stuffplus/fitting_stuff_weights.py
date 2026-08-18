import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from pybaseball.statcast import statcast
from pybaseball import bwar_pitch
from stuff_plus_calculator import StuffPlusCalculator, EQUAL_STUFFPLUS_WEIGHTS

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / 'data'


def load_statcast_2025(start_date="2025-03-28", end_date="2025-11-04"):
    print(f"Loading Statcast data for 2025 season: {start_date} → {end_date}")
    return statcast(start_date, end_date)


def load_pitcher_war(season=2025):
    print(f"Loading WAR data from pybaseball.league_pitching_stats.bwar_pitch() for {season}")
    war_df = bwar_pitch(return_all=False)
    if "year_ID" not in war_df.columns or "name_common" not in war_df.columns or "WAR" not in war_df.columns:
        raise RuntimeError("Unexpected bwar_pitch output: missing required columns.")

    war_df = war_df[war_df["year_ID"] == season].copy()
    if war_df.empty:
        raise RuntimeError(f"No WAR data found for season {season}.")

    war_df = war_df[["name_common", "WAR"]].copy()
    war_df.columns = ["Name", "WAR"]
    war_df["Name_norm"] = war_df["Name"].astype(str).str.strip().str.lower()
    return war_df[["Name_norm", "WAR"]]


def build_pitcher_name_map(df):
    if 'pitcher' in df.columns:
        name_col = 'pitcher'
    elif 'pitcher_name' in df.columns:
        name_col = 'pitcher_name'
    else:
        raise RuntimeError(
            "Cannot build pitcher name map: missing 'pitcher' or 'pitcher_name' column."
        )

    # Get unique pitcher_id and name pairs
    name_map_df = df.drop_duplicates('pitcher_id')[['pitcher_id', name_col]].copy()
    name_map_df = name_map_df.rename(columns={name_col: 'pitcher'})
    
    # Convert names from "Last, First" format to "First Last" format
    def flip_name(name):
        """Convert 'Last, First' to 'First Last'"""
        if pd.isna(name):
            return name
        name_str = str(name).strip()
        if ',' in name_str:
            parts = name_str.split(',')
            # parts[0] = Last, parts[1] = First
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
    
    if verbose:
        print(f"    Merged rows before dropna: {len(merged)}")
        print(f"    Non-null weighted_stuff_z: {merged['weighted_stuff_z'].notna().sum()}")
        print(f"    Non-null WAR: {merged['WAR'].notna().sum()}")
    
    merged = merged.dropna(subset=["weighted_stuff_z", "WAR"])
    
    if verbose:
        print(f"    Merged rows after dropna: {len(merged)}")
    
    if len(merged) < 10:
        return np.nan
    
    corr = merged["weighted_stuff_z"].corr(merged["WAR"])
    return corr


def search_best_weights(per_pt, pitcher_name_map, target_df, step=0.02):
    best = {"weights": None, "corr": -np.inf}
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    
    # Debug first combination
    first_combo = True
    
    for w0 in grid:
        for w1 in grid:
            w2 = 1.0 - w0 - w1
            if w2 < -1e-9 or w2 > 1.0:
                continue
            w2 = max(0.0, min(1.0, w2))
            
            verbose = first_combo
            corr = evaluate_weights(per_pt, pitcher_name_map, target_df, [w0, w1, w2], verbose=verbose)
            
            if verbose:
                print(f"  [Debug] First combo weights=[{w0:.2f}, {w1:.2f}, {w2:.2f}], corr={corr}")
                first_combo = False
            
            if pd.notna(corr) and corr > best["corr"]:
                best = {"weights": [w0, w1, w2], "corr": corr}

    if best["weights"] is None:
        print("Warning: no valid weight combination found during coarse search.")
        print(f"  Debug Info:")
        print(f"    per_pt shape: {per_pt.shape}")
        print(f"    per_pt columns: {per_pt.columns.tolist()}")
        print(f"    pitcher_name_map shape: {pitcher_name_map.shape}")
        print(f"    target_df shape: {target_df.shape}")
        best = {"weights": list(EQUAL_STUFFPLUS_WEIGHTS), "corr": np.nan}

    return best


def search_best_weights_constrained(per_pt, pitcher_name_map, target_df, step=0.05, min_weight=0.05):
    """
    Weight search with minimum weight constraints - no component can be below min_weight
    This ensures all Stuff+ components contribute meaningfully
    """
    best = {"weights": None, "corr": -np.inf}
    grid = np.arange(min_weight, 1.0 + 1e-9, step)
    
    # Debug first combination
    first_combo = True
    combo_count = 0
    
    for w0 in grid:
        for w1 in grid:
            w2 = 1.0 - w0 - w1
            if w2 < min_weight or w2 > 1.0:
                continue
                
            combo_count += 1
            verbose = first_combo
            
            corr = evaluate_weights(per_pt, pitcher_name_map, target_df, [w0, w1, w2], verbose=verbose)
            
            if verbose:
                print(f"  [Debug] weights=[{w0:.2f}, {w1:.2f}, {w2:.2f}], corr={corr}")
                first_combo = False
            
            if pd.notna(corr) and corr > best["corr"]:
                best = {"weights": [w0, w1, w2], "corr": corr}

    print(f"  Tested {combo_count} weight combinations with min_weight={min_weight}")
    
    if best["weights"] is None:
        print("Warning: no valid weight combination found with constraints.")
        best = {"weights": [min_weight, min_weight, 1.0 - 2*min_weight], "corr": np.nan}

    return best


def refine_weights(per_pt, pitcher_name_map, target_df, best, step=0.005, min_weight=0.0):
    if best["weights"] is None:
        raise ValueError("Cannot refine weights because no coarse weights were found.")
    w0_best, w1_best, _ = best["weights"]
    best_refined = best.copy()
    deltas = np.arange(-step * 2, step * 2 + 1e-9, step)
    for dw0 in deltas:
        for dw1 in deltas:
            w0 = w0_best + dw0
            w1 = w1_best + dw1
            w2 = 1.0 - w0 - w1
            if w0 < min_weight or w1 < min_weight or w2 < min_weight or w0 > 1 or w1 > 1 or w2 > 1:
                continue
            corr = evaluate_weights(per_pt, pitcher_name_map, target_df, [w0, w1, w2])
            if pd.notna(corr) and corr > best_refined["corr"]:
                best_refined = {"weights": [w0, w1, w2], "corr": corr}
    return best_refined


def main():
    df = load_statcast_2025()

    # Preserve original pitcher name before renaming to pitcher_id
    # Use 'player_name' (which has actual pitcher names) not 'pitcher' (which is an ID)
    if 'pitcher' in df.columns and 'pitcher_id' not in df.columns:
        df['pitcher_name'] = df['player_name']  # Use player_name instead of pitcher
        df = df.rename(columns={'pitcher': 'pitcher_id'})
    if 'pitch_type' not in df.columns and 'pitch_name' in df.columns:
        df = df.rename(columns={'pitch_name': 'pitch_type'})

    missing_cols = {'pitcher_id', 'pitch_type'} - set(df.columns)
    if missing_cols:
        raise RuntimeError(
            f"Statcast dataframe missing required columns: {', '.join(sorted(missing_cols))}. "
            f"Available columns: {', '.join(sorted(df.columns))}"
        )

    calc = StuffPlusCalculator(models_dir="models")
    results = calc.calculate(df)

    per_pt = results["per_pitch_type"]
    pitcher_name_map = build_pitcher_name_map(df)
    war_df = load_pitcher_war(2025)

    print("\n=== Data Summary ===")
    print(f"per_pitch_type shape: {per_pt.shape}")
    print(f"per_pitch_type columns: {per_pt.columns.tolist()}")
    print(f"pitcher_name_map shape: {pitcher_name_map.shape}")
    print(f"war_df shape: {war_df.shape}")
    print(f"Unique pitchers in statcast: {per_pt['pitcher_id'].nunique()}")
    print(f"Unique pitchers in WAR: {len(war_df)}")
    print()

    print("Running constrained grid search for best weights (min_weight=0.10)...")
    best_coarse = search_best_weights_constrained(per_pt, pitcher_name_map, war_df, step=0.1, min_weight=0.10)  # Larger step for speed
    print("Coarse best:", best_coarse)

    if best_coarse["weights"] is None:
        raise RuntimeError(
            "No valid weight combination was found during constrained search. "
            "Check WAR matching and input data alignment."
        )

    print("Refining best weights around the coarse optimum...")
    best_refined = refine_weights(per_pt, pitcher_name_map, war_df, best_coarse, step=0.01, min_weight=0.10)
    print("Refined best:", best_refined)

    best_weights = best_refined["weights"]
    best_corr = best_refined["corr"]

    print("Best Stuff+ weights for 2025 WAR correlation (constrained, min_weight=0.10):")
    print(f"  xwOBA weight:  {best_weights[0]:.4f}")
    print(f"  miss weight:   {best_weights[1]:.4f}")
    print(f"  chase weight:  {best_weights[2]:.4f}")
    print(f"  Pearson r:     {best_corr:.4f}")
    print(f"  R-squared:     {best_corr**2:.4f} ({best_corr**2*100:.2f}%)")

    pitcher_z = compute_pitcher_weighted_z(per_pt, best_weights)
    pitcher_results = pitcher_name_map.merge(pitcher_z, on="pitcher_id", how="left")
    pitcher_results = pitcher_results.merge(
        war_df, left_on="pitcher_norm", right_on="Name_norm", how="left"
    )
    pitcher_results.to_csv(_DEFAULT_DATA_DIR / "stuff_plus_constrained_war_2025_min10.csv", index=False)
    print("Saved constrained results to stuff_plus_constrained_war_2025_min10.csv")


main()