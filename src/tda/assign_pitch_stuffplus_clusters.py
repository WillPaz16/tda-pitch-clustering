"""
Assign Stuff+ values and TDA cluster IDs to individual pitches.

Combines:
- Per-pitch predictions from StuffPlusCalculator
- Optimized Stuff+ weights (xwOBA=0.72, miss=0.11, chase=0.17)
- TDA cluster assignments via nearest centroid to cluster features
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from pybaseball.statcast import statcast

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'src' / 'stuffplus'))
from stuff_plus_calculator import StuffPlusCalculator

_DEFAULT_MODEL_PATH = str(_ROOT / 'models' / 'tda_mapper_model.pkl')
_DEFAULT_DATA_DIR = _ROOT / 'data'


def load_tda_model(model_path=_DEFAULT_MODEL_PATH):
    """Load the fitted TDA model components."""
    with open(model_path, 'rb') as f:
        model_components = pickle.load(f)
    return model_components


def prepare_pitch_features_for_clustering(df):
    """
    Prepare raw Statcast data for TDA cluster assignment.
    Mirrors movement and position for LHP and calculates spin axis clock.
    """
    # Define standard stuff columns
    stuff_cols = [
        'release_speed', 'release_pos_x', 'release_pos_z',
        'pfx_x', 'pfx_z', 'spin_axis',
        'release_spin_rate', 'release_extension'
    ]
    
    stuff_df = df[stuff_cols + ['p_throws']].copy()
    
    # Mirror movement and position for LHP
    stuff_df.loc[stuff_df['p_throws'] == 'L', ['pfx_x', 'release_pos_x']] *= -1
    
    # Mirror spin axis to match RHP frame
    stuff_df.loc[stuff_df['p_throws'] == 'L', 'spin_axis'] = (
        360 - stuff_df.loc[stuff_df['p_throws'] == 'L', 'spin_axis']
    ) % 360
    
    # Calculate spin axis clock
    def degrees_to_clock(degrees):
        return (((degrees + 15) % 360) // 30 + 1).astype('Int64')
    
    stuff_df['spin_axis_clock'] = degrees_to_clock(stuff_df['spin_axis'])
    
    return stuff_df


def assign_tda_clusters(pitch_features_df, model_components, input_columns):
    """
    Assign each pitch to nearest TDA cluster centroid.
    
    Args:
        pitch_features_df: DataFrame with features for TDA clustering
        model_components: Loaded TDA model pickle
        input_columns: Feature columns to use
    
    Returns:
        DataFrame with cluster assignments
    """
    scaler = model_components['scaler']
    cluster_summary = model_components['cluster_summary']
    
    # Get input data and drop NaN
    X_new = pitch_features_df[input_columns].copy()
    initial_count = len(X_new)
    X_new_clean = X_new.dropna()
    valid_idx = X_new_clean.index
    
    if len(X_new_clean) == 0:
        print(f"  Warning: No valid pitches with all features present")
        return pd.DataFrame()
    
    print(f"  Using {len(X_new_clean)} of {initial_count} pitches with complete features")
    
    # Scale features - ensure all values are numeric
    try:
        X_new_values = X_new_clean.values.astype(np.float64)
        X_new_scaled = scaler.transform(X_new_values)
    except Exception as e:
        print(f"  Error scaling features: {e}")
        return pd.DataFrame()
    
    # Get cluster centroids in original and scaled space
    cluster_summary_original = cluster_summary.copy()
    cluster_summary_original['pfx_x'] = cluster_summary_original['HB'] / -12
    cluster_summary_original['pfx_z'] = cluster_summary_original['IVB'] / 12
    X_clusters = cluster_summary_original[input_columns].values.astype(np.float64)
    X_clusters_scaled = scaler.transform(X_clusters)
    
    # Assign each pitch to nearest cluster
    classifications = []
    for i, pitch_scaled in enumerate(X_new_scaled):
        distances = np.linalg.norm(X_clusters_scaled - pitch_scaled, axis=1)
        closest_cluster_idx = np.argmin(distances)
        closest_cluster = cluster_summary.iloc[closest_cluster_idx]
        
        classifications.append({
            'original_index': valid_idx[i],
            'cluster_id': str(closest_cluster['cluster']),
            'cluster_size': int(closest_cluster['size']),
            'distance_to_cluster': float(distances[closest_cluster_idx]),
            'dominant_pitch_type': closest_cluster['most_common_pitch_type'],
        })
    
    return pd.DataFrame(classifications)


def calculate_weighted_stuffplus(df_result, xwoba_weight=0.72, miss_weight=0.11, chase_weight=0.17):
    """
    Calculate per-pitch weighted Stuff+ using optimized weights.
    
    Weights should sum to 1.0 and represent the contribution of each component.
    """
    # Get z-scores for each component
    df_work = df_result[['pred_xw', 'pred_miss', 'pred_chase']].copy()
    
    # Calculate mean and std per component (league-wide from this dataset)
    mu_xw = df_work['pred_xw'].mean()
    sd_xw = df_work['pred_xw'].std()
    mu_miss = df_work['pred_miss'].mean()
    sd_miss = df_work['pred_miss'].std()
    mu_chase = df_work['pred_chase'].mean()
    sd_chase = df_work['pred_chase'].std()
    
    # Compute z-scores
    z_xw = (df_work['pred_xw'] - mu_xw) / (sd_xw + 1e-9)
    z_miss = (df_work['pred_miss'] - mu_miss) / (sd_miss + 1e-9)
    z_chase = (df_work['pred_chase'] - mu_chase) / (sd_chase + 1e-9)
    
    # Weighted combination
    weighted_z = xwoba_weight * z_xw + miss_weight * z_miss + chase_weight * z_chase
    
    # Scale to 100 mean, 10 std (standard Stuff+ scale)
    stuff_plus = 100 + (10 * weighted_z)
    stuff_plus = np.round(stuff_plus, 2)
    
    return stuff_plus, z_xw, z_miss, z_chase


def main():
    # Load Statcast data
    start_date = "2025-03-28"
    end_date = "2025-04-04"
    print(f"Loading Statcast data: {start_date} → {end_date}")
    df = statcast(start_date, end_date)
    print(f"Loaded {len(df)} pitches\n")
    
    # Rename columns if necessary to match StuffPlusCalculator expectations
    if 'pitcher' in df.columns and 'pitcher_id' not in df.columns:
        df = df.rename(columns={'pitcher': 'pitcher_id'})
    if 'pitch_name' in df.columns and 'pitch_type' not in df.columns:
        df = df.rename(columns={'pitch_name': 'pitch_type'})
    
    # Calculate per-pitch predictions with StuffPlusCalculator
    print("Calculating per-pitch predictions (xwOBA, miss, chase)...")
    calc = StuffPlusCalculator(models_dir="models")
    results = calc.calculate(df)
    df_result = results["pitch_level"]  # Full pitch-level results
    print(f"Got predictions for {len(df_result)} pitches\n")
    
    # Calculate per-pitch Stuff+ using optimized weights
    print("Calculating per-pitch Stuff+ with optimized weights (0.72, 0.11, 0.17)...")
    stuff_plus_values, z_xw, z_miss, z_chase = calculate_weighted_stuffplus(
        df_result,
        xwoba_weight=0.72,
        miss_weight=0.11,
        chase_weight=0.17
    )
    df_result['stuff_plus'] = stuff_plus_values
    df_result['z_xw'] = z_xw
    df_result['z_miss'] = z_miss
    df_result['z_chase'] = z_chase
    print(f"Stuff+ range: {stuff_plus_values.min():.2f} to {stuff_plus_values.max():.2f}\n")
    
    # Load TDA model and prepare features
    print("Loading TDA model and assigning clusters...")
    model_components = load_tda_model(_DEFAULT_MODEL_PATH)
    
    # Prepare features for clustering
    pitch_features = prepare_pitch_features_for_clustering(df)
    
    # Get input columns from TDA model
    stuff_columns = model_components['stuff_columns']
    input_columns = [col for col in stuff_columns if col != 'spin_axis_clock']
    
    # Assign clusters
    cluster_assignments = assign_tda_clusters(pitch_features, model_components, input_columns)
    print(f"Assigned {len(cluster_assignments)} pitches to {cluster_assignments['cluster_id'].nunique()} clusters\n")
    
    # Merge cluster assignments back to pitch data
    # Use the valid indices from cluster_assignments
    pitch_with_clusters = df_result.iloc[cluster_assignments['original_index'].values].copy()
    pitch_with_clusters['cluster_id'] = cluster_assignments['cluster_id'].values
    pitch_with_clusters['distance_to_cluster'] = cluster_assignments['distance_to_cluster'].values
    pitch_with_clusters['dominant_pitch_type'] = cluster_assignments['dominant_pitch_type'].values
    
    # Select key columns for output
    output_columns = [
        'pitcher_id', 'pitcher', 'batter_id', 'batter', 'pitch_type',
        'release_speed', 'spin_axis', 'release_spin_rate',
        'pred_xw', 'pred_miss', 'pred_chase',
        'z_xw', 'z_miss', 'z_chase',
        'stuff_plus',
        'cluster_id', 'dominant_pitch_type', 'distance_to_cluster'
    ]
    
    # Filter to available columns
    output_columns = [col for col in output_columns if col in pitch_with_clusters.columns]
    output_df = pitch_with_clusters[output_columns].copy()
    
    # Save to CSV
    output_file = str(_DEFAULT_DATA_DIR / "pitch_stuffplus_clusters.csv")
    output_df.to_csv(output_file, index=False)
    print(f"Saved {len(output_df)} pitches with Stuff+ and cluster assignments to {output_file}")
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total pitches: {len(output_df)}")
    print(f"Stuff+ mean: {output_df['stuff_plus'].mean():.2f}, std: {output_df['stuff_plus'].std():.2f}")
    print(f"Number of clusters: {output_df['cluster_id'].nunique()}")
    print(f"\nCluster distribution:")
    print(output_df['cluster_id'].value_counts().sort_index())


if __name__ == "__main__":
    main()
