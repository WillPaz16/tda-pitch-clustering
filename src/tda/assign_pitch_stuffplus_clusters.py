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
from pybaseball.statcast import statcast

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'src' / 'stuffplus'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stuff_plus_calculator import StuffPlusCalculator, OPTIMIZED_STUFFPLUS_WEIGHTS
from tda_classifier import load_tda_model, prepare_pitch_features, scaled_cluster_centroids, nearest_cluster

_DEFAULT_MODEL_PATH = str(_ROOT / 'models' / 'tda_mapper_model.pkl')
_DEFAULT_DATA_DIR = _ROOT / 'data'

_CLUSTERING_STUFF_COLS = [
    'release_speed', 'release_pos_x', 'release_pos_z',
    'pfx_x', 'pfx_z', 'spin_axis',
    'release_spin_rate', 'release_extension'
]


def prepare_pitch_features_for_clustering(df):
    """
    Prepare raw Statcast data for TDA cluster assignment.
    Mirrors movement and position for LHP and calculates spin axis clock.
    """
    return prepare_pitch_features(df, _CLUSTERING_STUFF_COLS + ['p_throws'])


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
    X_clusters_scaled, cluster_summary = scaled_cluster_centroids(model_components, input_columns)

    # Assign each pitch to nearest cluster
    classifications = []
    for i, pitch_scaled in enumerate(X_new_scaled):
        closest_cluster_idx, distance = nearest_cluster(pitch_scaled, X_clusters_scaled)
        closest_cluster = cluster_summary.iloc[closest_cluster_idx]

        classifications.append({
            'original_index': valid_idx[i],
            'cluster_id': str(closest_cluster['cluster']),
            'cluster_size': int(closest_cluster['size']),
            'distance_to_cluster': distance,
            'dominant_pitch_type': closest_cluster['most_common_pitch_type'],
        })

    return pd.DataFrame(classifications)


def calculate_weighted_stuffplus(df_result,
                                  xwoba_weight=OPTIMIZED_STUFFPLUS_WEIGHTS['xwoba'],
                                  miss_weight=OPTIMIZED_STUFFPLUS_WEIGHTS['miss'],
                                  chase_weight=OPTIMIZED_STUFFPLUS_WEIGHTS['chase']):
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
    print(f"Calculating per-pitch Stuff+ with optimized weights {tuple(OPTIMIZED_STUFFPLUS_WEIGHTS.values())}...")
    stuff_plus_values, z_xw, z_miss, z_chase = calculate_weighted_stuffplus(df_result)
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
