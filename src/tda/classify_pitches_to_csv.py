"""
TDA Pitch Classification to CSV
Classifies pitches from Statcast data into TDA-discovered clusters and exports to CSV.
"""

import io
import requests
import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_PATH = str(_ROOT / 'models' / 'tda_mapper_model.pkl')
_DEFAULT_DATA_DIR = _ROOT / 'data'


def load_model(model_path=_DEFAULT_MODEL_PATH):
    """Load the fitted TDA model components."""
    with open(model_path, 'rb') as f:
        model_components = pickle.load(f)
    return model_components


def prepare_pitch_data(df, stuff_cols):
    """
    Prepare raw Statcast data for classification.
    Mirrors LHP data to RHP frame and calculates spin axis clock.
    """
    stuff_df = df[stuff_cols].copy()
    
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


def classify_pitches(pitch_data, model_components):
    """Classify individual pitches into clusters."""
    scaler = model_components['scaler']
    cluster_summary = model_components['cluster_summary']
    stuff_columns = model_components['stuff_columns']
    
    # Use only base columns for scaling
    input_columns = [col for col in stuff_columns if col != 'spin_axis_clock']
    X_new = pitch_data[input_columns].copy()
    
    # Handle missing values
    initial_count = len(X_new)
    X_new = X_new.dropna()
    dropped_count = initial_count - len(X_new)
    if dropped_count > 0:
        print(f"  Note: Dropped {dropped_count} pitches with missing data")
    
    if len(X_new) == 0:
        print("  Error: No valid pitch data to classify")
        return pd.DataFrame()
    
    X_new_values = X_new.values
    X_new_scaled = scaler.transform(X_new_values)
    
    # Get cluster centroids
    cluster_summary_original = cluster_summary.copy()
    cluster_summary_original['pfx_x'] = cluster_summary_original['HB'] / -12
    cluster_summary_original['pfx_z'] = cluster_summary_original['IVB'] / 12
    X_clusters = cluster_summary_original[input_columns].values
    X_clusters_scaled = scaler.transform(X_clusters)
    
    classifications = []
    
    for i, pitch_scaled in enumerate(X_new_scaled):
        distances = np.linalg.norm(X_clusters_scaled - pitch_scaled, axis=1)
        closest_cluster_idx = np.argmin(distances)
        closest_cluster = cluster_summary.iloc[closest_cluster_idx]
        
        classifications.append({
            'cluster_id': closest_cluster['cluster'],
            'cluster_size': int(closest_cluster['size']),
            'distance_to_cluster': float(distances[closest_cluster_idx]),
            'dominant_pitch_type_in_cluster': closest_cluster['most_common_pitch_type'],
            'cluster_release_speed': float(closest_cluster['release_speed']),
            'cluster_HB': float(closest_cluster['HB']),
            'cluster_IVB': float(closest_cluster['IVB'])
        })
    
    return pd.DataFrame(classifications), X_new.index


def process_statcast_data(start_date, end_date, model_components, output_file=None):
    """
    Query Statcast data and classify pitches.
    
    Parameters:
    -----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    model_components : dict
        Loaded TDA model components
    output_file : str, optional
        Path to output CSV file. If None, uses default naming.
    
    Returns:
    --------
    pd.DataFrame : Classified pitch data
    """
    
    print(f"Querying Statcast data from {start_date} to {end_date}...")
    csv_url = (
        "https://baseballsavant.mlb.com/statcast_search/csv?"
        "all=true&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&"
        "hfGT=R%7CPO%7CS%7C=&hfSea=&hfSit=&player_type=pitcher&hfOuts=&opponent=&"
        "pitcher_throws=&batter_stands=&hfSA=&game_date_gt={}&game_date_lt={}&"
        "team=&position=&hfRO=&home_road=&hfFlag=&metric_1=&hfInn=&min_pitches=0&"
        "min_results=0&group_by=name&sort_col=pitches&player_event_sort=h_launch_speed&"
        "sort_order=desc&min_abs=0&type=details&"
    ).format(start_date, end_date)

    response = requests.get(csv_url, timeout=60)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    print(f"Retrieved {len(df)} pitch records")
    
    # Define columns to extract
    stuff_cols = [
        'pitcher', 'pitch_type', 'p_throws',
        'release_speed', 'pfx_x', 'pfx_z',
        'release_spin_rate', 'spin_axis',
        'release_extension', 'release_pos_x', 'release_pos_z'
    ]
    
    # Add game context columns if available
    context_cols = []
    for col in ['game_date', 'home_team', 'away_team', 'inning', 'outs_when_up', 
                'balls', 'strikes', 'batter', 'pitcher_1', 'des', 'type']:
        if col in df.columns:
            context_cols.append(col)
    
    # Check which columns exist
    available_cols = [col for col in stuff_cols + context_cols if col in df.columns]
    df_subset = df[available_cols].copy()
    
    print(f"Preparing {len(df_subset)} pitches for classification...")
    
    # Prepare pitch data
    prepared_df = prepare_pitch_data(df_subset, 
                                      [col for col in stuff_cols if col in df_subset.columns])
    
    # Classify pitches
    print("Classifying pitches into clusters...")
    classifications, valid_indices = classify_pitches(prepared_df, model_components)
    
    # Filter original data to match classified pitches (exclude rows with NaN)
    df_classified = df_subset.iloc[valid_indices].reset_index(drop=True)
    prepared_classified = prepared_df.iloc[valid_indices].reset_index(drop=True)
    classifications = classifications.reset_index(drop=True)
    
    # Combine results
    result_df = pd.concat([df_classified, prepared_classified, classifications], axis=1)
    
    # Reorder columns for clarity
    output_cols = [
        'game_date', 'pitcher', 'pitcher_1', 'batter', 'home_team', 'away_team',
        'inning', 'outs_when_up', 'balls', 'strikes',
        'pitch_type', 'p_throws', 'type', 'des',
        'release_speed', 'pfx_x', 'pfx_z', 'release_spin_rate',
        'spin_axis', 'spin_axis_clock', 'release_extension',
        'release_pos_x', 'release_pos_z',
        'cluster_id', 'cluster_size', 'distance_to_cluster',
        'dominant_pitch_type_in_cluster', 'cluster_release_speed',
        'cluster_HB', 'cluster_IVB'
    ]
    
    # Keep only columns that exist
    output_cols = [col for col in output_cols if col in result_df.columns]
    result_df = result_df[output_cols]
    
    # Save to CSV
    if output_file is None:
        output_file = str(_DEFAULT_DATA_DIR / f"classified_pitches_{start_date}_{end_date}.csv")
    
    print(f"\nSaving {len(result_df)} classified pitches to {output_file}...")
    result_df.to_csv(output_file, index=False)
    print(f"Successfully saved to {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("CLASSIFICATION SUMMARY")
    print("="*60)
    print(f"Total pitches classified: {len(result_df)}")
    print(f"\nCluster distribution:")
    print(result_df['cluster_id'].value_counts().sort_index())
    
    # Get pitch type from original df_subset if available
    if 'pitch_type' in df_classified.columns:
        print(f"\nPitch types in data:")
        print(df_classified['pitch_type'].value_counts())
    
    print(f"\nAverage distance to assigned cluster: {result_df['distance_to_cluster'].mean():.4f}")
    print("="*60 + "\n")
    
    return result_df


def main():
    """Main entry point for script."""
    parser = argparse.ArgumentParser(
        description='Classify pitches from Statcast data using TDA model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
SAMPLE USAGE:
  # Default: classify Sept 15, 2025
  python classify_pitches_to_csv.py
  
  # Classify single day (Sept 15, 2025)
  python classify_pitches_to_csv.py 2025-09-15 2025-09-15
  
  # Classify entire week (Sept 10-16, 2025)
  python classify_pitches_to_csv.py 2025-09-10 2025-09-16
  
  # Classify month (all of September 2025)
  python classify_pitches_to_csv.py 2025-09-01 2025-09-30
  
  # Classify with custom output file
  python classify_pitches_to_csv.py 2025-09-15 2025-09-15 -o sept_15_classified.csv
  
  # Classify specific date range with output file
  python classify_pitches_to_csv.py 2025-09-01 2025-09-05 -o first_week_sept.csv
  
  # Use custom model path
  python classify_pitches_to_csv.py 2025-09-15 2025-09-15 -m /path/to/tda_mapper_model.pkl
        '''
    )
    
    parser.add_argument('start_date', nargs='?', default='2025-09-15',
                       help='Start date in YYYY-MM-DD format (default: 2025-09-15)')
    parser.add_argument('end_date', nargs='?', default='2025-09-15',
                       help='End date in YYYY-MM-DD format (default: 2025-09-15)')
    parser.add_argument('-o', '--output', dest='output_file',
                       help='Output CSV file path (optional)')
    parser.add_argument('-m', '--model', dest='model_path',
                       default=_DEFAULT_MODEL_PATH,
                       help=f'Path to TDA model file (default: {_DEFAULT_MODEL_PATH})')
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        start = datetime.strptime(args.start_date, '%Y-%m-%d')
        end = datetime.strptime(args.end_date, '%Y-%m-%d')
        if start > end:
            parser.error("Start date must be before end date")
    except ValueError:
        parser.error("Dates must be in YYYY-MM-DD format")
    
    try:
        # Load model
        print(f"Loading model from {args.model_path}...")
        model_components = load_model(args.model_path)
        print("Model loaded successfully\n")
        
        # Process data
        result_df = process_statcast_data(
            args.start_date, 
            args.end_date, 
            model_components,
            args.output_file
        )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the model file exists in the current directory")
        exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        raise


if __name__ == '__main__':
    main()
