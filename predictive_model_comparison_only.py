#!/usr/bin/env python3
"""
Standalone Predictive Model Comparison

This script performs only the statcast-vs-cluster predictive comparison using the
same feature engineering and modeling process as predictive_model_comparison.py.
"""

import glob
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

TARGETS = ['ERA', 'SO9', 'SO/W']
STATCAST_TYPES = ['FF', 'SI', 'FC', 'CU', 'SL', 'CH', 'ST']


def load_all_data():
    pitch_files = glob.glob('classified_pitches_*.csv')
    dfs = []
    for file in pitch_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception:
            continue
    pitch_data = pd.concat(dfs, ignore_index=True)

    from pybaseball import pitching_stats_bref
    bref_df = pitching_stats_bref(2025)
    bref_df['pitcher_id'] = bref_df['mlbID'].astype(str)
    bref_df = bref_df[bref_df['IP'] >= 50]

    stuff_df = pd.read_csv('models/stuff_plus_pitcher_level.csv')
    stuff_df['pitcher_id'] = stuff_df['pitcher'].astype(str)

    return pitch_data, bref_df, stuff_df


def create_statcast_type_features(pitch_data, bref_df):
    pitch_data['pitcher_id'] = pitch_data['pitcher'].astype(str)
    type_features = []
    for pitcher_id, pitcher_pitches in pitch_data.groupby('pitcher_id'):
        if pitcher_id not in bref_df['pitcher_id'].values:
            continue
        pitch_counts = pitcher_pitches['pitch_type'].value_counts()
        total_pitches = len(pitcher_pitches)
        features = {'pitcher_id': pitcher_id}
        for ptype in STATCAST_TYPES:
            pct = (pitch_counts.get(ptype, 0) / total_pitches) * 100
            features[f'pct_{ptype}'] = pct
        for ptype in ['FF', 'SI', 'CU', 'SL', 'CH']:
            velocities = pitcher_pitches[pitcher_pitches['pitch_type'] == ptype]['release_speed']
            if len(velocities) > 0:
                features[f'avg_velo_{ptype}'] = velocities.mean()
        type_features.append(features)
    return pd.DataFrame(type_features).fillna(0)


def create_cluster_features(pitch_data, bref_df):
    pitch_data['pitcher_id'] = pitch_data['pitcher'].astype(str)
    try:
        stuff_df = pd.read_csv('models/stuff_plus_pitcher_level.csv')
        stuff_df['pitcher_id'] = stuff_df['pitcher'].astype(str)
        stuff_dict = dict(zip(stuff_df['pitcher_id'], stuff_df['stuff_plus_overall']))
    except Exception:
        stuff_dict = {}

    cluster_features = []
    for pitcher_id, pitcher_pitches in pitch_data.groupby('pitcher_id'):
        if pitcher_id not in bref_df['pitcher_id'].values:
            continue
        features = {'pitcher_id': pitcher_id}
        cluster_counts = pitcher_pitches['cluster_id'].value_counts()
        total_pitches = len(pitcher_pitches)
        top_clusters = cluster_counts.nlargest(8).index.tolist()
        for i, cluster in enumerate(top_clusters):
            pct = (cluster_counts.get(cluster, 0) / total_pitches) * 100
            features[f'cluster_{i}_pct'] = pct
        cluster_dist = cluster_counts / total_pitches
        entropy = -np.sum(cluster_dist * np.log(cluster_dist + 1e-10))
        features['cluster_diversity'] = entropy
        features['num_clusters'] = pitcher_pitches['cluster_id'].nunique()
        features['cluster_homogeneity'] = 1.0 / (1.0 + pitcher_pitches['distance_to_cluster'].mean())
        for i, cluster in enumerate(top_clusters[:3]):
            cluster_velos = pitcher_pitches[pitcher_pitches['cluster_id'] == cluster]['release_speed']
            if len(cluster_velos) > 0:
                features[f'cluster_{i}_avg_velo'] = cluster_velos.mean()
                features[f'cluster_{i}_velo_std'] = cluster_velos.std()
        all_pitches_clusters = pitcher_pitches['distance_to_cluster'].values
        features['cluster_consistency'] = np.std(all_pitches_clusters)
        features['cluster_tightness'] = 1.0 / (1.0 + features['cluster_consistency'])
        if pitcher_id in stuff_dict:
            features['stuff_plus'] = stuff_dict[pitcher_id]
        cluster_features.append(features)
    return pd.DataFrame(cluster_features).fillna(0)


def build_and_compare_models(type_features, cluster_features, bref_df, target_metric):
    success_data = bref_df[['pitcher_id', target_metric]].dropna()
    type_data = pd.merge(type_features, success_data, on='pitcher_id', how='inner')
    cluster_data = pd.merge(cluster_features, success_data, on='pitcher_id', how='inner')
    if len(type_data) < 10 or len(cluster_data) < 10:
        print(f"Insufficient data for {target_metric}")
        return None
    X_type = type_data.drop(['pitcher_id', target_metric], axis=1)
    X_cluster = cluster_data.drop(['pitcher_id', target_metric], axis=1)
    y = type_data[target_metric].values
    scaler_type = StandardScaler()
    scaler_cluster = StandardScaler()
    X_type_scaled = scaler_type.fit_transform(X_type)
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
    model_type_lr = LinearRegression()
    model_cluster_lr = LinearRegression()
    model_type_lr.fit(X_type_scaled, y)
    model_cluster_lr.fit(X_cluster_scaled, y)
    y_pred_type_lr = model_type_lr.predict(X_type_scaled)
    y_pred_cluster_lr = model_cluster_lr.predict(X_cluster_scaled)
    model_type_rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
    model_cluster_rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
    model_type_rf.fit(X_type, y)
    model_cluster_rf.fit(X_cluster, y)
    y_pred_type_rf = model_type_rf.predict(X_type)
    y_pred_cluster_rf = model_cluster_rf.predict(X_cluster)
    results = {
        'metric': target_metric,
        'n_samples': len(type_data),
        'r2_type_lr': r2_score(y, y_pred_type_lr),
        'r2_cluster_lr': r2_score(y, y_pred_cluster_lr),
        'r2_type_rf': r2_score(y, y_pred_type_rf),
        'r2_cluster_rf': r2_score(y, y_pred_cluster_rf),
        'mae_type_lr': mean_absolute_error(y, y_pred_type_lr),
        'mae_cluster_lr': mean_absolute_error(y, y_pred_cluster_lr),
        'mae_type_rf': mean_absolute_error(y, y_pred_type_rf),
        'mae_cluster_rf': mean_absolute_error(y, y_pred_cluster_rf),
    }
    return results


def main():
    pitch_data, bref_df, stuff_df = load_all_data()
    type_features = create_statcast_type_features(pitch_data, bref_df)
    cluster_features = create_cluster_features(pitch_data, bref_df)
    print(f"Loaded {len(pitch_data)} pitches for {pitch_data['pitcher'].nunique()} pitchers")
    print(f"Generated {len(type_features)} pitchers with {len(type_features.columns)} statcast features")
    print(f"Generated {len(cluster_features)} pitchers with {len(cluster_features.columns)} cluster features")
    all_results = []
    for target in TARGETS:
        print(f"\nPredicting {target}...")
        results = build_and_compare_models(type_features, cluster_features, bref_df, target)
        if results is None:
            continue
        all_results.append(results)
        print(f"  Linear Regression R²: Statcast={results['r2_type_lr']:.4f}, Cluster={results['r2_cluster_lr']:.4f}")
        print(f"  Random Forest R²:   Statcast={results['r2_type_rf']:.4f}, Cluster={results['r2_cluster_rf']:.4f}")
    if not all_results:
        return
    results_df = pd.DataFrame(all_results)
    print("\nSummary")
    print(results_df[['metric', 'r2_type_lr', 'r2_cluster_lr', 'r2_type_rf', 'r2_cluster_rf']].to_string(index=False))
    avg_type_rf = results_df['r2_type_rf'].mean()
    avg_cluster_rf = results_df['r2_cluster_rf'].mean()
    print(f"\nAverage RF R²: Statcast={avg_type_rf:.4f}, Cluster={avg_cluster_rf:.4f}, Delta={avg_cluster_rf-avg_type_rf:+.4f}")
    cluster_wins = (results_df['r2_cluster_rf'] > results_df['r2_type_rf']).sum()
    print(f"Cluster-based RF wins {cluster_wins}/{len(results_df)} targets")

if __name__ == '__main__':
    main()
