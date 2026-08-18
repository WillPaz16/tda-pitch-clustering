"""
Shared TDA model loading and nearest-centroid pitch classification logic.

Used by classify_pitches_to_csv.py and assign_pitch_stuffplus_clusters.py,
which both classify individual pitches against the fitted Mapper cluster
centroids and previously duplicated this logic independently.
"""

import io
import pickle
import numpy as np
import pandas as pd
import requests


def load_tda_model(model_path):
    """Load the fitted TDA model components."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def fetch_savant_csv(start_date, end_date):
    """
    Fetch raw Statcast data via the Baseball Savant CSV export directly.
    Avoids the pybaseball.statcast() postprocessing bug (duplicate-column
    crash) present in this environment -- see docs/METHODOLOGY_REVIEW.md.
    """
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


def prepare_pitch_features(df, feature_cols):
    """
    Prepare raw Statcast data for TDA classification.
    Mirrors LHP movement/position/spin_axis to the RHP frame and encodes
    spin_axis as (spin_cos, spin_sin) -- spin_axis is a circular quantity
    (359deg and 1deg are nearly identical directions), so it must be
    encoded this way rather than fed to Euclidean distance as a raw
    degree value. See docs/METHODOLOGY_REVIEW.md item 2.
    feature_cols must include 'p_throws' and 'spin_axis'.
    """
    stuff_df = df[feature_cols].copy()

    stuff_df.loc[stuff_df['p_throws'] == 'L', ['pfx_x', 'release_pos_x']] *= -1

    stuff_df.loc[stuff_df['p_throws'] == 'L', 'spin_axis'] = (
        360 - stuff_df.loc[stuff_df['p_throws'] == 'L', 'spin_axis']
    ) % 360

    spin_rad = np.deg2rad(stuff_df['spin_axis'])
    stuff_df['spin_cos'] = np.cos(spin_rad)
    stuff_df['spin_sin'] = np.sin(spin_rad)

    return stuff_df


def scaled_cluster_centroids(model_components, input_columns):
    """Return (scaled centroid matrix, original cluster_summary) for the model."""
    scaler = model_components['scaler']
    cluster_summary = model_components['cluster_summary']

    cluster_summary_original = cluster_summary.copy()
    cluster_summary_original['pfx_x'] = cluster_summary_original['HB'] / -12
    cluster_summary_original['pfx_z'] = cluster_summary_original['IVB'] / 12

    X_clusters = cluster_summary_original[input_columns].values.astype(np.float64)
    X_clusters_scaled = scaler.transform(X_clusters)

    return X_clusters_scaled, cluster_summary


def nearest_cluster(pitch_scaled, X_clusters_scaled):
    """Return (closest_cluster_idx, distance) for a single scaled pitch."""
    distances = np.linalg.norm(X_clusters_scaled - pitch_scaled, axis=1)
    idx = np.argmin(distances)
    return idx, float(distances[idx])
