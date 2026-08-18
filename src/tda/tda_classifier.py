"""
Shared TDA model loading and nearest-centroid pitch classification logic.

Used by classify_pitches_to_csv.py and assign_pitch_stuffplus_clusters.py,
which both classify individual pitches against the fitted Mapper cluster
centroids and previously duplicated this logic independently.
"""

import pickle
import numpy as np


def load_tda_model(model_path):
    """Load the fitted TDA model components."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def degrees_to_clock(degrees):
    return (((degrees + 15) % 360) // 30 + 1).astype('Int64')


def prepare_pitch_features(df, feature_cols):
    """
    Prepare raw Statcast data for TDA classification.
    Mirrors LHP movement/position/spin_axis to the RHP frame and adds
    spin_axis_clock. feature_cols must include 'p_throws' and 'spin_axis'.
    """
    stuff_df = df[feature_cols].copy()

    stuff_df.loc[stuff_df['p_throws'] == 'L', ['pfx_x', 'release_pos_x']] *= -1

    stuff_df.loc[stuff_df['p_throws'] == 'L', 'spin_axis'] = (
        360 - stuff_df.loc[stuff_df['p_throws'] == 'L', 'spin_axis']
    ) % 360

    stuff_df['spin_axis_clock'] = degrees_to_clock(stuff_df['spin_axis'])

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
