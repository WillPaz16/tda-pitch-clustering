"""
Shared TDA model loading and nearest-centroid pitch classification logic.

Used by classify_pitches_to_csv.py and assign_pitch_stuffplus_clusters.py,
which both classify individual pitches against the fitted Mapper cluster
centroids and previously duplicated this logic independently.
"""

import pickle
import re
import numpy as np
import pybaseball
from kmapper.cover import Cover
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler


def load_tda_model(model_path):
    """Load the fitted TDA model components."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def fetch_savant_csv(start_date, end_date):
    """
    Fetch raw Statcast data for a date range.
    A single Savant CSV export silently caps at 25,000 rows (~3 days of
    games), so multi-day ranges were truncated. pybaseball splits the range
    into small requests; its old postprocessing crash was fixed by 2.2.7.
    """
    return pybaseball.statcast(start_dt=start_date, end_dt=end_date, verbose=False)


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


def build_cover_index(model_components):
    """
    Rebuild the fitted Mapper cover so new pitches can be placed in it,
    reproducing KeplerMapper's internal steps (verified against the kmapper
    source, see docs/METHODOLOGY_REVIEW.md item 3):
      - the real lens is MinMaxScaler(PCA(X_scaled)): mapper.fit_transform()
        applies a default MinMaxScaler to the projection
      - node labels "cube{j}" index cubes AFTER Cover.transform() drops empty
        ones, not the raw index into Cover.centers_
      - cluster membership inside a cube is DBSCAN density (eps-ball), not a
        centroid rule
    """
    scaler, pca, graph = model_components['scaler'], model_components['pca'], model_components['graph']
    meta = graph['meta_data']
    eps = float(re.search(r'eps=([\d.]+)', meta['clusterer']).group(1))

    X_train = scaler.transform(model_components['original_data'][model_components['stuff_columns']].values.astype(np.float64))
    raw_lens = pca.transform(X_train)
    lens_scaler = MinMaxScaler().fit(raw_lens)
    lens = lens_scaler.transform(raw_lens)

    lens_with_ids = np.c_[np.arange(len(lens)), lens]
    cover = Cover(n_cubes=meta['n_cubes'], perc_overlap=meta['perc_overlap'])
    cover.fit(lens_with_ids)

    compacted = {}
    for i, center in enumerate(cover.centers_):
        if len(cover.transform_single(lens_with_ids, center, i)):
            compacted[i] = len(compacted)
    raw_by_cube = {j: i for i, j in compacted.items()}

    nodes = []  # (node_id, raw center index, member points in scaled space)
    for node_id, members in graph['nodes'].items():
        cube = int(node_id.split('_cluster')[0].replace('cube', ''))
        nodes.append((node_id, raw_by_cube[cube], X_train[members]))

    return {
        'scaler': scaler, 'pca': pca, 'lens_scaler': lens_scaler, 'eps': eps,
        'centers': np.array(cover.centers_), 'radius': cover.radius_, 'nodes': nodes,
    }


def member_clusters(X_scaled, cover_index):
    """
    Return, for each scaled pitch, the sorted list of Mapper nodes it belongs
    to: every cube its lens coordinate falls in, then every DBSCAN cluster in
    those cubes with a member within eps. Empty list = outside every cluster
    (DBSCAN noise). Reproduces the real training-set membership for 97.2% of
    points; mismatches only ever add an extra node (border points), never drop
    a true one -- noise points aren't stored in the model, so exact DBSCAN
    core/border status can't be recovered.
    """
    ci = cover_index
    lens = ci['lens_scaler'].transform(ci['pca'].transform(X_scaled))
    in_cube = np.all(np.abs(lens[:, None, :] - ci['centers'][None, :, :]) <= ci['radius'], axis=2)

    memberships = [[] for _ in range(len(X_scaled))]
    for node_id, raw_cube, members in ci['nodes']:
        candidates = np.flatnonzero(in_cube[:, raw_cube])
        if len(candidates) == 0:
            continue
        near = cdist(X_scaled[candidates], members).min(axis=1) <= ci['eps']
        for idx in candidates[near]:
            memberships[idx].append(node_id)
    return [sorted(m) for m in memberships]


def mapper_primary_labels(X_scaled, memberships, model_components):
    """
    Mapper-consistent single label per pitch, for stats that need one group
    per observation: the nearest centroid *among the pitch's own member
    clusters*, so it never overrides Mapper. None for DBSCAN noise (no
    membership) -- those pitches are excluded from group tests.
    """
    X_clusters_scaled, cs = scaled_cluster_centroids(model_components, model_components['stuff_columns'])
    pos = {cid: i for i, cid in enumerate(cs['cluster'])}
    labels = []
    for x, members in zip(X_scaled, memberships):
        if not members:
            labels.append(None)
            continue
        d = np.linalg.norm(X_clusters_scaled[[pos[m] for m in members]] - x, axis=1)
        labels.append(members[int(np.argmin(d))])
    return labels
