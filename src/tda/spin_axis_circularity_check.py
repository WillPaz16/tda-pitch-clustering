"""
Spin-Axis Circularity Check

Follow-up to docs/METHODOLOGY_REVIEW.md's objection: spin_axis is a
circular quantity (359 degrees and 1 degree are nearly identical spin
directions) but is fed into StandardScaler and Euclidean distance
exactly like release_speed, a genuinely linear quantity. This checks
whether that theoretical objection actually changes anything in the
fitted model, rather than just asserting it does.

Two checks:
1. How many training archetypes sit near the 0/360 wraparound, and does
   any fitted cluster's reported (arithmetic-mean) spin_axis differ
   substantially from the true circular mean of its members?
2. For points near the wraparound, does correctly encoding spin_axis as
   (cos, sin) and refitting an equivalent scaler + centroids change which
   cluster they're nearest to, compared to the current linear treatment?

Analysis/reporting script. See docs/METHODOLOGY_REVIEW.md for write-up.
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_PATH = _ROOT / 'models' / 'tda_mapper_model.pkl'
WRAPAROUND_WINDOW_DEG = 20


def load_model(model_path=_DEFAULT_MODEL_PATH):
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def circular_mean_deg(degrees):
    rad = np.deg2rad(degrees)
    return np.rad2deg(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) % 360


def check_cluster_centroid_distortion(mc):
    orig = mc['original_data']
    graph = mc['graph']
    cs = mc['cluster_summary'].set_index('cluster')

    near_wrap = orig[(orig['spin_axis'] < WRAPAROUND_WINDOW_DEG) | (orig['spin_axis'] > 360 - WRAPAROUND_WINDOW_DEG)]
    print(f"Training archetypes within {WRAPAROUND_WINDOW_DEG}deg of the 0/360 wraparound: "
          f"{len(near_wrap)} of {len(orig)}")

    flagged = []
    for cid, members in graph['nodes'].items():
        if not members or cid not in cs.index:
            continue
        member_spins = orig.iloc[members]['spin_axis'].values
        naive_mean = np.mean(member_spins) % 360
        circ_mean = circular_mean_deg(member_spins)
        diff = min(abs(naive_mean - circ_mean), 360 - abs(naive_mean - circ_mean))
        if diff > 10:
            flagged.append((cid, len(members), naive_mean, circ_mean, diff))

    print(f"Clusters where the arithmetic-mean spin_axis differs from the true circular "
          f"mean by >10deg: {len(flagged)} of {len(graph['nodes'])}")
    for cid, n, naive, circ, diff in flagged:
        print(f"  {cid} (n={n}): naive_mean={naive:.1f}, true_circular_mean={circ:.1f}, diff={diff:.1f}")


def check_classification_impact(mc):
    orig = mc['original_data']
    graph = mc['graph']
    stuff_columns = mc['stuff_columns']
    input_columns = [c for c in stuff_columns if c != 'spin_axis_clock']
    scaler = mc['scaler']
    cs = mc['cluster_summary'].copy()
    cs['pfx_x'] = cs['HB'] / -12
    cs['pfx_z'] = cs['IVB'] / 12
    X_clusters = cs[input_columns].values.astype(np.float64)
    X_clusters_scaled = scaler.transform(X_clusters)
    cluster_ids = cs['cluster'].values

    non_spin_cols = [c for c in input_columns if c != 'spin_axis']
    spin_rad_all = np.deg2rad(orig['spin_axis'].values)
    X_corrected = np.column_stack([
        orig[non_spin_cols].values.astype(np.float64),
        np.cos(spin_rad_all),
        np.sin(spin_rad_all),
    ])
    scaler_corrected = StandardScaler().fit(X_corrected)
    X_corrected_scaled = scaler_corrected.transform(X_corrected)

    cluster_pos = {cid: i for i, cid in enumerate(cluster_ids)}
    corrected_centroids = np.zeros((len(cluster_ids), X_corrected_scaled.shape[1]))
    for cid, members in graph['nodes'].items():
        if cid not in cluster_pos or not members:
            continue
        corrected_centroids[cluster_pos[cid]] = X_corrected_scaled[members].mean(axis=0)

    near_wrap_idx = np.where(
        (orig['spin_axis'].values < WRAPAROUND_WINDOW_DEG) | (orig['spin_axis'].values > 360 - WRAPAROUND_WINDOW_DEG)
    )[0]
    print(f"\nTesting {len(near_wrap_idx)} near-wraparound points: does correcting spin_axis's "
          f"circularity change which cluster they're nearest to?\n")

    n_changed = 0
    for idx in near_wrap_idx:
        row = orig.iloc[idx][input_columns].values.astype(np.float64).reshape(1, -1)
        row_scaled = scaler.transform(row)
        dists_orig = np.linalg.norm(X_clusters_scaled - row_scaled, axis=1)
        nearest_orig = cluster_ids[np.argmin(dists_orig)]

        row_corrected_scaled = X_corrected_scaled[idx]
        dists_corrected = np.linalg.norm(corrected_centroids - row_corrected_scaled, axis=1)
        nearest_corrected = cluster_ids[np.argmin(dists_corrected)]

        changed = nearest_orig != nearest_corrected
        n_changed += changed
        pitcher_id, pitch_type = orig.index[idx]
        print(f"  spin_axis={orig.iloc[idx]['spin_axis']:.1f}  pitcher={pitcher_id} type={pitch_type}  "
              f"orig->{nearest_orig}  corrected->{nearest_corrected}  {'CHANGED' if changed else 'same'}")

    print(f"\nNearest-cluster assignment changed for {n_changed} of {len(near_wrap_idx)} "
          f"near-wraparound points ({100 * n_changed / len(near_wrap_idx):.0f}%).")


def main():
    mc = load_model()
    check_cluster_centroid_distortion(mc)
    check_classification_impact(mc)


if __name__ == "__main__":
    main()
