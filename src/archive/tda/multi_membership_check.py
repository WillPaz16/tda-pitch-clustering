"""
Hard Nearest-Centroid vs. Mapper's Multi-Membership Theory

Follow-up to docs/METHODOLOGY_REVIEW.md item 3: true Mapper lets a point
belong to multiple simplices because cover sets overlap; inference here
picks a single argmin nearest cluster. Checks how much real ambiguity
that discards: for every training archetype, the margin between distance
to the 1st- and 2nd-nearest cluster centroid, normalized by the typical
(median) inter-centroid distance.

Analysis/reporting script. See docs/METHODOLOGY_REVIEW.md for write-up.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_PATH = _ROOT / 'models' / 'tda_mapper_model.pkl'

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tda_classifier import load_tda_model


def main():
    mc = load_tda_model(_DEFAULT_MODEL_PATH)
    scaler = mc['scaler']
    input_columns = [c for c in mc['stuff_columns'] if c != 'spin_axis_clock']
    cs = mc['cluster_summary'].copy()
    cs['pfx_x'] = cs['HB'] / -12
    cs['pfx_z'] = cs['IVB'] / 12
    X_clusters_scaled = scaler.transform(cs[input_columns].values.astype(np.float64))

    n = len(X_clusters_scaled)
    pairwise = [np.linalg.norm(X_clusters_scaled[i] - X_clusters_scaled[j])
                for i in range(n) for j in range(i + 1, n)]
    median_pairwise = np.median(pairwise)
    print(f"Median inter-centroid distance: {median_pairwise:.3f} "
          f"(min: {min(pairwise):.3f}, max: {max(pairwise):.3f})")
    if min(pairwise) < 1e-6:
        print("  -> at least two clusters have an effectively identical centroid (a true duplicate).")

    orig = mc['original_data']
    X = scaler.transform(orig[input_columns].values.astype(np.float64))
    dists = np.linalg.norm(X_clusters_scaled[None, :, :] - X[:, None, :], axis=2)
    order = np.argsort(dists, axis=1)
    d1 = np.take_along_axis(dists, order[:, :1], axis=1).ravel()
    d2 = np.take_along_axis(dists, order[:, 1:2], axis=1).ravel()
    margin = (d2 - d1) / median_pairwise

    print(f"\nMargin between 1st- and 2nd-nearest cluster, {len(margin)} training archetypes:")
    print(pd.Series(margin).describe())
    for thresh in (0.05, 0.10, 0.20):
        frac = (margin < thresh).mean()
        print(f"Fraction with margin < {thresh}: {100 * frac:.1f}% "
              f"(genuinely ambiguous between top-2 clusters -- a hard single label discards this)")


if __name__ == "__main__":
    main()
