"""
Density peaks of pitch-shape space (complements the Rips/Mapper analysis).

Our persistent homology is computed on the point cloud (Vietoris-Rips), so it
detects GAPS: "one connected continuum" means no long-lived H0 bars beyond a
Gaussian null. It says nothing about how many dense regions (modes) the
continuum has. Maloof, Haumacher & Bonicki (2026, SABR Tooth Tigers) count
modes instead: persistent homology of superlevel sets of a smoothed density,
on 4.47M NCAA D1 pitches in (dSpeed, HB, IVB), finding three peaks
(fastball, breaking ball, changeup). This script asks the same question of
our data.

  A. Pitch level, their recipe on MLB 2025: dSpeed = release speed minus the
     pitcher's 95th-percentile speed in that game; HB, IVB in inches,
     left-handers mirrored, + HB = arm side. 50^3 voxel grid over HB, IVB in
     [-25, 25] in and dSpeed in [-25, 5] mph, Gaussian smoothing sigma = 1
     voxel. Peaks = H0 of superlevel sets (union-find over 26-neighbors).
  B. Archetype level: 1-D density of the 3,943 pitch shapes along PC1 of the
     Mapper model's own lens, over a range of KDE bandwidths.

Significance: a covariance-matched Gaussian null (Topaz 2026; our Mapper
nulls), refit through the identical pipeline. Each secondary peak gets a
Monte Carlo p-value (1 + #null replicates whose largest secondary peak is at
least as persistent) / (B + 1); it counts if p < 0.05.

Run: python3 src/tda/density_peaks.py statcast_2025.parquet
     python3 src/tda/density_peaks.py --selftest
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

_ROOT = Path(__file__).resolve().parents[2]
EDGES = (np.linspace(-25, 25, 51), np.linspace(-25, 25, 51), np.linspace(-25, 5, 51))   # HB, IVB, dSpeed
SIGMA, FLOOR, N_NULL, SEED, ALPHA = 1.0, 1e-3, 50, 42, 0.05


def superlevel_persistence(field, floor=FLOOR):
    """H0 persistence of superlevel sets of a grid function (any dimension).
    Returns (birth height, persistence) rows, most persistent first; the global
    max gets persistence = its height. Cells below floor * max are ignored."""
    shape = field.shape
    flat = field.ravel()
    order = np.argsort(-flat)
    order = order[flat[order] >= floor * flat.max()]
    offs = [np.array(o) - 1 for o in np.ndindex(*(3,) * len(shape)) if any(x != 1 for x in o)]
    parent = -np.ones(flat.size, dtype=np.int64)          # -1 = not yet in the superlevel set
    birth = {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pers = []
    for idx in order:
        parent[idx] = idx
        birth[idx] = flat[idx]
        coord = np.array(np.unravel_index(idx, shape))
        for o in offs:
            c = coord + o
            if (c < 0).any() or (c >= shape).any():
                continue
            j = np.ravel_multi_index(tuple(c), shape)
            if parent[j] < 0:
                continue
            a, b = find(idx), find(j)
            if a == b:
                continue
            young, old = (a, b) if birth[a] < birth[b] else (b, a)
            if young != idx:                                  # a real peak dies here (idx is its saddle)
                pers.append((birth[young], birth[young] - flat[idx]))
            parent[young] = old
    roots = {find(i) for i in order}
    pers += [(birth[r], birth[r]) for r in roots]
    pers.sort(key=lambda p: -p[1])
    return np.array(pers)


def voxel_density(X):
    H, _ = np.histogramdd(X, bins=EDGES)
    return gaussian_filter(H, SIGMA)


def peak_test(X, rng, n_null=N_NULL, density=voxel_density):
    """Real persistences (normalised by the tallest peak), Monte Carlo p-value
    of each against the null's largest secondary persistence, null maxima."""
    real = superlevel_persistence(density(X))
    real_rel = real[:, 1] / real[0, 1]
    mu, cov = X.mean(axis=0), np.atleast_2d(np.cov(X, rowvar=False))
    null_sec = []
    for _ in range(n_null):
        Z = rng.multivariate_normal(mu, cov, size=len(X))
        p = superlevel_persistence(density(Z))
        null_sec.append(p[1, 1] / p[0, 1] if len(p) > 1 else 0.0)
    null_sec = np.array(null_sec)
    pval = np.array([(1 + (null_sec >= r).sum()) / (n_null + 1) for r in real_rel])
    return real_rel, pval, null_sec


def pitch_level(df):
    d = df[['pitcher', 'game_pk', 'p_throws', 'release_speed', 'pfx_x', 'pfx_z']].dropna()
    mirror = np.where(d['p_throws'] == 'L', -1.0, 1.0)
    hb = -d['pfx_x'].values * mirror * 12                  # mirrored RHP frame: + = arm side
    ivb = d['pfx_z'].values * 12
    top = d.groupby(['pitcher', 'game_pk'])['release_speed'].transform(lambda s: s.quantile(0.95))
    ds = (d['release_speed'] - top).values
    return np.c_[hb, ivb, ds]


def peak_locations(field, pers, mids, k=6):
    """Grid coordinates of the k most persistent peaks (matched by birth height)."""
    flat = field.ravel()
    return [tuple(round(float(mids[a][c]), 1) for a, c in
                  enumerate(np.unravel_index(int(np.argmin(np.abs(flat - b))), field.shape)))
            for b in pers[:k, 0]]


def archetype_pc1(bandwidths=(0.10, 0.15, 0.20, 0.30)):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tda_classifier import load_tda_model
    mc = load_tda_model(_ROOT / 'models' / 'tda_mapper_model.pkl')
    X = mc['scaler'].transform(mc['original_data'][mc['stuff_columns']].values.astype(float))
    pc1 = mc['pca'].transform(X)[:, 0]
    grid = np.linspace(pc1.min() - 1, pc1.max() + 1, 2000)
    rows = []
    for bw in bandwidths:
        dens = lambda v: gaussian_kde(v.ravel(), bw_method=bw)(grid)
        real, pval, null_sec = peak_test(pc1[:, None], np.random.default_rng(SEED), density=dens)
        f = dens(pc1)
        locs = peak_locations(f, superlevel_persistence(f), [grid], k=4)
        rows.append({'bandwidth': bw, 'peaks_raw': len(real), 'peaks_sig': int((pval[1:] < ALPHA).sum()) + 1,
                     'null_max': round(float(null_sec.max()), 4), 'persistence': np.round(real[:4], 3).tolist(),
                     'p': np.round(pval[1:4], 3).tolist(),
                     'at_pc1': [l[0] for l in locs]})
    return pd.DataFrame(rows)


def selftest():
    """Persistence machinery on clear-cut cases: 3 separated blobs give 3 peaks
    far above noise; one Gaussian gives 1 (all other bumps are noise-sized)."""
    rng = np.random.default_rng(0)
    blobs = np.vstack([rng.normal(c, 2.0, (4000, 3)) for c in ([10, 15, -1], [-5, 1, -10], [15, 6, -8])])
    real, pval, null_sec = peak_test(blobs, rng, n_null=10)
    assert (real > 0.2).sum() == 3 and (pval[1:3] < 0.1).all(), (real[:5], pval[:4])
    one = rng.normal([5, 5, -5], 3.0, (12000, 3))
    real, _, null_sec = peak_test(one, rng, n_null=10)
    assert (real > 0.05).sum() == 1 and null_sec.max() < 0.05, real[:5]
    print("selftest ok: 3 blobs -> 3 peaks, 1 Gaussian -> 1 peak")


def main(path):
    X = pitch_level(pd.read_parquet(path))
    real, pval, null_sec = peak_test(X, np.random.default_rng(SEED))
    field = voxel_density(X)
    locs = peak_locations(field, superlevel_persistence(field), [(e[:-1] + e[1:]) / 2 for e in EDGES])
    print(f"A. pitch level: {len(X):,} pitches")
    print(f"   persistence / tallest peak: {np.round(real[:8], 4).tolist()}")
    print(f"   null (cov-matched Gaussian, {N_NULL} reps): largest secondary persistence "
          f"median {np.median(null_sec):.4f}, max {null_sec.max():.4f}")
    print(f"   significant peaks (p < {ALPHA}): {int((pval[1:] < ALPHA).sum()) + 1}")
    for k, (pr, pv, loc) in enumerate(zip(real[:6], pval[:6], locs)):
        print(f"   peak {k + 1}: persistence {pr:.3f} (p = {pv:.3f}) at (HB, IVB, dSpeed) = {loc}")
    pd.DataFrame({'persistence_rel': real, 'p_value': pval}).to_csv(_ROOT / 'data' / 'density_peaks.csv', index=False)
    print("B. archetype level, PC1 of the Mapper lens:")
    print(archetype_pc1().to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:] == ['--selftest']:
        selftest()
    else:
        main(sys.argv[1])
