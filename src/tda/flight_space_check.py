"""
Do the shape results hold in FLIGHT space?

The shape results (Mapper features, nerve loops, persistent homology,
intrinsic dimension) were computed on 9 features that include the delivery
(release position, extension) and absolute velocity. The evaluation and
development analyses use flight space instead -- velocity relative to the
pitcher's fastest shape, pfx_x, pfx_z, spin rate, spin axis (cos, sin) --
because the delivery features make a pitcher's own pitches look alike (the
same-pitcher confound). This reruns the shape analyses in flight space with
the same settings and seeds, PCA lens only (a check of the feature-space
choice, not a full replication of the lens study). The density peaks were
already computed in a flight subspace (dSpeed, HB, IVB) and are not rerun.

Outputs: data/flight_space_<season>_<lens>_mapper.csv, and (PCA run only,
since they do not depend on the lens) data/flight_space_<season>_persistence.csv
and data/flight_space_<season>_idim.csv.

Run: python3 src/tda/flight_space_check.py statcast_<season>.parquet [lens] [season]
     lens in pca (default), baseball (relative velocity x IVB), isomap, density
"""
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tda_classifier import load_tda_model, season_shapes
from mapper_stability import (make_lens, fit_graph, features, gaussian_null, shuffle_within_type_null,
                              BASE, GRID, N_BOOT, SUBSAMPLE_FRAC, SEED)
from nerve_homology import betti1
from persistence_check import h0_deaths, summarize
from intrinsic_dimension import twonn, levina_bickel, MIN_TYPE_N

FLIGHT = ['rel_speed', 'pfx_x', 'pfx_z', 'release_spin_rate', 'spin_cos', 'spin_sin']
N_NULL = 10        # Mapper nulls, as in mapper_stability.py
N_NULL_SMALL = 5   # nerve / PH / dimension nulls, as in their scripts


def flight_shapes(path):
    mc = load_tda_model(_ROOT / 'models' / 'tda_mapper_model.pkl')
    shapes, n = season_shapes(pd.read_parquet(path), list(mc['stuff_columns']))
    sp = shapes['release_speed'].astype(float)
    shapes = shapes.assign(rel_speed=sp - sp.groupby(level=0).transform('max'))
    ptype = shapes.index.get_level_values('pitch_type').values.astype(str)
    pitcher = shapes.index.get_level_values(0).values
    return shapes[FLIGHT].values.astype(float), sp.values, ptype, pitcher


def main(path, lens_name='pca', tag='2025'):
    rng = np.random.default_rng(SEED)
    X_raw, speed, ptype, pitcher = flight_shapes(path)
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)
    ev = PCA(2).fit(X).explained_variance_ratio_
    print(f"{len(X)} shapes in flight space ({X.shape[1]} coordinates); PC1 {100 * ev[0]:.1f}%, PC2 {100 * ev[1]:.1f}%")
    spin_pair = [(FLIGHT.index('spin_cos'), FLIGHT.index('spin_sin'))]

    # Mapper: saved parameters, grid, subsamples, nulls (PCA lens)
    rows = []
    def run(kind, Xr, sp, pt, params):
        Xs = StandardScaler().fit_transform(Xr)
        lens = make_lens(Xs, lens_name)
        for p in params:
            G, nodes = fit_graph(Xs, lens, **p)
            gb1, nb1 = betti1(nodes)
            rows.append(dict(kind=kind, **p, **features(G, nodes, sp, pt), graph_b1=gb1, nerve_b1=nb1))
    grid = [dict(n_cubes=a, perc_overlap=b, eps=c) for a in GRID['n_cubes'] for b in GRID['perc_overlap'] for c in GRID['eps']]
    run('grid', X_raw, speed, ptype, grid)
    uniq = np.unique(pitcher)
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=int(SUBSAMPLE_FRAC * len(uniq)), replace=False)
        idx = np.flatnonzero(np.isin(pitcher, pick))
        run('subsample', X_raw[idx], speed[idx], ptype[idx], [BASE])
    for _ in range(N_NULL):
        Xg = gaussian_null(X_raw, rng)
        # null shapes inherit speed/type from their nearest real shape, as in mapper_stability.py
        nn = np.argmin(((scaler.transform(Xg)[:, None, :] - X[None, :, :]) ** 2).sum(-1), axis=1)
        run('null_gauss', Xg, speed[nn], ptype[nn], [BASE])
        Xsh = shuffle_within_type_null(X_raw, ptype, rng, spin_pair)
        run('null_shuffle', Xsh, speed, ptype, [BASE])
    mres = pd.DataFrame(rows)
    mres.to_csv(_ROOT / 'data' / f'flight_space_{tag}_{lens_name}_mapper.csv', index=False)
    summary(mres, None, None)
    if lens_name != 'pca':
        return

    # persistent homology
    unit = np.median(h0_deaths(X))
    prow = [summarize('real', X, unit, rng)]
    for _ in range(N_NULL_SMALL):
        prow.append(summarize('null_gauss', scaler.transform(gaussian_null(X_raw, rng)), unit, rng))
        prow.append(summarize('null_shuffle', scaler.transform(shuffle_within_type_null(X_raw, ptype, rng, spin_pair)), unit, rng))
    pres = pd.DataFrame(prow)
    pres.to_csv(_ROOT / 'data' / f'flight_space_{tag}_persistence.csv', index=False)

    # intrinsic dimension (fair null = spin-pair shuffle)
    types = [t for t, c in zip(*np.unique(ptype, return_counts=True)) if c >= MIN_TYPE_N]
    drow = []
    for kind, Xr in [('real', X_raw)] + [(k, f()) for _ in range(N_NULL_SMALL) for k, f in
                                          [('null_gauss', lambda: gaussian_null(X_raw, rng)),
                                           ('null_shuffle_spinpair', lambda: shuffle_within_type_null(X_raw, ptype, rng, spin_pair))]]:
        Xs = scaler.transform(Xr)
        drow.append(dict(scope='all', kind=kind, twonn=twonn(Xs), mle=levina_bickel(Xs)))
        if kind != 'null_gauss':
            for t in types:
                Xt = StandardScaler().fit_transform(Xs[ptype == t])
                drow.append(dict(scope=t, kind=kind, twonn=twonn(Xt), mle=levina_bickel(Xt)))
    dres = pd.DataFrame(drow)
    dres.to_csv(_ROOT / 'data' / f'flight_space_{tag}_idim.csv', index=False)
    summary(None, pres, dres)


def summary(mres, pres, dres):
    if mres is None:
        return summary_ph(pres, dres)
    order = lambda g: 100 * np.mean(g['ordering_rho'].dropna() >= 0.7)
    slow = lambda g: 100 * np.mean(g['slow_isolated'].dropna() >= 0.5)
    base = mres[(mres.kind == 'grid') & (mres.n_cubes == BASE['n_cubes']) & (mres.perc_overlap == BASE['perc_overlap']) & (mres.eps == BASE['eps'])].iloc[0]
    print(f"\nMapper at saved parameters: {base.n_nodes} nodes, {base.n_components} components, giant {base.giant_frac:.2f}, "
          f"slow isolated {base.slow_isolated:.2f}, ordering rho {base.ordering_rho:.2f}, graph b1 {base.graph_b1}, nerve b1 {base.nerve_b1}")
    for k, g in mres.groupby('kind'):
        print(f"  {k:13s} fits {len(g):3d}: ordering>=0.7 {order(g):5.1f}%  slow isolated(>=half) {slow(g):5.1f}%  "
              f"components {g.n_components.min()}-{g.n_components.max()}  graph b1 {g.graph_b1.min()}-{g.graph_b1.max()}  nerve b1 {g.nerve_b1.min()}-{g.nerve_b1.max()}")


def summary_ph(pres, dres):
    print("\nPersistent homology (units = real median H0 death):")
    print(pres.groupby('kind', sort=False)[['h0_median', 'h0_long_bars', 'h1_long_loops_mean', 'h1_max_persistence']].agg(['mean', 'min', 'max']).round(2).to_string())
    print("\nIntrinsic dimension (ambient 6 coordinates, 5 quantities):")
    m = dres.groupby(['scope', 'kind'])[['mle', 'twonn']].mean().unstack()
    print(m.round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:4])
