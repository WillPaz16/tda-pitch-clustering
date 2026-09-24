"""
Intrinsic dimension of the pitch-shape archetypes vs. null models.

Follow-up to persistence_check.py, which found the real archetypes far more
concentrated than a same-covariance Gaussian or a within-pitch-type shuffle.
Hypothesis: within-type coupling of the physical features puts the archetypes
on a lower-dimensional continuum inside the 9-D feature space.

Estimators (standard, nearest-neighbor based):
  TwoNN          -- Facco et al. 2017: d from the ratio of 2nd to 1st neighbor
                    distances (top 10% of ratios discarded for robustness)
  Levina-Bickel  -- MLE over k=10 neighbors (MacKay-Ghahramani averaging)

Reported overall and within each pitch type with >= 200 archetypes, for real
data and each null. Sanity-checked first on a 2-D plane embedded in 9-D
(expect ~2) and a 9-D Gaussian (expect ~9).
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tda_classifier import load_tda_model
from mapper_stability import gaussian_null, shuffle_within_type_null

SEED, N_NULL, K, MIN_TYPE_N = 42, 5, 10, 200


def neighbor_dists(X, k):
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    d = d[:, 1:]                       # drop self
    return d[d[:, 0] > 0]              # drop exact duplicates


def twonn(X):
    # MLE form, no tail trimming: trimming the largest ratios biased the
    # estimate upward (2.68 on a 2-D plane, 12.1 on a 9-D Gaussian)
    d = neighbor_dists(X, 2)
    mu = d[:, 1] / d[:, 0]
    return len(mu) / np.sum(np.log(mu))


def levina_bickel(X, k=K):
    d = neighbor_dists(X, k)
    inv = np.mean(np.log(d[:, -1:] / d[:, :-1]), axis=1)   # 1 / m_k(x)
    return 1.0 / np.mean(inv)


def sanity(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(9, 9)))
    plane = rng.uniform(-1, 1, (3000, 2)) @ Q[:2]
    gauss = rng.normal(size=(3000, 9))
    for name, X, target in (('2-D plane in 9-D', plane, 2), ('9-D Gaussian', gauss, 9)):
        print(f"Sanity {name}: TwoNN {twonn(X):.2f}, MLE {levina_bickel(X):.2f} (expect ~{target})")


def main():
    rng = np.random.default_rng(SEED)
    sanity(rng)

    mc = load_tda_model(_ROOT / 'models' / 'tda_mapper_model.pkl')
    orig = mc['original_data']
    X_raw = orig[mc['stuff_columns']].values.astype(np.float64)
    ptype = orig.index.get_level_values('pitch_type').values.astype(str)
    scaler = StandardScaler().fit(X_raw)
    types = [t for t, n in zip(*np.unique(ptype, return_counts=True)) if n >= MIN_TYPE_N]

    # spin_cos/spin_sin encode one angle (always on the unit circle); a fair
    # null must move them together, or it breaks a constraint the real data
    # has by construction and looks higher-dimensional for a trivial reason
    spin_pair = [(mc['stuff_columns'].index('spin_cos'), mc['stuff_columns'].index('spin_sin'))]
    datasets = [('real', X_raw)]
    for _ in range(N_NULL):
        datasets.append(('null_gauss', gaussian_null(X_raw, rng)))
        datasets.append(('null_shuffle', shuffle_within_type_null(X_raw, ptype, rng)))
        datasets.append(('null_shuffle_spinpair', shuffle_within_type_null(X_raw, ptype, rng, spin_pair)))

    rows = []
    for kind, Xr in datasets:
        X = scaler.transform(Xr)
        rows.append(dict(scope='all', kind=kind, twonn=twonn(X), mle=levina_bickel(X), n=len(X)))
        if kind == 'null_gauss':
            continue                   # Gaussian has no pitch types
        for t in types:
            Xt = X[ptype == t]
            # within a type, standardize per type so the estimate isn't driven by global scale
            Xt = StandardScaler().fit_transform(Xt)
            rows.append(dict(scope=t, kind=kind, twonn=twonn(Xt), mle=levina_bickel(Xt), n=len(Xt)))

    res = pd.DataFrame(rows)
    out = _ROOT / 'data' / 'intrinsic_dimension.csv'
    res.to_csv(out, index=False)
    pd.set_option('display.width', 160)
    summ = res.groupby(['scope', 'kind'], sort=False)[['twonn', 'mle']].mean().unstack('kind')
    print("\nIntrinsic dimension (ambient = 9; nulls averaged over draws):")
    print(summ.round(2).to_string())
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
