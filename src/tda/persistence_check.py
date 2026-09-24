"""
Persistent homology of the pitch-shape archetypes vs. null models.

Runs on the saved archetypes: one average profile per (pitcher, pitch_type),
the same unit the Mapper graph is fit on -- not individual pitches.

Question: does pitch-shape space have topological structure that a
structureless cloud (same-covariance Gaussian) or a per-pitch-type blob
mixture (features shuffled within pitch type) does not?

  H0 (all archetypes): merge scales of Rips components. Genuine gaps between
      modes show up as many long-lived H0 bars.
  H1 (repeated 1,000-point subsamples): loops. Tests whether the Mapper
      graph's many cycles (b1 = 76-86) reflect real homology.

All datasets use the real data's StandardScaler, and lifetimes are reported
in units of the real data's median H0 death, so scales are comparable.
A noisy circle is run first as a sanity check (expect exactly one long H1 bar).
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ripser import ripser
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tda_classifier import load_tda_model
from mapper_stability import gaussian_null, shuffle_within_type_null

SEED, N_NULL, N_SUB, SUB_SIZE = 42, 5, 5, 1000
GAP_MULT, LOOP_MULT = 3.0, 2.0   # "long" H0 bar / "real" H1 loop, in median-H0 units


def h0_deaths(X):
    d = ripser(X, maxdim=0)['dgms'][0][:, 1]
    return d[np.isfinite(d)]


def h1_persistence(X):
    dgm = ripser(X, maxdim=1)['dgms'][1]
    return dgm[:, 1] - dgm[:, 0] if len(dgm) else np.array([])


def summarize(kind, X, unit, rng):
    d0 = h0_deaths(X) / unit
    loops = []
    for _ in range(N_SUB):
        sub = X[rng.choice(len(X), size=min(SUB_SIZE, len(X)), replace=False)]
        p1 = h1_persistence(sub) / unit
        loops.append((np.sum(p1 > LOOP_MULT), p1.max() if len(p1) else 0.0))
    loops = np.array(loops)
    return dict(kind=kind, h0_median=np.median(d0), h0_p99=np.quantile(d0, 0.99), h0_max=d0.max(),
                h0_long_bars=int(np.sum(d0 > GAP_MULT)),
                h1_long_loops_mean=loops[:, 0].mean(), h1_max_persistence=loops[:, 1].mean())


def sanity_circle(rng):
    t = rng.uniform(0, 2 * np.pi, 500)
    X = np.c_[np.cos(t), np.sin(t)] + rng.normal(0, 0.05, (500, 2))
    p1 = h1_persistence(X)
    long = np.sum(p1 > 0.5 * p1.max())
    print(f"Sanity (noisy circle): {long} dominant H1 bar(s), max persistence {p1.max():.2f} "
          f"vs next {np.sort(p1)[-2]:.2f}")
    assert long == 1, "persistence pipeline failed the circle sanity check"


def main():
    rng = np.random.default_rng(SEED)
    sanity_circle(rng)

    mc = load_tda_model(_ROOT / 'models' / 'tda_mapper_model.pkl')
    orig = mc['original_data']
    X_raw = orig[mc['stuff_columns']].values.astype(np.float64)
    ptype = orig.index.get_level_values('pitch_type').values.astype(str)
    scaler = StandardScaler().fit(X_raw)
    X_real = scaler.transform(X_raw)
    unit = np.median(h0_deaths(X_real))

    rows = [summarize('real', X_real, unit, rng)]
    for _ in range(N_NULL):
        rows.append(summarize('null_gauss', scaler.transform(gaussian_null(X_raw, rng)), unit, rng))
        rows.append(summarize('null_shuffle', scaler.transform(shuffle_within_type_null(X_raw, ptype, rng)), unit, rng))

    res = pd.DataFrame(rows)
    out = _ROOT / 'data' / 'persistence_check.csv'
    res.to_csv(out, index=False)
    pd.set_option('display.width', 160)
    print(f"\nUnits: real median H0 death = {unit:.3f} (scaled feature space). "
          f"Long H0 bar > {GAP_MULT}, real loop > {LOOP_MULT} units.\n")
    print(res.groupby('kind', sort=False).agg(['mean', 'min', 'max']).round(2).T.to_string())
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
