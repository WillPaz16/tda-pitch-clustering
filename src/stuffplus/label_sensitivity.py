"""
Label sensitivity of per-pitch-type Stuff+ grades.

Production Stuff+ (stuff_plus_calculator.py) grades a pitch against the
league distribution of ITS OWN pitch-type label: component z-scores use that
label's 2022-24 reference stats, and the result is rescaled to 100 +/- 15
within the label. If labels are cuts through a connected, ridge-shaped space
(docs/DISCOVERY_FINDINGS.md: density peaks), a pitch near a label boundary
could be graded very differently just by which side its label falls on.

Test (2025 season, pitch shapes with n >= MIN_N pitches):
  1. Boundary pitches: among a shape's K nearest shapes FROM OTHER PITCHERS,
     measured on flight features only (velocity, movement, spin: the Mapper
     model's scaling, release point and extension dropped), the most common
     OTHER label holds >= BOUNDARY_SHARE of them. That label is the plausible
     alternative. (With all 9 features and same-pitcher neighbors allowed, a
     pitcher's own pitches look close through the shared delivery: the
     same-pitcher confound in DISCOVERY_FINDINGS.md.)
     Velocity enters either as absolute mph or, by default, RELATIVE to the
     pitcher's fastest pitch shape (as hitters experience it, and as in
     Maloof et al. 2026): with absolute mph a hard 91-mph changeup neighbours
     other pitchers' 91-mph sinkers.
  2. Grade each boundary pitch twice with the production formula: against its
     own label (must equal the calculator's stuff_plus_pt exactly) and
     against the alternative label. Same pitches, same model predictions.
  3. Compare the shift with the grade's own sampling error (bootstrap over the
     shape's pitches, reference stats held fixed).

Run: python3 src/stuffplus/label_sensitivity.py statcast_2025.parquet [relative|absolute]
"""
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_ROOT / 'src' / 'tda'))
from stuff_plus_calculator import StuffPlusCalculator
from tda_classifier import load_tda_model

K, BOUNDARY_SHARE, MIN_N, N_BOOT, SEED = 20, 0.5, 150, 200, 42
FLIGHT = ['release_speed', 'pfx_x', 'pfx_z', 'release_spin_rate', 'spin_cos', 'spin_sin']


def stuff_z(m, s, w):
    """Production component z-scores (xwOBA sign-flipped) and weighted sum."""
    zx = -(m[..., 0] - s['mu_xw']) / s['sd_xw']
    zm = (m[..., 1] - s['mu_miss']) / s['sd_miss']
    zc = (m[..., 2] - s['mu_chase']) / s['sd_chase']
    return w[0] * zx + w[1] * zm + w[2] * zc


def main(path, velo='relative'):
    calc = StuffPlusCalculator()
    df = pd.read_parquet(path).rename(columns={'pitcher': 'pitcher_id'})
    res = calc.calculate(df)
    pl, pt = res['pitch_level'], res['per_pitch_type']
    stats = calc.stuff_plus_metadata['pitch_type_stats']
    w = np.array(calc.stuff_plus_metadata['weights_used'], float)
    min_pt = calc.stuff_plus_metadata['min_pitches_per_type']

    # per-label rescaling constants, exactly as calculate() computes them
    pt = pt.set_index(['pitcher_id', 'pitch_type'])
    scale = {}
    for lab, g in pt.groupby(level='pitch_type'):
        st = g[g['pitch_count_pt'] >= min_pt]
        st = g if len(st) < 10 else st
        scale[lab] = (st['stuff_z_pt'].mean(), max(st['stuff_z_pt'].std(ddof=0), 0.01))

    def grade(m, lab):
        mu, sd = scale[lab]
        return 100 + 15 * (stuff_z(m, stats[lab], w) - mu) / sd

    # nearest labelled shapes in the Mapper model's feature space
    mc = load_tda_model(_ROOT / 'models' / 'tda_mapper_model.pkl')
    arch = mc['original_data']
    cols = list(mc['stuff_columns'])
    X = mc['scaler'].transform(arch[cols].values.astype(float))[:, [cols.index(c) for c in FLIGHT]]
    labels = arch.index.get_level_values('pitch_type').astype(str).values
    pitcher = arch.index.get_level_values(0).values
    if velo == 'relative':                      # speed minus the pitcher's fastest shape, standardized
        sp = arch['release_speed'].astype(float)
        d = (sp - sp.groupby(level=0).transform('max')).values
        X[:, FLIGHT.index('release_speed')] = (d - d.mean()) / d.std()
    _, cand = NearestNeighbors(n_neighbors=K + 15).fit(X).kneighbors(X)
    nb = [[j for j in row if pitcher[j] != pitcher[i]][:K] for i, row in enumerate(cand)]

    by_shape = {k: g[['pred_xw', 'pred_miss', 'pred_chase']].values
                for k, g in pl.groupby(['pitcher_id', 'pitch_type'])}
    rng = np.random.default_rng(SEED)
    rows = []
    for i, key in enumerate(arch.index):
        if key not in pt.index or pt.loc[key, 'pitch_count_pt'] < MIN_N or labels[i] not in stats:
            continue
        others = pd.Series(labels[nb[i]])
        others = others[(others != labels[i]) & others.isin(list(stats)) & others.isin(list(scale))]
        if others.empty:
            continue
        alt, cnt = others.value_counts().index[0], others.value_counts().iloc[0]
        p = by_shape[key]
        m = p.mean(axis=0)
        own, other = grade(m, labels[i]), grade(m, alt)
        assert abs(own - pt.loc[key, 'stuff_plus_pt']) < 1e-6, (key, own, pt.loc[key, 'stuff_plus_pt'])
        boots = p[rng.integers(0, len(p), (N_BOOT, len(p)))].mean(axis=1)
        rows.append({'pitcher_id': key[0], 'pitch_type': labels[i], 'n': len(p), 'alt_label': alt,
                     'boundary_share': cnt / K, 'grade_own': own, 'grade_alt': other,
                     'shift': other - own, 'se_grade': grade(boots, labels[i]).std(ddof=1)})
    out = pd.DataFrame(rows)
    eligible = sum(1 for i, key in enumerate(arch.index)
                   if key in pt.index and pt.loc[key, 'pitch_count_pt'] >= MIN_N and labels[i] in stats)
    names = df.drop_duplicates('pitcher_id').set_index('pitcher_id')['player_name']
    out['name'] = out['pitcher_id'].map(names)
    out.to_csv(_ROOT / 'data' / f'label_sensitivity_{velo}.csv', index=False)

    b = out[out['boundary_share'] >= BOUNDARY_SHARE]
    a = b['shift'].abs()
    print(f"[{velo} velocity] {eligible} pitch shapes with n >= {MIN_N}; own-label grades match the production calculator exactly")
    print(f"boundary pitches (>= {BOUNDARY_SHARE:.0%} of {K} nearest other-pitcher shapes carry one other label): "
          f"{len(b)} ({len(b) / eligible:.0%})")
    print(f"  |grade shift|: median {a.median():.1f}, mean {a.mean():.1f}; "
          f">= 10 points: {(a >= 10).mean():.0%}; >= 15 (1 SD): {(a >= 15).mean():.0%}; >= 30: {(a >= 30).mean():.0%}")
    print(f"  grade sampling error (bootstrap SE): median {b['se_grade'].median():.1f}; "
          f"shift > 2 SE: {(a > 2 * b['se_grade']).mean():.0%}")
    print(f"  crosses 100 (above-average under one label, below under the other): "
          f"{((b['grade_own'] - 100) * (b['grade_alt'] - 100) < 0).mean():.0%}")
    pairs = b.apply(lambda r: f"{r['pitch_type']}->{r['alt_label']}", axis=1)
    summ = b.assign(pair=pairs).groupby('pair').agg(n=('shift', 'size'), median_shift=('shift', 'median'),
                                                    median_abs=('shift', lambda s: s.abs().median()))
    print(summ.sort_values('n', ascending=False).head(12).round(1).to_string())
    ex = b.reindex(a.sort_values(ascending=False).index).head(8)
    print(ex[['name', 'pitch_type', 'alt_label', 'n', 'boundary_share', 'grade_own', 'grade_alt', 'se_grade']].round(1).to_string(index=False))
    return out


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or ['relative']))
