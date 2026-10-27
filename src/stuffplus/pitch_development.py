"""
Pitch development test: does moving a pitch's shape toward its better-scoring
neighbours predict a real improvement the next season?

PRE-SPECIFIED DESIGN (written and committed before the analysis was run;
docs/DISCOVERY_FINDINGS.md records the result, whatever it is):

Unit. A (pitcher, pitch_type) with >= MIN_N pitches in season s AND s+1.
Shape space. Flight features as in label_sensitivity.py: velocity RELATIVE to
  the pitcher's fastest shape that season, pfx_x, pfx_z, spin rate, spin axis
  (cos, sin); LHP mirrored; fixed standardization (the Mapper model's scaler;
  relative velocity standardized with fixed 2025 constants) so year-over-year
  changes are comparable.
Better neighbours. In season s, the K nearest shapes from OTHER pitchers
  (n >= MIN_N); "better" = top third by a pooled, physics-only model score
  (mean per-pitch CatBoost xwOBA/miss/chase predictions, equal-weight z across
  all shapes, NOT per label). Direction v = centroid(better) - own shape.
Progress. delta = shape(s+1) - shape(s); progress = (delta . v) / |v|^2, the
  fraction of the way moved toward the better-neighbour centroid; winsorized
  at the 1st/99th percentiles.
Outcome (actual, model-free). Primary: change in SwStr% (swinging strikes per
  pitch). Secondary: change in chase% (swings on out-of-zone pitches / out-of-
  zone pitches) and in mean xwOBA on contact.
Model. WLS: d_outcome ~ progress + |delta| + baseline outcome(s) + change in
  the pitch's absolute velocity + pitch-type and season-pair fixed effects;
  weights = harmonic mean of the two seasons' pitch counts; standard errors
  clustered by pitcher.
Primary season pairs: 2024->2025 and 2025->2026 (the outcome season is after
  the models' 2022-24 training data). Secondary: 2023->2024.
Hypothesis. beta(progress) > 0 for SwStr%.
Placebos. Same model with v toward (a) a random third of the neighbours:
  expected ~0; (b) the worst third: expected beta < 0 (moving toward worse
  shapes hurts) -- if (a) or (b) look like the real effect, the "better
  neighbour" direction carries no specific information.
Caveat. Observational: shape changes come with injuries, role changes, coaching
  and usage changes; a positive result is an association, not an effect of
  pitch design.

Run: python3 src/stuffplus/pitch_development.py <dir with statcast_<year>.parquet>
"""
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.neighbors import NearestNeighbors

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_ROOT / 'src' / 'tda'))
from stuff_plus_calculator import StuffPlusCalculator
from tda_classifier import load_tda_model, season_shapes
from label_sensitivity import FLIGHT

K, MIN_N, SEED = 20, 150, 42
PAIRS = {(2024, 2025): 'primary', (2025, 2026): 'primary', (2023, 2024): 'secondary'}
MISS = ['swinging_strike', 'swinging_strike_blocked', 'missed_bunt']
SWING = MISS + ['foul', 'foul_bunt', 'foul_tip', 'bunt_foul_tip', 'hit_into_play']


def season_table(df, mc, calc, rel_const):
    """Per shape: standardized flight vector, n, model score, actual outcomes."""
    cols = list(mc['stuff_columns'])
    shapes, n = season_shapes(df, cols)
    X = mc['scaler'].transform(shapes[cols].values.astype(float))[:, [cols.index(c) for c in FLIGHT]]
    sp = shapes['release_speed'].astype(float)
    rel = (sp - sp.groupby(level=0).transform('max')).values
    X[:, FLIGHT.index('release_speed')] = (rel - rel_const[0]) / rel_const[1]
    t = pd.DataFrame(X, index=shapes.index, columns=FLIGHT)
    t['n'], t['abs_speed'] = n, sp

    d = df.reset_index(drop=True)
    Xf = calc._build_features(d)
    for name in ['xwoba', 'miss', 'chase']:
        d['pred_' + name] = calc._predict_safe(calc.models[name], Xf.reindex(columns=calc.metadata[name]['features'], fill_value=0))
    in_zone = (d['plate_x'].abs() <= 8.5 / 12) & d['plate_z'].between(d['sz_bot'], d['sz_top'])
    d['ozone'] = ~in_zone.fillna(False)
    d['miss'] = d['description'].isin(MISS)
    d['chase'] = d['description'].isin(SWING) & d['ozone']
    g = d.groupby(['pitcher', 'pitch_type'])
    agg = g.agg(px=('pred_xwoba', 'mean'), pm=('pred_miss', 'mean'), pc=('pred_chase', 'mean'),
                swstr=('miss', 'mean'), oz=('ozone', 'sum'), ch=('chase', 'sum'),
                xwcon=('estimated_woba_using_speedangle', 'mean'))
    agg['chase_pct'] = agg['ch'] / agg['oz'].where(agg['oz'] > 0)
    t = t.join(agg[['px', 'pm', 'pc', 'swstr', 'chase_pct', 'xwcon']], how='inner')
    q = t[t['n'] >= MIN_N]
    z = lambda c, s: s * (t[c] - q[c].mean()) / q[c].std(ddof=0)
    t['score'] = (z('px', -1) + z('pm', 1) + z('pc', 1)) / 3
    return t


def directions(t, rng):
    """Direction to the better / random / worse third of K other-pitcher neighbours."""
    e = t[t['n'] >= MIN_N]
    X, pitcher, score = e[FLIGHT].values, e.index.get_level_values(0).values, e['score'].values
    _, cand = NearestNeighbors(n_neighbors=K + 15).fit(X).kneighbors(X)
    out = {}
    for i, key in enumerate(e.index):
        nb = np.array([j for j in cand[i] if pitcher[j] != pitcher[i]][:K])
        order = nb[np.argsort(-score[nb])]
        third = len(nb) // 3
        out[key] = {'better': X[order[:third]].mean(0) - X[i],
                    'worse': X[order[-third:]].mean(0) - X[i],
                    'random': X[rng.choice(nb, third, replace=False)].mean(0) - X[i]}
    return out


def pair_rows(t0, t1, dirs, tag):
    rows = []
    for key in t0.index.intersection(t1.index):
        if key not in dirs or t1.loc[key, 'n'] < MIN_N:
            continue
        delta = t1.loc[key, FLIGHT].values.astype(float) - t0.loc[key, FLIGHT].values.astype(float)
        r = {'pitcher': key[0], 'pitch_type': key[1], 'pair': tag,
             'w': 2 / (1 / t0.loc[key, 'n'] + 1 / t1.loc[key, 'n']), 'move': np.linalg.norm(delta),
             'd_speed': t1.loc[key, 'abs_speed'] - t0.loc[key, 'abs_speed']}
        for kind, v in dirs[key].items():
            r['prog_' + kind] = float(delta @ v / (v @ v)) if v @ v > 1e-9 else np.nan
        for o in ['swstr', 'chase_pct', 'xwcon']:
            r[o + '0'], r['d_' + o] = t0.loc[key, o], t1.loc[key, o] - t0.loc[key, o]
        rows.append(r)
    return rows


def fit(df, outcome, prog):
    d = df.dropna(subset=[prog, 'd_' + outcome, outcome + '0'])
    lo, hi = d[prog].quantile([0.01, 0.99])
    d = d.assign(progress=d[prog].clip(lo, hi))
    m = smf.wls(f"d_{outcome} ~ progress + move + {outcome}0 + d_speed + C(pitch_type) + C(pair)",
                data=d, weights=d['w']).fit(cov_type='cluster', cov_kwds={'groups': d['pitcher']})
    ci = m.conf_int().loc['progress']
    return {'n': int(m.nobs), 'beta': m.params['progress'], 'lo': ci[0], 'hi': ci[1], 'p': m.pvalues['progress']}


def main(folder):
    folder = Path(folder)
    mc = load_tda_model(_ROOT / 'models' / 'tda_mapper_model.pkl')
    calc = StuffPlusCalculator()
    base = pd.read_parquet(folder / 'statcast_2025.parquet')
    sh, _ = season_shapes(base, list(mc['stuff_columns']))
    sp = sh['release_speed'].astype(float)
    rel = sp - sp.groupby(level=0).transform('max')
    rel_const = (rel.mean(), rel.std())
    years = sorted({y for p in PAIRS for y in p})
    tables = {y: season_table(pd.read_parquet(folder / f'statcast_{y}.parquet'), mc, calc, rel_const) for y in years}
    rng = np.random.default_rng(SEED)
    rows = []
    for (a, b), kind in PAIRS.items():
        rows += pair_rows(tables[a], tables[b], directions(tables[a], rng), f'{a}-{b}')
    df = pd.DataFrame(rows)
    df.to_csv(_ROOT / 'data' / 'pitch_development.csv', index=False)

    print(f"pitch shapes in consecutive seasons (n >= {MIN_N} both): "
          + ", ".join(f"{p}: {(df['pair'] == p).sum()}" for p in df['pair'].unique()))
    res = []
    for scope, pairs in [('primary', [f'{a}-{b}' for (a, b), k in PAIRS.items() if k == 'primary']),
                         ('secondary', [f'{a}-{b}' for (a, b), k in PAIRS.items() if k == 'secondary'])]:
        sub = df[df['pair'].isin(pairs)]
        for outcome in ['swstr', 'chase_pct', 'xwcon']:
            for prog in ['prog_better', 'prog_random', 'prog_worse']:
                res.append({'scope': scope, 'outcome': outcome, 'direction': prog[5:], **fit(sub, outcome, prog)})
    res = pd.DataFrame(res)
    res.to_csv(_ROOT / 'data' / 'pitch_development_results.csv', index=False)
    print(res.round({'beta': 4, 'lo': 4, 'hi': 4, 'p': 4}).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
