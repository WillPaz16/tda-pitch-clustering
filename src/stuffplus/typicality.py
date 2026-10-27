"""
Typicality test: do pitches with crowded, typical shapes get fewer whiffs than
their physics alone predicts?

PRE-SPECIFIED DESIGN (written and committed before the analysis was run;
docs/DISCOVERY_FINDINGS.md records the result, whatever it is). Motivated by
the exploratory pattern in pitch_development.py (moving toward the
neighbourhood went with fewer whiffs); this is the confirmatory test.

Unit. A (pitcher, pitch_type) shape in season s with >= MIN_N pitches.
Shape space. Flight features as in label_sensitivity.py / pitch_development.py
  (velocity relative to the pitcher's fastest shape, pfx_x, pfx_z, spin rate,
  spin axis cos/sin; LHP mirrored; fixed standardization).
Typicality. typ = -log(mean distance to the K nearest shapes from OTHER
  pitchers with >= MIN_N pitches in the same season), z-scored within season.
  Higher = more crowded neighbourhood = more typical shape.
Physics control. The mean per-pitch predicted swinging-strike probability
  from the physics-only CatBoost model (pm); the model was trained on
  2022-24, so it is out-of-sample for 2025 and 2026.
Outcome (actual). Primary: SwStr% (swinging strikes per pitch).
  Secondary: chase% (control: predicted chase), xwOBA on contact (control:
  predicted xwOBA).
Model. WLS: outcome ~ typ + control + pitch-type FE + season FE; weights = n;
  standard errors clustered by pitcher.
Primary seasons: 2025 and 2026. Secondary: 2023 and 2024 (in the model's
  training years, so the control is in-sample there).
Hypothesis. beta(typ) < 0 for SwStr%: more typical shapes get fewer whiffs
  than their physics predicts.
Robustness. K = 10 and 30 (primary seasons, SwStr%).
Caveats. Observational; typicality may partly proxy for physics the model
  misses (a nonlinear misfit), and for location/sequencing, which no shape
  model sees.

Run: python3 src/stuffplus/typicality.py <dir with statcast_<year>.parquet>
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
from pitch_development import season_table, MIN_N

K = 20
PRIMARY, SECONDARY = [2025, 2026], [2023, 2024]
OUTCOMES = [('swstr', 'pm'), ('chase_pct', 'pc'), ('xwcon', 'px')]


def typicality(t, k):
    e = t[t['n'] >= MIN_N].copy()
    X, pitcher = e[FLIGHT].values, e.index.get_level_values(0).values
    dist, idx = NearestNeighbors(n_neighbors=k + 15).fit(X).kneighbors(X)
    typ = []
    for i in range(len(e)):
        keep = [d for d, j in zip(dist[i], idx[i]) if pitcher[j] != pitcher[i]][:k]
        typ.append(-np.log(np.mean(keep)))
    typ = np.array(typ)
    e['typ'] = (typ - typ.mean()) / typ.std()
    return e


def fit(d, outcome, control):
    d = d.dropna(subset=[outcome, control, 'typ'])
    m = smf.wls(f"{outcome} ~ typ + {control} + C(pitch_type) + C(season)", data=d, weights=d['n']).fit(
        cov_type='cluster', cov_kwds={'groups': d['pitcher']})
    ci = m.conf_int().loc['typ']
    return {'n': int(m.nobs), 'beta': m.params['typ'], 'lo': ci[0], 'hi': ci[1], 'p': m.pvalues['typ']}


def main(folder):
    folder = Path(folder)
    mc = load_tda_model(_ROOT / 'models' / 'tda_mapper_model.pkl')
    calc = StuffPlusCalculator()
    sh, _ = season_shapes(pd.read_parquet(folder / 'statcast_2025.parquet'), list(mc['stuff_columns']))
    sp = sh['release_speed'].astype(float)
    rel = sp - sp.groupby(level=0).transform('max')
    rel_const = (rel.mean(), rel.std())
    tables = {y: season_table(pd.read_parquet(folder / f'statcast_{y}.parquet'), mc, calc, rel_const)
              for y in PRIMARY + SECONDARY}

    res = []
    for k in [K, 10, 30]:
        frames = []
        for y, t in tables.items():
            e = typicality(t, k).reset_index()
            e['season'] = y
            frames.append(e)
        allx = pd.concat(frames)
        if k == K:
            allx.to_csv(_ROOT / 'data' / 'typicality.csv', index=False)
            scopes, outs = [('primary', PRIMARY), ('secondary', SECONDARY)], OUTCOMES
        else:
            scopes, outs = [('primary', PRIMARY)], OUTCOMES[:1]
        for scope, yrs in scopes:
            for o, c in outs:
                res.append({'scope': scope, 'k': k, 'outcome': o, **fit(allx[allx['season'].isin(yrs)], o, c)})
    res = pd.DataFrame(res)
    res.to_csv(_ROOT / 'data' / 'typicality_results.csv', index=False)
    print(res.round({'beta': 4, 'lo': 4, 'hi': 4, 'p': 4}).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
