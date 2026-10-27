"""
Sample sizes behind the pitch shapes, and whether the Mapper features survive
a minimum pitch count. Saves the numbers the paper's Data section quotes
(the original table in DISCOVERY_FINDINGS.md was computed ad hoc and never
saved).

  data/shape_counts_2025.json  pitches per shape (all shapes; slow < 70 mph)
  data/sample_size_check.csv   one PCA Mapper fit at the chosen parameters
                               per minimum count: shapes, slow shapes,
                               components, slow isolated, CU->FF ordering rho

Shapes are rebuilt from the season pull (tda_classifier.season_shapes), so
counts can differ by a few from the saved model's 3,943 (Statcast relabels
pitches between pulls).

Run: python3 src/tda/sample_size_check.py statcast_2025.parquet
"""
import json
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tda_classifier import load_tda_model, season_shapes
from mapper_stability import make_lens, fit_graph, features, BASE, SLOW_MPH

MIN_COUNTS = [1, 10, 20]


def main(path):
    mc = load_tda_model(_ROOT / 'models' / 'tda_mapper_model.pkl')
    cols = list(mc['stuff_columns'])
    shapes, n = season_shapes(pd.read_parquet(path), cols)
    speed = shapes['release_speed'].values
    slow = speed < SLOW_MPH
    counts = {'n_shapes': int(len(n)), 'median_pitches': float(n.median()),
              'frac_lt5': float((n < 5).mean()), 'frac_lt20': float((n < 20).mean()),
              'n_slow': int(slow.sum()), 'slow_median_pitches': float(n[slow].median()),
              'slow_q25_pitches': float(n[slow].quantile(0.25))}
    (_ROOT / 'data' / 'shape_counts_2025.json').write_text(json.dumps(counts, indent=1))
    print(counts)

    rows = []
    for k in MIN_COUNTS:
        keep = (n >= k).values
        X = StandardScaler().fit_transform(shapes[cols].values[keep].astype(float))
        ptype = shapes.index.get_level_values('pitch_type').values[keep].astype(str)
        G, nodes = fit_graph(X, make_lens(X, 'pca'), **BASE)
        f = features(G, nodes, speed[keep], ptype)
        rows.append({'min_n': k, 'shapes': int(keep.sum()), 'slow': int((speed[keep] < SLOW_MPH).sum()),
                     'n_components': f['n_components'], 'slow_isolated': f['slow_isolated'],
                     'ordering_rho': f['ordering_rho']})
    out = pd.DataFrame(rows)
    out.to_csv(_ROOT / 'data' / 'sample_size_check.csv', index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
