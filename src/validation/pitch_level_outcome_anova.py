"""
Pitch-level outcome ANOVA across Mapper clusters (replaces the
anova_real_data.py / integrate_real_data.py pipeline).

Why a rebuild, not a patch (docs/METHODOLOGY_REVIEW.md item 6):
  - the old pipeline copied one per-pitcher outcome value (April 2023) onto
    every one of that pitcher's pitches: 11,441 rows but only 187
    independent values -- pseudo-replication that inflated F.
  - it joined 2023 outcomes onto 2026 pitches labeled by the pre-refit model.

Here one Statcast pull supplies both the pitches and their outcomes, and each
outcome is measured on the pitch itself:
  whiff  = swinging strike, among swings
  chase  = swing, among pitches outside the batter's real sz_top/sz_bot zone
  gb/fb  = ground ball / fly ball, among balls in play
  xwoba  = estimated_woba_using_speedangle, among balls in play
(Velocity is left out: clusters are built from it, so testing it is circular.)

Each metric is tested under both labelings:
  centroid_primary -- nearest centroid overall (the old cluster_id)
  mapper_primary   -- nearest centroid among the pitch's own Mapper clusters;
                      DBSCAN-noise pitches excluded
and at two levels:
  pitch   -- every pitch an observation (pitches from one pitcher are still
             correlated, so p-values here are optimistic)
  pitcher -- one mean per (pitcher, cluster) cell, so a pitcher counts once
             per cluster
eta^2 (share of outcome variance explained by cluster) is reported alongside
p, since with tens of thousands of pitches almost anything is "significant".
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pybaseball import statcast
from scipy import stats
from scipy.spatial.distance import cdist

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'src' / 'tda'))
from tda_classifier import (load_tda_model, prepare_pitch_features, scaled_cluster_centroids,
                            build_cover_index, member_clusters, mapper_primary_labels)

MIN_PITCHES_PER_GROUP = 30
MIN_PITCHES_PER_CELL = 10     # pitcher x cluster cell, pitcher-level test
MIN_CELLS_PER_GROUP = 5
SWINGS = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt', 'foul',
          'foul_bunt', 'foul_tip', 'bunt_foul_tip', 'hit_into_play'}
WHIFFS = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}


def label_pitches(df, mc):
    cols = ['release_speed', 'pfx_x', 'pfx_z', 'release_spin_rate', 'spin_axis',
            'release_extension', 'release_pos_x', 'release_pos_z', 'p_throws']
    feats = prepare_pitch_features(df, cols)
    ok = feats[mc['stuff_columns']].notna().all(axis=1)
    df, feats = df[ok].copy(), feats[ok]
    X = mc['scaler'].transform(feats[mc['stuff_columns']].values.astype(np.float64))

    Xc, cs = scaled_cluster_centroids(mc, mc['stuff_columns'])
    df['centroid_primary'] = cs['cluster'].values[np.argmin(cdist(X, Xc), axis=1)]
    df['mapper_primary'] = mapper_primary_labels(X, member_clusters(X, build_cover_index(mc)), mc)
    return df


def add_outcomes(df):
    desc = df['description']
    swing = desc.isin(SWINGS)
    in_zone = (df['plate_x'].abs() <= 8.5 / 12) & df['plate_z'].between(df['sz_bot'], df['sz_top'])
    bip = df['bb_type'].notna()
    df['whiff'] = np.where(swing, desc.isin(WHIFFS).astype(float), np.nan)
    df['chase'] = np.where(~in_zone & df['plate_z'].notna(), swing.astype(float), np.nan)
    df['gb'] = np.where(bip, (df['bb_type'] == 'ground_ball').astype(float), np.nan)
    df['fb'] = np.where(bip, (df['bb_type'] == 'fly_ball').astype(float), np.nan)
    df['xwoba'] = np.where(bip, df['estimated_woba_using_speedangle'], np.nan)
    return df


def anova(groups):
    groups = [np.asarray(g, float) for g in groups]
    f, p = stats.f_oneway(*groups)
    allv = np.concatenate(groups)
    ss_total = ((allv - allv.mean()) ** 2).sum()
    ss_between = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
    return f, p, ss_between / ss_total, len(groups), len(allv)


def run(df, metrics, labelings):
    rows = []
    for label in labelings:
        for m in metrics:
            d = df[[label, 'pitcher', m]].dropna()
            g = d.groupby(label)[m]
            pitch_groups = [v.values for _, v in g if len(v) >= MIN_PITCHES_PER_GROUP]

            cells = d.groupby([label, 'pitcher'])[m].agg(['mean', 'size'])
            cells = cells[cells['size'] >= MIN_PITCHES_PER_CELL].reset_index()
            cell_groups = [v['mean'].values for _, v in cells.groupby(label) if len(v) >= MIN_CELLS_PER_GROUP]

            for level, groups in (('pitch', pitch_groups), ('pitcher', cell_groups)):
                if len(groups) < 2:
                    continue
                f, p, eta2, k, n = anova(groups)
                rows.append(dict(labeling=label, level=level, metric=m, F=f, p=p, eta2=eta2, groups=k, n=n))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2025-06-01')
    ap.add_argument('--end', default='2025-06-30')
    args = ap.parse_args()

    df = statcast(args.start, args.end).reset_index(drop=True)
    print(f"Fetched {len(df)} pitches, {args.start} to {args.end}")
    mc = load_tda_model(_ROOT / 'models' / 'tda_mapper_model.pkl')
    df = add_outcomes(label_pitches(df, mc))
    noise = df['mapper_primary'].isna().mean()
    agree = (df['mapper_primary'] == df['centroid_primary']).mean()
    print(f"Labeled {len(df)} pitches | DBSCAN noise {noise:.1%} | labelings agree on {agree:.1%}")

    res = run(df, ['whiff', 'chase', 'gb', 'fb', 'xwoba'], ['centroid_primary', 'mapper_primary'])
    res['bonferroni_sig'] = res['p'] < 0.05 / len(res)
    pd.set_option('display.width', 160)
    print(res.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    print(f"\nBonferroni alpha over {len(res)} tests: {0.05 / len(res):.2g}")

    out = _ROOT / 'data' / 'pitch_level_anova_results.csv'
    res.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
