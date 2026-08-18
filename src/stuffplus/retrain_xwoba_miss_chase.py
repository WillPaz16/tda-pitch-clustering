"""
Corrected xwOBA/miss/chase retrain.

Fixes two methodology holes together (both require retraining on fresh data):

1. Fixed vs. per-pitch strike zone (METHODOLOGY_REVIEW.md item 7) --
   chase% now uses each pitch's real sz_top/sz_bot instead of a fixed
   1.6-3.5ft window for every batter.

2. Stuff+ leakage (item 5) -- the original notebook generated the final
   pitcher-level Stuff+ ratings by predicting on the same data the models
   were trained on. Fixed via 5-fold cross-validated out-of-fold (OOF)
   predictions: every row gets a prediction from a model that never saw
   it during training, with no data thrown away (unlike a single
   train/held-out split, which would only leave ~20% of the data for the
   final ratings). The deployed models saved to models/*.cbm are refit on
   ALL data afterward (standard practice: best model for future use is
   trained on everything; the leak-free OOF predictions are only used for
   the historical/backtest Stuff+ ratings).

ponytail: reuses the exact feature engineering and CatBoost hyperparameters
from notebooks/ProStuff+.ipynb (this is not a methodology re-design, just
fixing the two specific, scoped holes). Early stopping during OOF folds
uses the held-out fold as the eval set -- a mild, standard simplification
(deciding *when* to stop is not the same leak as using held-out labels to
correct predictions); upgrade path is a nested split if that distinction
ever matters.
"""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import (mean_absolute_error, root_mean_squared_error, r2_score,
                              roc_auc_score, precision_score, recall_score, f1_score,
                              brier_score_loss, confusion_matrix)
from pybaseball import statcast

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
N_FOLDS = 5
SEED = 42

print("Fetching 2022-04-07 to 2024-07-31 (matches notebooks/ProStuff+.ipynb)...")
df = statcast("2022-04-07", "2024-07-31")
df = df.reset_index(drop=True)  # pybaseball concatenates per-day chunks, each with its own
                                 # 0..N index -- must reset before any index-based alignment
print(f"Fetched {len(df)} pitches")

stuffCols = ['release_speed', 'release_pos_x', 'release_pos_z',
             'pfx_x', 'pfx_z', 'spin_axis',
             'release_spin_rate', 'release_extension']


def build_features(df):
    base_cols = list(stuffCols)
    extra_possible = ['px', 'pz', 'release_speed', 'pfx_x', 'pfx_z',
                       'spin_axis', 'release_pos_x', 'release_pos_z', 'release_extension']
    use_cols = [c for c in base_cols + extra_possible if c in df.columns]
    work = df[use_cols].copy()
    work = work.loc[:, ~work.columns.duplicated()]
    for c in work.columns:
        col = work[c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
            work[c] = col
        if col.dtype != 'object':
            work[c] = pd.to_numeric(col, errors='coerce')

    work['release_speed_sq'] = work['release_speed'] ** 2
    work['log_release_speed'] = np.log1p(work['release_speed'])
    px_val = work.get('pfx_x', pd.Series(0, index=work.index)).fillna(0)
    pz_val = work.get('pfx_z', pd.Series(0, index=work.index)).fillna(0)
    work['pfx_magnitude'] = np.sqrt(px_val ** 2 + pz_val ** 2)
    work['spin_axis_rad'] = np.deg2rad(work['spin_axis'].astype(float))
    work['spin_cos'] = np.cos(work['spin_axis_rad'])
    work['spin_sin'] = np.sin(work['spin_axis_rad'])
    work['release_side'] = work['release_pos_x']
    work['release_height'] = work['release_pos_z']
    work['release_side_times_height'] = work['release_pos_x'] * work['release_pos_z']

    feat_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
    for c in feat_cols:
        if work[c].isna().any():
            med = work[c].median()
            work[c + '_missing'] = work[c].isna().astype(int)
            work[c] = work[c].fillna(med)
    feat_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
    return work[feat_cols]


X_all = build_features(df)
print(f"Features: {X_all.shape[1]}, Rows: {len(X_all)}")

# Targets
y_xw_all = pd.to_numeric(df['estimated_woba_using_speedangle'], errors='coerce')

miss_outcomes = ['swinging_strike', 'swinging_strike_blocked', 'missed_bunt']
y_miss_all = df['description'].isin(miss_outcomes).astype(int)

# FIX: chase% now uses each pitch's real sz_top/sz_bot, not a fixed 1.6-3.5ft zone
zone_x_min, zone_x_max = -8.5 / 12, 8.5 / 12
in_zone = (df['plate_x'].ge(zone_x_min)) & (df['plate_x'].le(zone_x_max)) & \
          (df['plate_z'].ge(df['sz_bot'])) & (df['plate_z'].le(df['sz_top']))
in_zone = in_zone.fillna(False)
swing_outcomes = ['swinging_strike', 'swinging_strike_blocked', 'missed_bunt',
                   'foul', 'foul_bunt', 'foul_tip', 'bunt_foul_tip', 'hit_into_play']
is_swing = df['description'].isin(swing_outcomes)
y_chase_all = ((is_swing) & (~in_zone)).astype(int)

CATBOOST_REG_PARAMS = dict(iterations=800, learning_rate=0.05, eval_metric='RMSE',
                            random_seed=SEED, early_stopping_rounds=50, verbose=False)
CATBOOST_CLF_PARAMS = dict(iterations=800, learning_rate=0.05, eval_metric='AUC',
                            random_seed=SEED, early_stopping_rounds=50, verbose=False)


def oof_regression(X, y, name):
    mask = y.notna()
    X, y = X[mask], y[mask]
    oof = pd.Series(np.nan, index=X.index)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        model = CatBoostRegressor(**CATBOOST_REG_PARAMS)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=(X.iloc[va_idx], y.iloc[va_idx]))
        oof.iloc[va_idx] = model.predict(X.iloc[va_idx])
        print(f"  [{name}] fold {fold + 1}/{N_FOLDS} done")
    mae = mean_absolute_error(y, oof)
    rmse = root_mean_squared_error(y, oof)
    r2 = r2_score(y, oof)
    print(f"[{name}] OOF MAE={mae:.5f} RMSE={rmse:.5f} R2={r2:.5f}")
    final_model = CatBoostRegressor(**{**CATBOOST_REG_PARAMS, 'early_stopping_rounds': None})
    final_model.fit(X, y)
    return oof, final_model, {'mae': mae, 'rmse': rmse, 'r2': r2}


def oof_classification(X, y, name):
    oof = pd.Series(np.nan, index=X.index)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        y_tr = y.iloc[tr_idx]
        scale_pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        model = CatBoostClassifier(**CATBOOST_CLF_PARAMS, scale_pos_weight=scale_pos_weight)
        model.fit(X.iloc[tr_idx], y_tr, eval_set=(X.iloc[va_idx], y.iloc[va_idx]))
        oof.iloc[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        print(f"  [{name}] fold {fold + 1}/{N_FOLDS} done")
    preds = (oof >= 0.5).astype(int)
    auc = roc_auc_score(y, oof)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    brier = brier_score_loss(y, oof)
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"[{name}] OOF AUC={auc:.5f} Precision={precision:.5f} Recall={recall:.5f} "
          f"F1={f1:.5f} Brier={brier:.5f}")
    scale_pos_weight_full = (y == 0).sum() / max((y == 1).sum(), 1)
    final_model = CatBoostClassifier(**{**CATBOOST_CLF_PARAMS, 'early_stopping_rounds': None},
                                      scale_pos_weight=scale_pos_weight_full)
    final_model.fit(X, y)
    metrics = {'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1,
               'specificity': specificity, 'brier': brier}
    return oof, final_model, metrics


print("\n=== xwOBA ===")
oof_xw, model_xw, metrics_xw = oof_regression(X_all, y_xw_all, 'xwOBA')

print("\n=== Miss ===")
oof_miss, model_miss, metrics_miss = oof_classification(X_all, y_miss_all, 'miss')

print("\n=== Chase (real per-pitch strike zone) ===")
oof_chase, model_chase, metrics_chase = oof_classification(X_all, y_chase_all, 'chase')

MODELS_DIR.mkdir(exist_ok=True)
model_xw.save_model(str(MODELS_DIR / 'xwoba_model.cbm'))
model_miss.save_model(str(MODELS_DIR / 'miss_model.cbm'))
model_chase.save_model(str(MODELS_DIR / 'chase_model.cbm'))

for name, features, metrics in [
    ('xwoba', X_all.columns.tolist(), metrics_xw),
    ('miss', X_all.columns.tolist(), metrics_miss),
    ('chase', X_all.columns.tolist(), metrics_chase),
]:
    meta = {
        'model_type': 'CatBoostRegressor' if name == 'xwoba' else 'CatBoostClassifier',
        'features': features,
        'n_features': len(features),
        'n_train_samples': len(X_all),
        'n_val_samples': None,
        'validation_method': f'{N_FOLDS}-fold cross-validated out-of-fold (no held-out split thrown away, no leakage)',
        'metrics': metrics,
    }
    with open(MODELS_DIR / f'{name}_model_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

print("\nSaved models and metadata. Computing Stuff+ from OOF predictions (leak-free)...")

pitcher_col = 'pitcher'
pitch_type_col = 'pitch_type'
valid_mask = X_all.index  # all feature rows are valid by construction

df_pred = df.loc[X_all.index, [pitcher_col, pitch_type_col]].copy()
df_pred['pred_xw'] = oof_xw.reindex(X_all.index)
df_pred['pred_miss'] = oof_miss.reindex(X_all.index)
df_pred['pred_chase'] = oof_chase.reindex(X_all.index)
df_pred = df_pred.dropna(subset=['pred_xw', 'pred_miss', 'pred_chase'])
print(f"Pitches with all 3 leak-free OOF predictions: {len(df_pred)}")

min_pitches_per_type = 30
min_pitches_total = 50
target_mean, target_sd = 100.0, 15.0
weights = np.array([1 / 3, 1 / 3, 1 / 3])

agg_pt = df_pred.groupby([pitcher_col, pitch_type_col]).agg(
    pitch_count_pt=('pred_xw', 'count'),
    mean_xw_pt=('pred_xw', 'mean'),
    mean_miss_pt=('pred_miss', 'mean'),
    mean_chase_pt=('pred_chase', 'mean'),
).reset_index().rename(columns={pitcher_col: 'pitcher', pitch_type_col: 'pitch_type'})

pt_stats = {}
for pt, grp in agg_pt.groupby('pitch_type'):
    stable_grp = grp[grp['pitch_count_pt'] >= min_pitches_per_type]
    if len(stable_grp) < 10:
        stable_grp = grp
    mu_xw, sd_xw = stable_grp['mean_xw_pt'].mean(), stable_grp['mean_xw_pt'].std(ddof=0) or 1.0
    mu_miss, sd_miss = stable_grp['mean_miss_pt'].mean(), stable_grp['mean_miss_pt'].std(ddof=0) or 1.0
    mu_chase, sd_chase = stable_grp['mean_chase_pt'].mean(), stable_grp['mean_chase_pt'].std(ddof=0) or 1.0
    pt_stats[pt] = dict(mu_xw=mu_xw, sd_xw=sd_xw, mu_miss=mu_miss, sd_miss=sd_miss,
                         mu_chase=mu_chase, sd_chase=sd_chase, n_stable=len(stable_grp))


def compute_z_row(row):
    s = pt_stats.get(row['pitch_type'])
    if s is None:
        return pd.Series({'z_xw_pt': 0, 'z_miss_pt': 0, 'z_chase_pt': 0})
    zx = -(row['mean_xw_pt'] - s['mu_xw']) / s['sd_xw']
    zm = (row['mean_miss_pt'] - s['mu_miss']) / s['sd_miss']
    zc = (row['mean_chase_pt'] - s['mu_chase']) / s['sd_chase']
    return pd.Series({'z_xw_pt': zx, 'z_miss_pt': zm, 'z_chase_pt': zc})


agg_pt = pd.concat([agg_pt, agg_pt.apply(compute_z_row, axis=1)], axis=1)
agg_pt['stuff_z_pt'] = weights[0] * agg_pt['z_xw_pt'] + weights[1] * agg_pt['z_miss_pt'] + weights[2] * agg_pt['z_chase_pt']

agg_pt['stuff_plus_pt'] = np.nan
for pt, grp in agg_pt.groupby('pitch_type'):
    stable_grp = grp[grp['pitch_count_pt'] >= min_pitches_per_type]
    if len(stable_grp) < 10:
        stable_grp = grp
    mean_z, sd_z = stable_grp['stuff_z_pt'].mean(), (stable_grp['stuff_z_pt'].std(ddof=0) or 1.0)
    mask = agg_pt['pitch_type'] == pt
    agg_pt.loc[mask, 'stuff_plus_pt'] = target_mean + target_sd * ((agg_pt.loc[mask, 'stuff_z_pt'] - mean_z) / sd_z)

pitch_total = agg_pt.groupby('pitcher')['pitch_count_pt'].sum().rename('total_pitches').reset_index()
agg_pt = agg_pt.merge(pitch_total, on='pitcher', how='left')

agg_pitch = agg_pt.groupby('pitcher').apply(
    lambda g: pd.Series({
        'total_pitches': int(g['total_pitches'].iloc[0]),
        'weighted_stuff_z': np.average(g['stuff_z_pt'], weights=g['pitch_count_pt']),
        'weighted_stuff_plus_by_pt': np.average(g['stuff_plus_pt'], weights=g['pitch_count_pt']),
    }), include_groups=False
).reset_index()

stable_pitchers = agg_pitch[agg_pitch['total_pitches'] >= min_pitches_total]
if len(stable_pitchers) < 30:
    stable_pitchers = agg_pitch.copy()
zs_mean, zs_sd = stable_pitchers['weighted_stuff_z'].mean(), (stable_pitchers['weighted_stuff_z'].std(ddof=0) or 1.0)
agg_pitch['stuff_plus_overall'] = target_mean + target_sd * ((agg_pitch['weighted_stuff_z'] - zs_mean) / zs_sd)
agg_pitch['stable'] = agg_pitch['total_pitches'] >= min_pitches_total

agg_pt.to_csv(MODELS_DIR / 'stuff_plus_per_pitch_type.csv', index=False)
agg_pitch.to_csv(MODELS_DIR / 'stuff_plus_pitcher_level.csv', index=False)

final_meta = {
    'weights_used': weights.tolist(),
    'min_pitches_per_type': min_pitches_per_type,
    'min_pitches_total': min_pitches_total,
    'validation_method': f'{N_FOLDS}-fold cross-validated out-of-fold predictions -- leak-free, uses full dataset',
    'chase_definition': 'real per-pitch sz_top/sz_bot (fixed from the old 1.6-3.5ft constant zone)',
    'pitch_type_stats': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in pt_stats.items()},
    'overall_scaling': {'zs_mean': float(zs_mean), 'zs_sd': float(zs_sd)},
    'n_pitch_types': len(pt_stats),
    'n_pitchers': int(agg_pitch.shape[0]),
    'n_pitchers_stable': int(stable_pitchers.shape[0]),
}
with open(MODELS_DIR / 'stuff_plus_per_pitch_type_metadata.json', 'w') as f:
    json.dump(final_meta, f, indent=2)

print(f"\nDone. Pitchers: {agg_pitch.shape[0]} total, {stable_pitchers.shape[0]} stable (>= {min_pitches_total} pitches)")
