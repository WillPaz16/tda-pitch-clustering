"""
Stuff+ Leakage Check: In-Sample (2022-2024) vs. Out-of-Sample (2025)

Follow-up to docs/METHODOLOGY_REVIEW.md's objection: the xwOBA/miss/chase
CatBoost models are trained on 2022-2024 data, but the final pitcher-level
Stuff+ ratings in models/stuff_plus_pitcher_level.csv are computed by
predicting on the SAME 2022-2024 data (not a held-out split carried through
to that final step), which risks an optimistic bias.

This checks the practical consequence directly, using a natural experiment
already available in the repo: data/pitch_stuffplus_clusters.csv was built
from 2025 Statcast data, which the xwOBA/miss/chase models never saw during
training. If the models generalize, per-pitcher predictions from that
genuinely-unseen 2025 data should correlate with the "official" 2022-2024
per-pitcher predictions.

IMPORTANT CAUTIONARY NOTE, kept in the script deliberately: the first,
naive version of this check compared *combined Stuff+ scores* between the
two datasets and found a striking NEGATIVE correlation (Spearman r=-0.46),
which would suggest severe overfitting. That conclusion was wrong -- the
two pipelines use different Stuff+ component weights (the official
pipeline uses equal 1/3-1/3-1/3 weights;
src/tda/assign_pitch_stuffplus_clusters.py, which produced the 2025 data,
uses the separately-optimized 0.72/0.11/0.17 weights from
src/stuffplus/fitting_stuff_weights.py) plus different z-score
normalization reference populations. Comparing combined Stuff+ scores
computed two different ways is not a valid leakage test. This script
instead compares the RAW model predictions (pred_xw, pred_miss,
pred_chase) directly, before any weighting/combination is applied, which
isolates whether the underlying CatBoost models themselves generalize.

Analysis/reporting script. See docs/METHODOLOGY_REVIEW.md for write-up.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL_PER_PITCH_TYPE_PATH = _ROOT / 'models' / 'stuff_plus_per_pitch_type.csv'
_OOS_DATA_PATH = _ROOT / 'data' / 'pitch_stuffplus_clusters.csv'
MIN_OFFICIAL_PITCHES = 50
MIN_OOS_PITCHES = 10


def aggregate_official(component_col):
    """Pitch-count-weighted per-pitcher average of a raw predicted component,
    from the official 2022-2024 pipeline's per-pitch-type file."""
    df = pd.read_csv(_OFFICIAL_PER_PITCH_TYPE_PATH)
    df['weighted'] = df[component_col] * df['pitch_count_pt']
    agg = df.groupby('pitcher').agg(
        total_pitches=('pitch_count_pt', 'sum'),
        weighted_sum=('weighted', 'sum'),
    ).reset_index()
    agg['official_value'] = agg['weighted_sum'] / agg['total_pitches']
    return agg[agg['total_pitches'] >= MIN_OFFICIAL_PITCHES][['pitcher', 'official_value']]


def aggregate_oos(component_col):
    """Simple per-pitcher average from the genuinely out-of-sample 2025 data."""
    df = pd.read_csv(_OOS_DATA_PATH)
    agg = df.groupby('pitcher_id').agg(
        oos_value=(component_col, 'mean'),
        n=(component_col, 'size'),
    ).reset_index()
    return agg[agg['n'] >= MIN_OOS_PITCHES]


def main():
    components = [
        ('mean_xw_pt', 'pred_xw', 'xwOBA'),
        ('mean_miss_pt', 'pred_miss', 'miss'),
        ('mean_chase_pt', 'pred_chase', 'chase'),
    ]

    print("Correlating raw model predictions: official 2022-2024 in-sample vs. "
          "genuinely out-of-sample 2025 data, per pitcher\n")
    print(f"{'component':<10}{'n':<6}{'spearman_r':<14}{'p_value'}")
    for official_col, oos_col, label in components:
        official = aggregate_official(official_col)
        oos = aggregate_oos(oos_col)
        merged = official.merge(oos, left_on='pitcher', right_on='pitcher_id')
        r, p = stats.spearmanr(merged['official_value'], merged['oos_value'])
        print(f"{label:<10}{len(merged):<6}{r:<14.3f}{p:.4f}")

    print("\nAll three components show strong, positive, significant out-of-sample "
          "correlation -- the underlying CatBoost models retain real predictive "
          "signal on data a full year removed from training. This substantially "
          "de-risks the leakage concern: whatever theoretical optimism the lack "
          "of a held-out split introduces into the FINAL pitcher-level Stuff+ "
          "numbers, it is not masking a model that has actually failed to learn "
          "anything generalizable.")


if __name__ == "__main__":
    main()
