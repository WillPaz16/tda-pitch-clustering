# Reorganization notes (2026-08-17)

Flat directory reorganized into a standard research-repo layout. Baseline
(pre-reorg) state is preserved in git history (first commit).

## Layout
- `src/tda/` — TDA/Mapper clustering pipeline (core methodology)
- `src/stuffplus/` — Stuff+ / xwOBA CatBoost modeling (`stuff_plus_calculator.py` is a shared library imported by other scripts)
- `src/validation/` — pitcher consistency, variance, ANOVA, predictive comparisons
- `src/data_fetch/` — Statcast/Savant data pulling and integration
- `notebooks/` — the 3 source notebooks (TDA model build, Pro Stuff+, College Stuff+)
- `models/` — trained model artifacts (.cbm/.pkl + metadata)
- `data/` — all CSVs/JSON (raw, intermediate, and output), plus `data/mappings/`
- `results/` — figures and interactive HTML visualizations
- `docs/` — README, requirements.txt, this file

## Deleted (confirmed dead/duplicate/superseded, still recoverable from the baseline git commit)
- `test_constrained.py` — mocked/fake correlation value, not real logic
- `stuff_weighted_consistency.py` — empty file
- `demo_cluster_distances.py` — print-only demo
- `penalty_function_analysis.py` — exploratory scratch
- `validation_summary.py` — hardcoded numbers, not data-driven
- `query_cluster_distances.py` — graph loader was a hardcoded mock, never finished
- `fetch_real_statcast.py` — had a duplicate function definition bug; superseded by `fetch_real_statcast_new.py`
- `fetch_advanced_metrics_clean.py` — truncated/malformed duplicate of `fetch_advanced_metrics.py`
- `predictive_model_comparison_only.py` — explicit subset of `predictive_model_comparison.py`
- `analyze_weights.py`, `analyze_correlation.py` — read CSVs that no longer exist anywhere in the repo

## Kept despite overlap
- `src/validation/cluster_outcome_validation.py` subsumes `fetch_real_statcast_new.py` + `integrate_real_data.py` + `anova_real_data.py` as an all-in-one script. Kept as a possible end-to-end reproduction entry point — not deleted.
- `src/tda/classify_pitches_to_csv.py` vs `src/tda/assign_pitch_stuffplus_clusters.py` — different scope (classify-only vs classify+Stuff+ combined), both kept.

## Known pre-existing bug (not touched — flagged, not fixed)
`src/data_fetch/fetch_advanced_metrics.py` has a syntax error: orphaned code
after the `if __name__ == "__main__": main()` block (lines ~120+), leftover
from an editing mistake predating this reorg. It does not currently run.
Needs a decision on whether that trailing fragment belongs to a missing
function or should just be deleted.

## Known follow-up: notebooks still assume old flat-layout paths
`notebooks/*.ipynb` load/save models and CSVs using bare filenames (e.g.
`'tda_mapper_model.pkl'`, `'chase_model.cbm'`) that assumed the working
directory was the old project root. They were NOT edited as part of this
pass (safer to update inside Jupyter directly). When next opened, either:
- run them from the repo root and update paths to `models/...`, `data/...`, or
- add a path-setup cell at the top pointing to the new `models/`/`data/mappings/` locations.

## Scripts updated to use path-independent defaults
All moved `.py` scripts now compute a `_ROOT`/`_DEFAULT_DATA_DIR`/etc. via
`Path(__file__).resolve().parents[...]` so they work regardless of the
working directory they're invoked from, instead of relying on bare relative
filenames as before.
