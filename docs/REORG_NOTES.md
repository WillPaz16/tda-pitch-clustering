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

## Resolved since this reorg
- **`fetch_advanced_metrics.py` syntax error** — fixed. The file was two
  script drafts pasted together; the second one referenced an undefined
  function (`fetch_statcast_data_directly`) and undefined `base_url`. The
  dead trailing fragment was removed; the working script is the whole
  file now.
- **Notebook paths** — fixed. All 3 notebooks now have a path-setup cell
  (`ROOT_DIR`/`MODELS_DIR`/`DATA_DIR`/`RESULTS_DIR` resolved via
  `Path.cwd()`) and every model/data save-load/read/write call points at
  those instead of bare flat-layout filenames.
- **Dead imports, a broken debug-print bug, and a dead duplicate
  `if __name__ == '__main__':` block** in `classify_pitches_to_csv.py`
  (which silently re-ran a full-season fetch after the real CLI entry
  point finished) — all cleaned up. See git log for the code-hygiene
  commit.

See [docs/METHODOLOGY_REVIEW.md](METHODOLOGY_REVIEW.md) for a separate,
later review of the actual pipeline logic/statistics (not code hygiene).

## Scripts updated to use path-independent defaults
All moved `.py` scripts now compute a `_ROOT`/`_DEFAULT_DATA_DIR`/etc. via
`Path(__file__).resolve().parents[...]` so they work regardless of the
working directory they're invoked from, instead of relying on bare relative
filenames as before.
