# TDA Pitch Clustering

Graduate thesis project combining the Mapper algorithm (Topological Data
Analysis) with a CatBoost-based Stuff+ model to study MLB pitch quality
and outcomes. See [CLAUDE.md](../CLAUDE.md) at the project root for
current status and next steps, [REORG_NOTES.md](REORG_NOTES.md) for the
codebase layout, and [METHODOLOGY_REVIEW.md](METHODOLOGY_REVIEW.md) for a
critical review of the pipeline logic.

**[Live demo: interactive Mapper graph, colored by pitch type](https://willpaz16.github.io/tda-pitch-clustering/results/mapper_pitch_types_colored.html)**

## Current findings (2026-09-24)

The thesis claim is about the **shape** of pitch-shape space, not outcome
prediction: Statcast's own pitch-type tags explain more outcome variance
than the Mapper clusters. The Mapper graph is fit on one average profile
per pitcher per pitch type (3,943 archetypes, 2025 season), and every
shape feature is tested against null models.

- **Connected continuum.** One giant component (CU → SL/ST → FC → FF → CH)
  plus isolated slow-pitch outliers. Stable across 94–100% of parameter
  grid points and pitcher subsamples, under four different lenses (PCA,
  velocity × vertical break, Isomap, density).
- **No hidden structure.** Persistent homology finds no loops and no more
  gaps than a same-covariance Gaussian. The graph's cycles are cover
  artifacts.
- **Moderately constrained.** Intrinsic dimension is about 6 of 9, which
  is 0.5–0.8 of a dimension below a fair null. The physical features are
  coupled within each pitch type.
- **Tags cut a continuum.** The pitch-type pairs that overlap most in the
  graph (FF/SI, CH/FF, FC/FF, CU/ST, SL/ST) are the ones automated pitch
  classifiers are known to confuse. About 55% of live pitches belong to two
  or more overlapping Mapper nodes.
- **Label reliability depends on node size.** Big, crowded core nodes are
  the least self-consistent, and graph topology adds no reliable signal
  beyond size.

Details: [DISCOVERY_FINDINGS.md](DISCOVERY_FINDINGS.md) (findings) and
[METHODOLOGY_REVIEW.md](METHODOLOGY_REVIEW.md) (pipeline fixes).

## Setup

```bash
python -m pip install -r requirements.txt
```

## Layout

- `src/tda/` — TDA/Mapper clustering pipeline (core methodology)
- `src/stuffplus/` — Stuff+ / xwOBA CatBoost modeling
- `src/validation/` — pitcher consistency, outcome ANOVA, predictive comparisons
- `src/data_fetch/` — Statcast/Savant data pulling and integration
- `notebooks/` — model-building notebooks (TDA Mapper fit, Pro Stuff+, College Stuff+)
- `models/`, `data/`, `results/` — trained model artifacts, CSV/JSON data, and figures/visualizations
- `docs/` — this file, methodology notes, and the presentation source

## Pipeline order

1. `notebooks/TDA_Pitch_Clustering.ipynb` — fits the Mapper model on a season of Statcast data, saves `models/tda_mapper_model.pkl`.
2. `notebooks/ProStuff+.ipynb` / `CollegeStuff+.ipynb` — train the xwOBA/miss/chase CatBoost models and compute pitcher-level Stuff+.
3. `src/tda/classify_pitches_to_csv.py` or `src/tda/assign_pitch_stuffplus_clusters.py` — assign new pitches to existing clusters (and Stuff+, in the latter) via nearest-centroid distance. See [CLASSIFY_PITCHES_README.md](CLASSIFY_PITCHES_README.md).
4. `src/validation/pitch_level_outcome_anova.py` — outcome ANOVA of the clusters against a pitch-type baseline; other `src/validation/*.py` scripts cover consistency and predictive checks.
5. `src/tda/mapper_stability.py`, `persistence_check.py`, `intrinsic_dimension.py` — shape analyses against null models (see [DISCOVERY_FINDINGS.md](DISCOVERY_FINDINGS.md)).
