# TDA Pitch Clustering

Graduate thesis project combining the Mapper algorithm (Topological Data
Analysis) with a CatBoost-based Stuff+ model to study MLB pitch quality
and outcomes. See [CLAUDE.md](../CLAUDE.md) at the project root for
current status and next steps, [REORG_NOTES.md](REORG_NOTES.md) for the
codebase layout, and [METHODOLOGY_REVIEW.md](METHODOLOGY_REVIEW.md) for a
critical review of the pipeline logic.

**[Live demo: interactive Mapper graph](https://willpaz16.github.io/TDAPitchClustering/results/tda_mapper_graph.html)**

## Setup

```bash
python -m pip install -r requirements.txt
```

## Layout

- `src/tda/` — TDA/Mapper clustering pipeline (core methodology)
- `src/stuffplus/` — Stuff+ / xwOBA CatBoost modeling
- `src/validation/` — pitcher consistency, variance, ANOVA, predictive comparisons
- `src/data_fetch/` — Statcast/Savant data pulling and integration
- `notebooks/` — model-building notebooks (TDA Mapper fit, Pro Stuff+, College Stuff+)
- `models/`, `data/`, `results/` — trained model artifacts, CSV/JSON data, and figures/visualizations
- `docs/` — this file, methodology notes, and the presentation source

## Pipeline order

1. `notebooks/TDA_Pitch_Clustering.ipynb` — fits the Mapper model on a season of Statcast data, saves `models/tda_mapper_model.pkl`.
2. `notebooks/ProStuff+.ipynb` / `CollegeStuff+.ipynb` — train the xwOBA/miss/chase CatBoost models and compute pitcher-level Stuff+.
3. `src/tda/classify_pitches_to_csv.py` or `src/tda/assign_pitch_stuffplus_clusters.py` — assign new pitches to existing clusters (and Stuff+, in the latter) via nearest-centroid distance. See [CLASSIFY_PITCHES_README.md](CLASSIFY_PITCHES_README.md).
4. `src/tda/tda_graph_visualization.py` — builds the cluster graph visualization.
5. `src/validation/*.py` — consistency, variance, and predictive validation of the resulting clusters against outcomes.
