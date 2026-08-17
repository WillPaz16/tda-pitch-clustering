# Topology discovery pass (2026-08-17)

Follow-up to [docs/METHODOLOGY_REVIEW.md](METHODOLOGY_REVIEW.md)'s central
objection: does the *shape* of the Mapper graph (loops, branch points,
connectivity) tell you anything a flat clustering wouldn't? This is the
first real attempt to answer that, using the actual fitted graph in
`models/tda_mapper_model.pkl` rather than the recomputed similarity graph
`src/tda/tda_graph_visualization.py` builds for visualization.

Reproducible via `src/tda/graph_topology_analysis.py`.

## What the fitted graph actually looks like

The saved KeplerMapper graph has 57 nodes and is **not one connected
structure**. It splits into a giant component (46 nodes, holding the large
majority of pitcher-pitch-type archetypes) plus 8 small isolated pieces
(11 nodes total).

**The isolated pieces are not random** — every one of them is an extreme
outlier by velocity: eephus pitches (~42–51 mph), unclassified slow
"fastballs" (~59–62 mph), a slow curveball (~48.5 mph), a pair of unusually
slow four-seamers (~79 mph), and one isolated 90 mph four-seam singleton.
This held up across a robustness check varying `nerve_min_intersection`
(how many shared points are required to draw an edge) from 1 to 4 — the
same outliers kept splitting off at every threshold.

Within the giant component, ordering nodes by (horizontal break, induced
vertical break, velocity) traces a physically sensible continuum:
**curveballs → sweepers/sliders → cutters → four-seam fastballs →
changeups**, recovered without the model ever being told pitch-type labels.

**Caveat, not a finding:** the giant component is a dense tangle, not a
clean loop. The independent-loop count (first Betti number) ranged from
52 to 79 depending on the min_intersection threshold tested — it never
collapsed toward something small and citable. Don't present a specific
loop count; present the shape (dense local continuum) instead.

**Caveat on hub nodes:** the two highest-degree nodes in the graph are
also the two largest, most heterogeneous clusters (1088 and 353 pitches,
folding in 16 and 13 different Statcast pitch-type labels). High degree
here likely reflects "big catch-all cluster at a cover-grid boundary,"
not a meaningful topological junction. Don't build a slide around a
specific hub node without checking this first.

## The Stuff+ cross-check — and what it actually revealed

The plan was straightforward: check whether the disconnected "outlier"
clusters show a measurably different Stuff+ distribution than the giant
component, using the real per-pitch data in
`data/pitch_stuffplus_clusters.csv`.

**Result: no significant difference** (giant mean 100.34 vs. outlier mean
100.36, Mann-Whitney p = 0.85). On its own that would be a clean null
result. But digging into *why* revealed something more important:

**The production nearest-centroid classifier does not reliably route real
pitches to the archetype clusters they should match, specifically for
these small/rare clusters.** Checked directly against
`data/pitch_stuffplus_clusters.csv`:

- 72 real pitches under 70 mph exist in the dataset. Only **1** of them
  landed in an outlier (slow-trained) cluster — the other 71 were assigned
  to giant-component clusters trained on 78–95 mph archetypes.
- Conversely, every real pitch that *did* land in an outlier cluster
  averaged 86–94 mph — nowhere near the 42–79 mph the cluster was actually
  trained on. E.g. `cube46_cluster0` was fit on 5 eephus pitches averaging
  42 mph; the 5 real pitches assigned to it in production average 88.4 mph.

This is bidirectional and large — not noise. It directly confirms the
"feature-space mismatch between fit and inference" issue flagged in
`docs/METHODOLOGY_REVIEW.md`: the clusters were fit on one feature space
and the saved inference scaler is a separate refit, so nearest-centroid
matching for new pitches is an approximation of the original clustering,
not a faithful reproduction of it — and that approximation breaks down
specifically for the rare, small-sample (3–13 point) outlier clusters.

**Two candidate root causes, not yet distinguished:**
1. The outlier clusters were fit on tiny samples (3–13 pitcher-pitch-type
   pairs), so their centroids in the other 7 (non-speed) dimensions are
   noisy/unrepresentative — a new point matching on speed alone can still
   end up closer, in aggregate 8-D distance, to a large dense fast
   cluster than to the correct sparse slow one.
2. A units/sign/data-source discrepancy between the training data
   (`pybaseball.statcast()`, full season pull) and the production
   inference fetch (`classify_pitches_to_csv.py` hits the Baseball Savant
   CSV export directly) for one or more features — not ruled out, would
   need a controlled test (push one known real slow pitch through the
   exact production function and inspect its raw feature vector against
   the trained centroid) to confirm or eliminate.

## Practical implications

- The outlier-disconnection finding is **real in the fitted graph** (the
  ~3,900 pitcher-pitch-type training archetypes) but **not currently
  trustworthy as a claim about live/new pitch data** — the inference
  pipeline can't reliably route new pitches to those sparse archetype
  clusters. If this finding goes in the talk, it needs to be scoped
  explicitly to "the training data used to build the graph," not
  "pitches we classify going forward."
- This also means every downstream script reading
  `pitch_stuffplus_clusters.csv` (`pitcher_consistency.py`,
  `variance_analysis.py`, `predictive_model_comparison.py`) has some
  fraction of its cluster assignments affected by this same mismatch —
  worst for the rare/small clusters, presumably negligible for the large
  well-populated ones, but not yet quantified.

## Recommended follow-up (not done yet)

Run one known, genuinely slow real pitch through the exact function
`assign_pitch_stuffplus_clusters.py`/`classify_pitches_to_csv.py` uses in
production, and inspect its full raw 8-feature vector next to the trained
eephus centroid, to determine whether root cause 1 or 2 above is
responsible. That determines whether the fix is statistical (weight
distance by cluster sample size / don't trust tiny clusters at inference
time) or a straightforward data-source/units bug.
