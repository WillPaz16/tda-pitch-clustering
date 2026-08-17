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

## Controlled follow-up test — root cause identified

Ran the actual training data itself (ground truth, no live fetch, no
external data source at all) back through the exact production
nearest-centroid logic (`scaler.transform` + `argmin` distance to
`cluster_summary` centroids), for every point belonging to one of the 11
outlier/small clusters, to isolate whether the classifier itself is
broken versus something about how live pitches are prepared before
reaching it.

**Result: 45 of 58 (78%) of a cluster's own training points correctly
round-trip back to their own cluster.** The 13 "failures" are not wild
misassignments — every one is a swap between near-identical neighboring
clusters (e.g. a 57 mph point landing in a 59 mph cluster instead of its
own 56 mph one; two clusters both centered at 78.9 mph that are
effectively duplicates of each other). **This rules out a scaler or
distance-metric defect** — the classifier correctly and sensibly routes
its own training data.

**Root cause: a train/apply unit-of-analysis mismatch.** The model was
fit on `avgStuff` — one row per `(pitcher, pitch_type)`, each row an
average over potentially hundreds of individual pitches, which washes
out pitch-to-pitch noise. But `assign_pitch_stuffplus_clusters.py`
classifies **individual raw pitches** against those same archetype
centroids. A single raw pitch naturally has far more scatter than its
own archetype's mean — especially for rare archetypes like eephus, whose
centroid was built from only 4–8 training points and occupies a tiny
region of the 8-dimensional feature space. A noisy individual pitch
easily drifts out of that narrow region toward the giant, dense
fastball/breaking-ball cluster, which explains the asymmetric direction
observed (slow real pitches pulled toward the big fast cluster far more
often than the reverse — the giant cluster is large and "attractive,"
the tiny archetype regions are easy to overshoot). This is a genuine
statistical limitation of the pipeline design, not a code bug in the
classifier or a data-source/units mismatch.

(A fully clean "watch one specific individual raw pitch fail" demo was
not run, since that requires fetching fresh live per-pitch data before
aggregation, which wasn't done here — but the ground-truth round-trip
result plus the direct visibility of the averaging-vs.-individual
mismatch in the code is enough evidence to be confident in this as the
primary driver.)

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

## How to change the claim for the talk

Options, roughly ordered from "say it differently" to "actually fix the
pipeline" — none of these have been implemented, this is a menu to choose
from:

1. **Scope the claim to the training data, not live classification
   (cheapest, no code changes).** Present the connectivity finding
   (giant component + isolated velocity outliers) as a property of the
   ~3,900 pitcher-pitch-type archetypes the graph was built from — a
   descriptive result about the fitted model, not an operational claim
   about how new pitches get classified. This sidesteps the whole issue
   because the finding genuinely is true and robust at that level; it's
   only the "and here's how we'd classify a new pitch into this
   structure" extension that's shaky.

2. **State the limitation directly, with the numbers already in hand
   (turns a weakness into a methods-rigor point).** Say explicitly:
   "the classifier is validated at 78% self-consistency on its own
   training archetypes, and known to be less reliable for individual
   live pitches against rare, small-sample archetypes, because it was
   trained on per-pitcher-pitch-type averages rather than raw pitches."
   Committees generally respond much better to a stated, quantified
   limitation than to one they have to catch themselves.

3. **Restrict any live-classification claims to the well-populated
   clusters.** The mismatch is worst for the 11 small/rare clusters;
   the large clusters in the giant component (hundreds to 1000+ training
   points each) don't have this small-sample problem. If the talk needs
   a "here's how we classify a new pitch" moment, use one of those as
   the example and don't lean on the outlier clusters for anything
   beyond describing the fitted graph.

4. **Actually fix the mismatch (real work, most defensible, not done
   yet).** Two ways to do it: (a) aggregate new pitches the same way the
   training data was aggregated — average by `(pitcher, pitch_type)`
   before classifying, so training and inference share the same unit of
   analysis, or (b) fit the Mapper model directly on individual raw
   pitches instead of pitcher-pitch-type averages, so there's no
   aggregation mismatch to begin with (this changes the methodology more
   substantially and would need its own validation pass). Either removes
   the root cause rather than working around it.

Given the timeline (defense a few months out, presentation work
intentionally paused for now), **option 1 or 2 are the realistic
near-term choices** — they require no pipeline changes, just precise
language in the deck, and the underlying numbers (78% round-trip rate,
the asymmetric direction of the errors) are already documented above if
you want to cite them directly. Option 4 is the right thing to do
eventually and is now a clearly scoped, well-understood fix — worth
doing if there's time before the defense, but it's a methodology change,
not a wording change, so it should happen deliberately and separately
from deck work.
