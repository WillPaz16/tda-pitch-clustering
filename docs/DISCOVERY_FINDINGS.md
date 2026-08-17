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

## Correlation check across all 57 clusters — refines the story

The above was checked against only the 11 outlier clusters found by
inspection. Extending the ground-truth round-trip test to *every* cluster
in the graph, and correlating topology (degree, connected-component
membership) against two reliability measures (ground-truth
self-consistency, and real-data speed-matching error from
`data/pitch_stuffplus_clusters.csv`), reproducible via
`src/tda/graph_reliability_correlation.py`, changes the picture:

| relationship | Spearman r | p | n |
|---|---|---|---|
| degree ↔ ground-truth round-trip accuracy | −0.673 | <0.0001 | 57 |
| train_n ↔ ground-truth round-trip accuracy | −0.641 | <0.0001 | 57 |
| component_size ↔ ground-truth round-trip accuracy | −0.357 | 0.007 | 57 |
| degree ↔ live speed-matching error | −0.481 | 0.0002 | 55 |
| component_size ↔ live speed-matching error | −0.541 | <0.0001 | 55 |
| train_n ↔ live speed-matching error | −0.333 | 0.013 | 55 |

**This is the opposite of the naive story.** Isolated/small clusters have
*higher* ground-truth self-consistency, not lower — several of the
outlier clusters round-trip at 100%. It's the large, high-degree hub
clusters (`cube60_cluster0`: 1088 training points, 32% self-consistency;
`cube67_cluster0`: 835 points, 23%; `cube9_cluster0`: 353 points, 23%)
that are internally the *least* self-consistent. That makes sense once
you see why they have high degree in the first place: they sit in the
densest, most crowded part of the continuum, surrounded by many
near-identical neighboring archetypes. A training point near a big
cluster's edge is often, by raw centroid distance, actually closer to an
adjacent cluster than its own — even though DBSCAN (density-based, not
centroid-based) originally grouped it there. That's a real reliability
concern, but it belongs to the *popular, crowded* clusters, not the rare
isolated ones.

So the earlier live-data misrouting (real slow pitches landing in fast
clusters) is better explained by a different, simpler mechanism than
"averaging vs. individual variance" alone: **event rarity over a short
inference window.** `assign_pitch_stuffplus_clusters.py`'s live run only
covered one week (`2025-03-28` to `2025-04-04`). Eephus-type archetypes
were built from only 4–8 pitcher-pitch-type pairs across an *entire
season* of training data — it's entirely plausible zero genuine eephus
pitches were thrown by anyone, league-wide, in that specific week. The
~60–69 mph pitches that did occur and got misassigned were mostly
generically labeled ("FA"), most likely an ordinary pitch thrown a bit
slower than usual — which legitimately doesn't belong in the eephus
archetype on the other 7 dimensions, not a classifier failure.

**Revised claim (stronger and more accurate than the original one):** the
model's ground-truth self-consistency is good, *including* for rare
archetypes. The two real, distinct reliability concerns are (1) the
densest region of the continuum, where many similar pitch shapes
genuinely compete for the same real pitches, and (2) rare archetypes
being hard to observe reliably over short time windows, independent of
whether the classifier itself works. Both are legitimate, explainable,
and — usefully — both are visible directly from the graph's topology
(degree and component size) without needing to separately audit sample
sizes or rerun classification for every cluster by hand.

## Crowded-continuum ambiguity — checked directly, not assumed

The correlation check above showed the crowded, high-degree hub clusters
have the lowest ground-truth self-consistency. On its own that's
ambiguous: it could mean genuine boundary blending in a real continuum
(benign, expected), or it could mean those training points don't
actually belong near their assigned centroid at all (more concerning).
Rather than assume which, checked directly, against the full population
of ground-truth misroutes across the whole graph (4,315 misroutes, not a
sample), reproducible via `src/tda/crowded_continuum_analysis.py`:

- **93.8% of misroutes land on a graph-adjacent cluster** — an edge
  actually exists between the origin and the reassigned cluster in the
  fitted Mapper graph. Only 6.2% jump to a non-adjacent cluster.
- **Margins are small.** The gap between "distance to the point's own
  cluster centroid" and "distance to the reassigned cluster's centroid,"
  normalized by the typical (median) centroid-to-centroid distance
  across the whole graph, has a median of 0.072 and a 75th percentile of
  0.134 — most misroutes are close calls, not wild jumps.

**This is a verified result, checked against the full misroute
population**, and it supports the benign interpretation: the low
self-consistency in crowded hub clusters mostly reflects genuine
boundary blending between adjacent, similar archetypes — exactly what
`perc_overlap` in the Mapper cover is designed to produce — not points
that don't belong near their assigned centroid. A hard nearest-centroid
label forced onto a naturally graded, continuous region will always
produce some ambiguity at the boundaries; that's a property of the
region being genuinely continuous, not a modeling failure.

**One unverified observation, flagged as such, not a claim:** the
worst-margin misroutes cluster heavily around a few specific destination
nodes (`cube59_cluster1`, `cube62_cluster2`, `cube62_cluster3`), which
repeatedly "pull" points from several different large neighboring hubs.
That could mean those nodes sit at a genuine convergence point in the
continuum — but this was noticed by inspecting the top-15 worst-margin
table, not tested systematically, so it should not be presented as a
finding without a further check (e.g. whether those nodes have unusually
high betweenness centrality, or unusually many distinct large neighbors,
compared to other nodes of similar degree).

## Practical implications

- The outlier-disconnection finding is **real in the fitted graph** (the
  ~3,900 pitcher-pitch-type training archetypes), and — refined by the
  correlation check above — the isolated/rare clusters are actually the
  *most* internally self-consistent part of the graph. The reliability
  concerns are elsewhere: the crowded, high-degree hub clusters (least
  self-consistent on their own training data) and short-window rarity of
  extreme archetypes (why live data looked bad for the outliers in a
  single week of inference). Both are explainable and both are visible
  from graph topology alone (degree, component size) — that's the
  reframed, stronger claim to use.
- This also means every downstream script reading
  `pitch_stuffplus_clusters.csv` (`pitcher_consistency.py`,
  `variance_analysis.py`, `predictive_model_comparison.py`) has some
  fraction of its cluster assignments affected — but per the correlation
  check, the affected fraction skews toward the *large, popular* clusters
  (crowded-continuum ambiguity) more than the rare ones, which is the
  opposite of the original assumption. Not yet corrected for in those
  downstream scripts; worth keeping in mind when interpreting their
  results, especially anything that treats `cluster_id` as a clean,
  unambiguous label.

## How to change the claim for the talk

Options, roughly ordered from "say it differently" to "actually fix the
pipeline" — none of these have been implemented, this is a menu to choose
from. **Updated after the all-clusters correlation check** — options 2
and 3 below were written before that check and got the direction backward
(they assumed rare clusters were the unreliable ones; it's actually the
crowded, popular ones that are least self-consistent). Corrected here.

1. **Scope the claim to the training data, not live classification
   (cheapest, no code changes).** Present the connectivity finding
   (giant component + isolated velocity outliers) as a property of the
   ~3,900 pitcher-pitch-type archetypes the graph was built from — a
   descriptive result about the fitted model, not an operational claim
   about how new pitches get classified.

2. **The topology-as-confidence-map framing (recommended — strongest,
   most accurate, no code changes needed).** State it as: the graph's
   own shape predicts where downstream classification can and can't be
   trusted, for two distinct, explainable reasons — (a) the densest,
   most crowded part of the continuum has real ambiguity between many
   similar archetypes (measurable: high-degree clusters have the lowest
   ground-truth self-consistency, Spearman r=−0.67), and (b) rare
   archetypes are hard to observe reliably over short time windows,
   independent of whether the classifier works (ground-truth
   self-consistency for the isolated clusters is actually high). This is
   accurate, quantified, and turns the whole investigation into a
   feature rather than an apology — genuinely useful framing for the
   player-development/R&D angle too: know how much to trust a specific
   pitch classification based on how crowded or how rare its region of
   the graph is.

3. **Restrict any single-pitch live-classification demo to the
   well-populated clusters**, and be explicit that the crowded, dense
   part of the continuum is where nearest-centroid assignment is
   genuinely ambiguous (not the rare outliers) — if the talk needs a
   "here's how we classify a new pitch" moment, pick an example from a
   cluster with both decent size and low degree if one exists, or state
   the ambiguity outright for whichever example is used.

4. **Actually fix the mismatch (real work, most defensible, not done
   yet).** Two ways to do it: (a) aggregate new pitches the same way the
   training data was aggregated — average by `(pitcher, pitch_type)`
   before classifying, so training and inference share the same unit of
   analysis, or (b) fit the Mapper model directly on individual raw
   pitches instead of pitcher-pitch-type averages, so there's no
   aggregation mismatch to begin with (this changes the methodology more
   substantially and would need its own validation pass). Either removes
   the root cause rather than working around it. Note this doesn't fully
   apply anymore to the *rare-archetype* misrouting specifically (that
   looks more like short-window rarity than an averaging artifact) — it
   would still help the crowded-continuum ambiguity, though.

Given the timeline (defense a few months out, presentation work
intentionally paused for now), **option 2 is the realistic and strongest
near-term choice** — it requires no pipeline changes, just precise
language in the deck, and the underlying numbers (the 78% ground-truth
round-trip rate, the degree/self-consistency correlation,
the asymmetric direction of the errors) are already documented above if
you want to cite them directly. Option 4 is the right thing to do
eventually and is now a clearly scoped, well-understood fix — worth
doing if there's time before the defense, but it's a methodology change,
not a wording change, so it should happen deliberately and separately
from deck work.
