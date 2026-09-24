# Topology discovery pass (2026-08-17)

Follow-up to [docs/METHODOLOGY_REVIEW.md](METHODOLOGY_REVIEW.md)'s central
objection: does the *shape* of the Mapper graph (loops, branch points,
connectivity) tell you anything a flat clustering wouldn't? This is the
first real attempt to answer that, using the actual fitted graph in
`models/tda_mapper_model.pkl` rather than the recomputed similarity graph
`src/tda/tda_graph_visualization.py` builds for visualization.

Reproducible via `src/tda/graph_topology_analysis.py`.

> **Model note (2026-09-24):** sections below dated 2026-08-17 were
> computed on the pre-refit 57-node model. The model was refit 2026-08-18
> (66 nodes). See "Re-check on the refit model" at the end for which
> numbers still hold.

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

**Follow-up: the "attractor node" observation above was checked
systematically and does not hold up as originally described.**
Reproducible via `src/tda/attractor_node_analysis.py`. Betweenness
centrality (the standard network measure of how often a node lies on
shortest paths between other node pairs — the actual definition of a
graph "crossroads") is **exactly zero** for all three candidate nodes
(`cube59_cluster1`, `cube62_cluster2`, `cube62_cluster3`) — same as most
other degree-3 nodes in the graph. Degree and betweenness are almost
perfectly correlated overall (Spearman r=0.89), and these three nodes
aren't unusual even controlling for degree: `cube62_cluster2` actually
has *fewer* large neighbors than the average node of its degree. The
original pattern was noticed in a small, unrepresentative slice (the
top-15 worst-margin table) and didn't survive a systematic check. Good
that it was flagged as unverified before being presented as a finding.

**What the systematic check found instead is a real, verified,
baseball-grounded result.** The nodes that actually top the betweenness
ranking (excluding the giant hub endpoints already discussed elsewhere)
form a tight, coherent group: `cube24_cluster0` (SL, 85.1 mph),
`cube32_cluster0` (SL, 86.8 mph), `cube43_cluster0` / `cube42_cluster0` /
`cube51_cluster0` (FC, 88–90 mph) — all clustered around 85–90 mph with
mild glove-side horizontal break (HB ≈ −3 to −5.5) and moderate rise
(IVB ≈ 1–10). That band sits exactly between the curveball/sweeper
region (`cube9_cluster0`, 81 mph, HB=−10.7, IVB=−7.2 — strong
glove-side drop) and the four-seam fastball region (`cube60_cluster0` /
`cube61_cluster0`, 92–93 mph, HB≈+10.7, IVB≈+11.5 — arm-side ride).

**This matches real baseball domain knowledge directly, without being
told pitch-type labels:** sliders and cutters are widely understood in
scouting and pitching-analytics circles as *bridge* pitches — designed
and thrown with velocity and movement intermediate between a fastball
and a true breaking ball. The graph's actual network-theoretic
crossroads is the slider/cutter velocity-and-movement band connecting
the fastball and curveball regions of the continuum — which is exactly
where a pitching coach or scout would expect the bridge to be. This is a
stronger and more defensible finding than the disproven one it replaced,
precisely because it's both statistically verified (a real centrality
ranking, not an eyeballed pattern) and independently sensible in
baseball terms — the two checks corroborate each other rather than one
being taken on faith.

## Named-pitcher repertoire overlap — the R&D-actionable version

The betweenness finding says a *region* of stuff-space is ambiguous.
The player-development-relevant question is: which real pitchers'
*actual* pitches sit there, and does the topology say anything about
their specific repertoires? Checked directly against the full training
data (3,932 pitcher-pitch-type archetypes, all real 2025 pitchers,
identified by MLBAM `pitcher_id` — name resolution via
`pybaseball.playerid_reverse_lookup()` wasn't available in this
environment, a Chadwick-register parsing issue unrelated to this
analysis; IDs can be cross-referenced manually on Baseball Savant).
Reproducible via `src/tda/pitcher_repertoire_overlap.py`.

One subtlety had to be handled carefully first: a single archetype can
legitimately be a member of multiple *overlapping* Mapper cover cells at
once — that's the cover's `perc_overlap` parameter working as designed,
not two different pitches. The first pass at this script conflated that
with genuine repertoire overlap and overcounted; fixed by deduplicating
to distinct pitch-type labels per pitcher before counting anything as
"two different pitches converging."

**After that fix: 156 real pitchers have two (or more) differently
labeled pitch types both landing in the verified slider/cutter bridge
region** (out of 546 pitchers with any pitch there at all) — most
commonly a labeled slider (SL) and a labeled cutter (FC) sitting in the
same or adjacent nodes, occasionally with a curveball (CU), sweeper
(ST), or four-seamer (FF) in the mix. For those pitchers specifically,
their own two "different" pitches are, by this model's continuous
stuff-features, not clearly differentiated in movement/velocity space —
independent of what Statcast's own auto-labeling calls them.

**Extending this check across the whole graph** (not just the bridge
region) surfaces something that reads as a genuine validation of the
whole approach: the most common overlapping pitch-type pairs are
(FF, SI) 492 times, (CH, FF) 424, (CH, SI) 335, (FC, FF) 179, (FC, SI)
140, (FF, FS) 122, (CU, ST) 110, (SL, ST) 77, (CH, FC) 73, (CU, SL) 55 —
and every one of those pairs (four-seam/sinker, changeup/fastball,
cutter/four-seam, curveball/sweeper, slider/sweeper) is a well-known
source of real ambiguity in pitch classification within the baseball
industry itself. Statcast's own automated pitch-type labeler has
publicly documented difficulty distinguishing sweepers from curveballs
and sliders, and cutters from four-seamers, for exactly the pitchers
whose shapes sit at that boundary. **The model recovered those same
ambiguous boundaries independently, from raw physical measurements
alone, with no pitch-type label ever given to it.** That's a legitimate,
checkable form of validation — if the topology only agreed with
labels where labeling is easy and disagreed randomly everywhere else,
that would undercut the method; instead it disagrees exactly where the
industry's own labeling systems disagree.

**The actionable, R&D-flavored framing:** for a specific pitcher on this
list, this isn't just "their slider and cutter are similar" as an
abstract fact — it's a concrete pitch-design/scouting question. Is the
overlap a problem (the two pitches aren't functionally distinct, one may
be redundant or worth consolidating/differentiating further) or an asset
(deliberately blurred shapes create deception, the batter can't
distinguish them out of hand)? The topology can't answer which on its
own, but it gives a principled, data-driven starting list of exactly
which pitchers and pitch pairs are worth that conversation — instead of
a scout or analyst having to notice it by eye on a movement plot.

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

2. **Node size as a reliability map (reworded 2026-09-24; secondary to
   the shape claim below).** The original wording said graph degree
   predicts where classification can be trusted. On the refit model,
   degree's apparent effect turned out to be mostly node size (see "Re-check
   on the refit model"). State it as: **how much to trust a cluster label
   depends on how crowded the node is, i.e. how many archetypes it holds.**
   - Big nodes in the dense core are the least self-consistent. Their own
     training archetypes round-trip to them least often (Spearman ρ = −0.78
     between node size and round-trip accuracy, n = 66), because many
     near-identical neighboring nodes compete for the same points.
   - Small nodes, including the isolated slow-pitch ones, are highly
     self-consistent on training data.
   - The graph's topology (degree, component) adds no reliable signal beyond
     size. Degree adds −0.21 to −0.35 after controlling for size, depending
     on whether two tiny nodes are included, so the effect isn't robust.
   - R&D reading: treat a cluster label as a confident call for a pitch in
     a small, distinct node, and as "one of several near-identical shapes"
     in a big core node. Multi-membership (docs/METHODOLOGY_REVIEW.md item
     3) is the better tool for the core, since it reports all the nodes a
     pitch belongs to.

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
intentionally paused for now), **option 2 is the realistic near-term
choice** — it requires no pipeline changes, just precise language in the
deck, and the underlying numbers (the 78% ground-truth round-trip rate,
the node-size/self-consistency correlation, the asymmetric direction of
the errors) are already documented above if you want to cite them
directly. Option 4 is the right thing to do
eventually and is now a clearly scoped, well-understood fix — worth
doing if there's time before the defense, but it's a methodology change,
not a wording change, so it should happen deliberately and separately
from deck work.

## Merit of Mapper — what the evidence supports (2026-09-24)

This section supersedes the reframing options above. It records a
step-back assessment after the pitch-type baseline showed that Statcast's
own tags explain more pitch-outcome variance than the clusters
(docs/METHODOLOGY_REVIEW.md item 6). **Decision: the thesis claim moves
from outcome prediction to shape.** Every test below runs on the saved
archetypes (one average profile per pitcher per pitch type, n = 3,943),
the same unit the graph is fit on, and each is compared against null
models: a same-covariance Gaussian, and each feature shuffled within pitch
type.

**1. Stability** (`src/tda/mapper_stability.py`, `data/mapper_stability.csv`).
Verified at scale.
- Features were tracked by content, not node ID: slow-pitch isolation,
  the slider/cutter bridge, and the CU→FF velocity ordering.
- They hold in 93–100% of a 27-point grid over n_cubes, overlap, and eps.
  The chosen-parameter point reproduces the saved model exactly.
- **But both null models reproduce all three features.** They are stable
  but not distinctive: they describe where pitch types sit in feature
  space, not topology beyond that. Don't present them as topological
  discoveries.
- Resampling: 30 fits on 80% of pitchers drawn without replacement (an
  earlier with-replacement bootstrap duplicated points and doubled the
  node count). The three features hold in 97–100% of them.
- **Lens robustness (verified at scale, 2026-09-24).** The same grid,
  subsamples, and nulls were rerun under three more lenses: baseball
  (velocity × induced vertical break), Isomap 2-D, and density (1-D mean
  distance to 10 nearest neighbors). All four lenses see identical
  subsample and null draws, and the PCA rows reproduce the earlier run
  exactly.

  | Feature (share of real grid + subsample fits) | PCA | Baseball | Isomap | Density |
  |---|---|---|---|---|
  | Slow pitches isolated | 97–100% | 94–97% | 100% | 94–100% |
  | CU→FF velocity ordering (ρ ≥ 0.7) | 100% | 100% | 100% | 100% |
  | Slider/cutter bridge (≥ 3 of top 5) | 93–100% | 3–19% | 81–90% | 7–17% |

  - **Lens-independent:** the continuum ordering and slow-pitch
    isolation. These are safe to present.
  - **Lens-dependent: the bridge.** Under the baseball lens, sliders still
    take the top two betweenness spots, but sinker, changeup, and curveball
    nodes fill out the top five. Isomap makes all five SL/FC. Since
    betweenness picks out the middle of a roughly path-shaped graph, and
    the graph is ordered by velocity, "mid-velocity pitches are central"
    is close to automatic. Combined with the nulls reproducing it, **the
    bridge shouldn't be presented as a finding.** The graph-wide
    label-overlap pairs don't depend on it.
  - The density lens is 1-D and gives far fewer nodes (median 25 vs 68),
    so it isn't directly comparable. It still isolates the slow pitches.
  - Nulls under the new lenses match the PCA picture: they reproduce the
    ordering and mostly reproduce isolation. One exception: under Isomap,
    nulls isolate slow pitches in only about half the fits, against 100%
    for real data. That's a single-lens result, so don't lean on it.

**Data provenance and small-sample archetypes (verified, 2026-09-24).**
- Provenance: a fresh pull of 2025-03-28 to 11-04 returns exactly 721,799
  pitches and rebuilds 3,943 archetypes. Only 0.3% of archetypes differ
  from the saved ones by more than 0.01 scaled units, and 3 changed pitch
  type (Statcast reclassification after the fit). The saved model is the
  full season as documented.
- The notebook applies **no minimum pitch count**. The median archetype
  averages 98 pitches, but 8.8% average fewer than 5 and 22% fewer than
  20. The slow (< 70 mph) archetypes are especially thin: median 9
  pitches, and a quarter average 2 or fewer.
- Sensitivity (one PCA fit at the chosen parameters per threshold):

  | Min pitches | Archetypes | Slow | Components | Slow isolated | CU→FF ρ |
  |---|---|---|---|---|---|
  | 1 (as fitted) | 3,940 | 131 | 9 | 97% | 1.00 |
  | 10 | 3,388 | 63 | 5 | 83% | 1.00 |
  | 20 | 3,073 | 29 | 8 | none in any node | 0.98 |

  The continuum ordering is unaffected. The slow pitches stay off the
  continuum at every threshold, but at ≥ 20 pitches they're too sparse to
  form nodes at all (DBSCAN noise). So "slow pitches sit apart from the
  continuum" is robust, while "they form their own small components"
  depends on keeping low-count archetypes. Persistent homology and
  intrinsic dimension have not been rerun with a threshold.

**2. Persistent homology** (`src/tda/persistence_check.py`). Passes a
noisy-circle sanity check.
- **No persistent H₁**, in real data or either null (5 subsamples of
  1,000). The Mapper graph's 76–86 cycles are cover artifacts, not loops.
- No more long-lived H₀ gaps than a Gaussian (17 vs about 19).
- Real archetypes are more concentrated (median merge scale 1.00 vs
  1.52–1.54 Gaussian, 1.24–1.25 shuffle).
- Caveat: that shuffle null breaks the spin_cos/spin_sin circle; see 3.
- **Why the graph has cycles anyway (verified at scale,
  `src/tda/nerve_homology.py`).** KeplerMapper draws only the 1-skeleton
  of the nerve. With a 2-D lens, many triples of nodes share archetypes,
  so the nerve has 2-simplices (113 triangles in the saved model) that
  fill the graph's cycles.

  | | graph b₁ | full-nerve b₁ |
  |---|---|---|
  | saved model | 86 | 1 |
  | real, 27 grid fits | 33–147 (median 76) | 0–4 (median 1) |
  | Gaussian null, 5 draws | 56–97 | 2–9 |
  | shuffle null, 5 draws | 56–80 | 0–2 |

  Graph cycles track the parameters, not the data. Once the nerve is
  treated as the simplicial complex it is, b₁ is small, unstable, and
  within the null range, consistent with persistent homology. This is
  the precise version of "the cycles are cover artifacts", and it ties
  straight back to the simplicial-complex slides. Computed over ℤ₂ to
  match the theorem below (identical to ℚ on all 38 fits).
- **Theorem backing it (citation verified against the paper).**
  T. K. Dey, F. Mémoli, Y. Wang, *Topological Analysis of Nerves, Reeb
  Spaces, Mappers, and Multiscale Mappers*, SoCG 2017, LIPIcs 77:36
  ([arXiv:1703.07387](https://arxiv.org/abs/1703.07387)).
  - **Theorem 8:** if a cover 𝒰 of X is path connected, H₁(X) → H₁(N(𝒰))
    is a surjection.
  - **Theorem 18** applies this to Mapper, N(f\*𝒰), whose pullback cover
    (the path components of f⁻¹(U_α)) is path connected by construction.
    So **b₁(Mapper) ≤ b₁(X)**: "nerves can only kill" H₁.
  - **Hypotheses:** X compact; f : X → Z continuous and well-behaved
    (preimages of path-connected open sets have finitely many path
    components); 𝒰 an open cover of Z; ℤ₂ coefficients.
  - The same paper shows the quotient X → Reeb space is surjective on H₁
    (Claim 4.2, in the proof of Theorem 27).
  - **How it applies here, honestly stated.** The theorem is about a
    space X, and we have a finite sample. DBSCAN clusters only approximate
    the path components of preimages (of, say, a union of ε-balls around
    the core points). So the argument is: persistent homology estimates
    b₁(X) ≈ 0, the theorem says any correct Mapper of X has b₁ = 0, and
    the observed nerve b₁ of 0–4 (null-range, unstable) is consistent
    with that. The 76–86 graph cycles come from reading only the
    1-skeleton. Present it as "theorem + consistent evidence", not as a
    proof about the data.

**3. Intrinsic dimension** (`src/tda/intrinsic_dimension.py`). The TwoNN
and Levina–Bickel MLE estimators both pass sanity checks (2-D plane → 2.0,
9-D Gaussian → 8.6–8.9).
- The spin_cos/spin_sin pair encodes one angle, so a fair null must
  permute it as a unit.
- Against that null, real archetypes are lower-dimensional overall (MLE
  6.26 vs 7.07, TwoNN 6.76 vs 7.52) and within every major pitch type, by
  roughly 0.4–0.8 of a dimension.
- **Supported:** real within-type coupling of the physical features.
  **Not supported:** a "low-dimensional continuum." The space is about 6
  of 9 dimensions, only moderately constrained.

**What the thesis can defensibly claim**
- Pitch-shape space is **connected, with no loops, no hidden branches,
  and no extra gaps beyond a Gaussian**, plus a tail of slow-pitch
  outliers.
- It is **moderately constrained** by within-type physical coupling
  (~0.5–0.8 of a dimension below a fair null).
- Pitch-type tags draw hard boundaries across it. The label-overlap
  result supports this. So does multi-membership: ~55% of live pitches
  belong to 2+ overlapping nodes, and for ~45% the single centroid label
  isn't one of the pitch's actual memberships.
- Mapper's merit here is not discovering exotic structure. It produces an
  interpretable summary of a simple, continuous space, and, **paired with
  null models, persistent homology, and dimension estimates**, it
  certifies what that topology is and isn't. The null-model comparisons
  are arguably the novel methodological contribution: applied Mapper work
  rarely checks graph features against structureless baselines, and here
  that check shows that apparent features (bridges, loops) can be
  reproduced without any real structure.

**Not built:** outcome smoothness on the graph (#4). It was deprioritized
because, if the graph mostly encodes feature-space geometry, "outcomes
are smooth on the graph" largely restates what the Stuff+ models already
capture. The within-tag, usage-controlled test is deferred to future
work.

## Re-check on the refit model (2026-09-24)

The 2026-08-17 findings above came from the pre-refit 57-node model. All
model-derived scripts were rerun on the current 66-node model (fit on the
full 2025 season, 3,943 archetypes). Hardcoded node IDs were replaced with
ones derived from the graph, since node IDs change on every refit.

| Finding | Pre-refit (57 nodes) | Refit (66 nodes) | Status |
|---|---|---|---|
| Misroutes landing on a graph-adjacent cluster | 93.8% | 89.7% (3,972 of 4,430) | holds |
| degree ↔ round-trip accuracy (Spearman) | −0.67 | −0.72 | holds, **but see below** |
| Top-5 betweenness nodes, majority type | SL/FC band | SL, FF, FC, ST, FC | mostly holds (3/5 SL/FC, 4/5 counting ST) |
| Pitchers with 2+ own types in bridge nodes | 156 | 66 (3 bridge nodes, not 5) | smaller; count depends on how many nodes qualify |
| Most common same/adjacent label pairs | FF/SI, CH/FF, FC/FF, CU/ST, SL/ST | FF/SI, CH/FF, CH/SI, FC/FF, FC/SI, FF/FS, CH/FC, CU/ST, …, SL/ST | holds |
| "Attractor node" hypothesis | refuted | still no pattern (re-derived candidates: betweenness at or below same-degree peers except the top hub itself) | still refuted |

**Caveat on the confidence-map claim (verified on all 66 nodes, but small
n).**
- Node size (`train_n`) predicts round-trip accuracy more strongly than
  degree does (ρ = −0.78 vs −0.72), and degree and size are strongly
  correlated (ρ = 0.84).
- Controlling for size, degree's partial Spearman correlation with
  accuracy is only −0.21 (p = 0.10). Controlling for degree, size's
  partial correlation is −0.47 (p < 10⁻⁴).
- Dropping the two tiny nodes that have no live data moves degree's
  partial correlation to −0.35 (p = 0.004), so the residual degree effect
  isn't robust. Size's effect stays strong either way (−0.47 to −0.64).
- Live-data check: `data/pitch_stuffplus_clusters.csv` (2025-03-28 to
  04-04) was truncated at the 25k Savant cap and has been regenerated in
  full (28,522 pitches). On the truncated pull, degree seemed to predict
  live speed-matching error beyond size (partial −0.46). On the full pull
  that disappears (−0.16, p = 0.20). Degree, size, and component size each
  correlate about −0.4 with live error, and none separates from the
  others.
- So "crowded = less reliable" is a statement about node *size*, not graph
  *topology*. Option 2 above has been reworded to match.
