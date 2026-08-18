# Project notes

TDA Pitch Clustering — Will Paz's graduate thesis. Combines the Mapper
algorithm (Topological Data Analysis) with a CatBoost-based Stuff+ model to
study MLB pitch quality and outcomes. See [docs/REORG_NOTES.md](docs/REORG_NOTES.md)
for the codebase layout and reorganization history, and
[docs/METHODOLOGY_REVIEW.md](docs/METHODOLOGY_REVIEW.md) for a critical
review of the pipeline logic (mismatches, statistical holes, and what's
genuinely solid) written 2026-08-17, before any methodology changes were
made.

Thesis defense/talk is a few months out (as of 2026-08-17). Presentation
source lives in `docs/presentation/` (`main.tex` — paper skeleton with
abstract only; `presi.tex` — Beamer slide deck, the real content).

## Presentation status

Both `.tex` files are drafts, not close to finished. `main.tex` has only a
title and abstract written. `presi.tex` has 3 sections:

1. **Introduction** — 2 blank frames, unwritten.
2. **Topological Foundations** (frames ~6–18) — topological spaces →
   subspaces/pairs → homotopy → simplices → simplicial complexes → nerve →
   nerve theorem → Reeb graphs → topological graphs. Drafted but has
   precision gaps (see advisor feedback below).
3. **The Mapper Algorithm** (frames ~19–47) — filter/lens function → PCA
   (mean-centering, covariance matrix, spectral theorem, SVD, projections,
   explained variance) → building a cover → DBSCAN clustering (density-
   reachable, density-connected, noise) → constructing the graph →
   KeplerMapper implementation with a code block. Drafted, same kind of gaps.
4. **MLB Pitch-Level Data** — **entirely empty**, 5 blank frames. This is
   where the actual repo results (TDA graph, Stuff+ model, pitcher
   consistency/variance findings) need to land. Currently the biggest gap
   in the deck — the application section doesn't exist yet.

## Advisor feedback (received 2026-08-17, not yet acted on)

Framing from the advisor: assume the audience will ask about anything
underspecified on a slide, and the speaker should be ready to answer —
i.e., these gaps need to be fixed on the slide itself, not just answerable
verbally.

Slide numbers below count every `\begin{frame}` in `presi.tex` in document
order, including section/subsection/subsubsection divider frames (title
page = 1, outline = 2, etc.) — confirmed against the actual frame content,
so they can be used directly to find the right frame in the source.

- **Slide 11 (n-Simplices)** — give an actual mathematical definition, not
  just the informal "geometric object with n+1 vertices" description.
- **Slide 12 (Simplicial Complexes)** — precisely define
  $\langle x_0, ..., x_n \rangle$ notation before using it.
- **Slide 13 (Nerve)** — the definition ("simplicial complex whose vertex
  set corresponds to the index set A with a finite subset...") is unclear,
  compounded by "vertex set of a simplicial complex" never being defined.
  Should be mathematically precise, then followed by a concrete example:
  a topological space, a covering of it, and the resulting nerve/simplex
  from the construction.
- **Slide 14 (The Nerve Theorem)** — "closed convex covering" is used but
  never defined.
- **Slides 15–17 (Reeb Graphs / Torus Example / Topological Graphs)** — no
  explicit definition of a Reeb graph is given, which makes the Slide 16
  example hard to follow. If a Reeb graph is being treated as a kind of
  topological graph, that relationship should be stated after a proper
  definition. The construction of $G$ on Slide 17 isn't followable — it's
  formally described as a collection of subsets of $A$, but how a
  simplicial complex is actually built from that isn't clear.
- **Slide 18 (Notes about Topological Graphs)** — "mapper graph" and
  "point cloud" are used without definition.
- **Slide 24 (PCA Setup)** — why is the mean set to 0 (mean-centering)?
  Needs justification, not just an instruction.
- **Slide 25 (Covariance Matrix)** — "symmetric" is redundant to state here.
- **Slide 26 (Spectral Theorem)** — writing $Cov(\mathbf{A})$ throughout
  instead of just $\mathbf{A}$ is confusing; also, point 2 (diagonalizable)
  can be cut since it's restated in the concluding sentence of the slide.
- **Slide 33 (Selecting k)** — "Linear Summary" is unclear language, needs
  to be spelled out.
- **Slide 38 (DBSCAN subsubsection divider)** — appears out of order or
  redundant (DBSCAN was already named on the prior slide).
- **Slide 39 (Density Reachable)** — very unclear: what is $N_\varepsilon$?
  Elements of what set? What is $X$? These need to be established before
  use.
- **Slide 40 (Density Connectivity and Clusters)** — "point cloud" used
  without definition (first real use of the term).
- **Slide 41 (Noise and Uniqueness)** — are $\varepsilon_i$ and $M_i$ given
  as inputs? In particular, does "Noise" depend on the specific
  $\varepsilon_i, M_i$ chosen? Needs clarification — per-cluster epsilon/M
  subscripts are unusual for standard DBSCAN and may just be inconsistent
  notation.
- **Slide 42 (DBSCAN Algorithm pseudocode)** — expect this slide to need a
  lot of verbal walkthrough; make sure that's planned for, not just left to
  the pseudocode.

## General pacing guidance from advisor

Rough target for an hour-long talk (fine to run over/under):

- **First 15 minutes** — intro through topological graphs (through Slide
  18). Audience should come away understanding what a simplicial complex
  is and how a topological graph is one, with concrete examples provided.
  If time is short, the topological space / homotopy / nerve theorem
  definitions can be cut.
- **Middle 30 minutes** — dimensionality reduction and the Mapper
  algorithm. Skip the spectral theorem and go straight to SVD (same
  mathematical content, saves time for more examples elsewhere).
- **Last 15 minutes** — explain what was actually done with the MLB data,
  and precisely how it ties back to the first 45 minutes of theory. This
  is the section that currently doesn't exist in the deck yet.

## Next steps (not started)

- [ ] Write the Introduction section (currently 2 blank frames)
- [ ] Fix the precision/definition gaps listed above, slide by slide
- [ ] Consider cutting the spectral theorem slide per advisor guidance
- [ ] Build out the MLB Pitch-Level Data section from scratch — this is
      the connective tissue between the theory and the actual repo results
      (TDA mapper graph, Stuff+ model, pitcher consistency/variance
      findings)
- [ ] Rebalance timing to roughly 15/30/15 minutes per the advisor's split
- [x] Do the "discovery" pass on the actual Mapper graph output — done,
      see [docs/DISCOVERY_FINDINGS.md](docs/DISCOVERY_FINDINGS.md) and
      `src/tda/graph_topology_analysis.py`. Found a real, robust
      connectivity finding (a giant component + isolated velocity-outlier
      sub-components, stable across nerve min_intersection thresholds)
      — but the Stuff+ cross-check surfaced a bigger, more urgent issue:
      concrete empirical proof that the production nearest-centroid
      classifier misroutes real pitches for the small/rare outlier
      clusters (real <70mph pitches mostly land in fast clusters and
      vice versa). This directly confirms the feature-space
      fit/inference mismatch flagged in METHODOLOGY_REVIEW.md and now
      has hard numbers behind it.
- [x] Run the controlled root-cause test — done. Ground-truth training
      points round-trip to their own cluster correctly 78% of the time
      when pushed through the actual production classifier logic, which
      rules out a scaler/code defect. Root cause identified: a
      train/apply unit-of-analysis mismatch — the model was fit on
      per-`(pitcher, pitch_type)` *averages*, but production classifies
      *individual raw pitches* against those same archetype centroids,
      and individual pitches have far more scatter than their own
      archetype's mean, especially for the rare small-sample clusters.
      Full writeup and 4 reframing options (ranging from "say it more
      precisely, no code changes" to "fix the aggregation mismatch") in
      docs/DISCOVERY_FINDINGS.md's "How to change the claim for the
      talk" section.
- [x] Ran the correlation check across all 57 clusters (not just the 11
      outliers) — done, see `src/tda/graph_reliability_correlation.py`
      and docs/DISCOVERY_FINDINGS.md's "Correlation check across all 57
      clusters" section. **This flipped and improved the story**: the
      rare/isolated clusters actually have the *highest* ground-truth
      self-consistency (several at 100%); it's the big, popular hub
      clusters that are least self-consistent (Spearman r=−0.67 between
      degree and self-consistency), because they sit in the densest,
      most crowded part of the continuum with many near-identical
      neighbors. The earlier live-data misrouting for rare clusters is
      better explained by short-window event rarity (the live test only
      covered one week) than by a classifier flaw. Reframing options in
      DISCOVERY_FINDINGS.md updated accordingly — recommended option is
      now "topology as a confidence map" (option 2 in the current list):
      graph degree/component-size predict *where* to trust downstream
      classification, for two distinct, explainable, quantified reasons.
- [x] Verified the "attractor node" observation from the crowded-continuum
      check — see `src/tda/attractor_node_analysis.py` and
      DISCOVERY_FINDINGS.md. **It did not hold up**: betweenness
      centrality is zero for all 3 candidate nodes, same as most other
      degree-3 nodes; the original pattern was an artifact of a small,
      unrepresentative sample. Good outcome of flagging it as unverified
      first. The systematic check found a real result instead: the
      graph's actual highest-betweenness nodes are a tight, coherent
      slider/cutter band (85-90mph) sitting exactly between the
      curveball/sweeper region and the four-seam fastball region —
      matching the well-known baseball fact that sliders/cutters are
      "bridge" pitches between fastball and breaking-ball families. This
      is independently corroborated by both the network statistic and
      real domain knowledge — a strong, presentable, verified finding.
- [x] Checked the R&D-actionable version of the bridge finding — see
      `src/tda/pitcher_repertoire_overlap.py` and DISCOVERY_FINDINGS.md's
      "Named-pitcher repertoire overlap" section. 156 real pitchers (of
      3,932 in the training data, IDs only — name resolution unavailable
      in this environment) have two differently-labeled pitches (mostly
      SL/FC) both landing in the verified bridge region. More
      importantly: extending the check graph-wide, the most commonly
      overlapping pitch-type-label pairs (FF/SI, CH/FF, FC/FF, CU/ST,
      SL/ST) are exactly the pairs the baseball industry's own automated
      pitch classifiers are known to struggle distinguishing — the model
      recovered that ambiguity independently from raw physical
      measurements, with no labels given. This is a real validation of
      the whole approach and gives a concrete, actionable framing:
      specific pitchers whose "two different" pitches may not be
      functionally distinct, worth a real pitch-design conversation.
- [ ] **Decide whether to adopt the "topology as confidence map" framing
      for the MLB section** (see docs/DISCOVERY_FINDINGS.md) — this is
      now the recommended option, needs no pipeline changes, just deck
      language once presentation work resumes. The slider/cutter bridge
      finding and the named-pitcher repertoire-overlap validation are
      strong complementary talking points alongside it.
- [x] Checked the circular-`spin_axis`-treated-as-linear hole — see
      `src/tda/spin_axis_circularity_check.py` and
      docs/METHODOLOGY_REVIEW.md item 2. Verified, not just asserted:
      only 0.5% of training archetypes sit near the 0/360 wraparound and
      no fitted cluster's centroid is meaningfully distorted by it, but
      re-encoding `spin_axis` circularly does flip the nearest-cluster
      assignment for 3 of 20 near-wraparound points (15%) — small in
      absolute count, real and nonzero, worth fixing for correctness but
      not a driver of the bigger anomalies found in the discovery pass.
- [x] Checked the Stuff+ leakage hole — see
      `src/validation/stuff_plus_leakage_check.py` and
      docs/METHODOLOGY_REVIEW.md item 5. **First attempt got the wrong
      answer and is kept on record**: comparing combined Stuff+ scores
      between the official pipeline and genuinely out-of-sample 2025 data
      gave an alarming negative correlation, but that used two different
      Stuff+ weighting formulas (1/3-1/3-1/3 vs the optimized
      0.72/0.11/0.17), an invalid comparison regardless of the result.
      Redone comparing the raw model predictions directly (no weighting
      confound): xwOBA r=0.647, miss r=0.586, chase r=0.586, all
      significant, on pitchers the models never saw during training.
      Substantially de-risks the leakage concern — the models
      demonstrably generalize; the missing held-out split may still
      introduce a modest optimistic bias in the exact final numbers, but
      not the "no real signal" scenario the original hole raised.
- [x] Checked the fixed-vs-per-pitch strike zone hole — see
      `src/validation/strike_zone_check.py` and
      docs/METHODOLOGY_REVIEW.md item 7. Run against one real day of
      Statcast data (2,729 pitches): aggregate chase rate barely moves
      (17.41% fixed zone vs 18.32% real per-batter zone), but 2.53% of
      individual pitches get a different chase/not-chase label between
      the two definitions — real training-data mislabeling, small but
      nonzero. Real strike zone top ranged 2.56-4.18ft across batters in
      that one day, confirming genuine batter-to-batter variation the
      fixed 1.6-3.5ft window discards. Easy, mechanical fix (swap in the
      sz_top/sz_bot columns already being fetched), worth doing.
      **Side finding**: `pybaseball.statcast()` currently fails in this
      environment (network succeeds, postprocessing crashes on a
      duplicate-column bug) — affects `fitting_stuff_weights.py` and the
      notebooks if re-run here. **Fixed (2026-08-18)** for
      `assign_pitch_stuffplus_clusters.py`, which was the last script
      still calling `pybaseball.statcast()` directly: it and
      `classify_pitches_to_csv.py` now both go through
      `src/tda/tda_classifier.fetch_savant_csv`, a shared helper that
      hits the Baseball Savant CSV export directly (verified end-to-end
      against live data). `fitting_stuff_weights.py` and the notebooks
      still call `pybaseball.statcast()`/`bwar_pitch()` and would need
      the same treatment if rerun in a broken environment.
- [x] Checked the hard nearest-centroid vs. multi-membership hole — see
      `src/tda/multi_membership_check.py` and METHODOLOGY_REVIEW.md item
      3. Quantified how much ambiguity the single-label choice discards:
      64.6% of training archetypes have a top-2 nearest-cluster margin
      under 5% of the typical inter-centroid spacing, 89.4% under 10%.
      Minimum inter-centroid distance across all pairs is exactly 0.0
      (cube25_cluster1/cube26_cluster0 are literal duplicates). This
      makes it a practically significant issue, not just a theoretical
      one — most points are genuinely ambiguous between 2+ clusters.
- [x] Checked the multiple-comparisons hole — see
      `src/validation/anova_multiple_comparisons_check.py` and
      METHODOLOGY_REVIEW.md item 6. `anova_results_real_data.csv` is 6
      omnibus F-tests (one per outcome metric across all 48 clusters),
      not pairwise cluster comparisons — a smaller surface than assumed.
      **Bigger, separate bug found along the way**: the reported
      p-values aren't real. `anova_real_data.py` (and duplicated in
      `cluster_outcome_validation.py`) converts a correctly-computed
      F-statistic to a p-value with a hardcoded threshold
      (`p=1.0 if F<3.0 else 0.001`), not the real F-distribution CDF.
      Recomputed real p-values are astronomically smaller than the fake
      placeholder in every case, and all 6 results survive Bonferroni
      correction comfortably — so the conclusions already reported
      aren't wrong, but the fake p-value logic is a latent bug that
      would misfire on a more marginal comparison. Worth fixing (swap in
      `scipy.stats.f.sf`) independent of the multiple-comparisons
      question.

**All items in docs/METHODOLOGY_REVIEW.md's ranked hole list have now
been checked** (not necessarily fixed — see each item for the specific
recommendation). Remaining open items are the MLB-section framing
decision and the presentation work itself, both intentionally paused.
