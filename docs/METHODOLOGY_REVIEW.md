# Methodology review (2026-08-17)

A critical, PhD-committee-style pass over the actual pipeline logic (not
code hygiene — this is about whether the math/stats hold up). Written
before any "discovery" work (finding an actual interpretable topological
result) so the open questions are on record first. Nothing in the
pipeline has been changed as a result of this review yet.

## The central objection

The deck's first 45 minutes builds up nerves, simplicial complexes,
covers, overlapping preimages — the entire point of Mapper over plain
clustering is that it preserves connectivity, loops, and branching that
flat clustering destroys. But downstream, `cluster_id` is used almost
exactly like a k-means label: ANOVA across clusters, variance comparison
across clusters, "consistency" = does a pitcher land in the same cluster.
None of that requires a *graph* — a partition would do.

The one place the graph structure is actually load-bearing is
`src/validation/graph_consistency_analysis.py`'s distance-penalized
consistency score (penalizing landing in a *nearby* cluster less than a
*distant* one) — that is a genuinely topological use of the construction
and is probably the strongest existing idea in the repo.

**Open question to resolve before the defense:** what does the *shape* of
the Mapper graph (loops, branch points, connectivity) tell you that a
plain clustering wouldn't? If there's an interpretable loop or branch
(e.g. a fastball/slider continuum forking into two-seam vs. sinker
mechanics), that's the strongest topology-specific finding available and
it isn't currently surfaced anywhere in the code or the deck. This is
the "discovery" work to do next.

## Concrete methodological holes, ranked

1. **Feature-space mismatch between fit and inference.** The clusters
   were fit on `avgStuff` with 9 features (8 raw stuff columns +
   `spin_axis_clock`, a derived feature) — both the grid search that
   picked `n_cubes=10, eps=1, min_samples=4` and the final `mapper.map()`
   call ran on that 9-column standardized space
   (`notebooks/TDA_Pitch_Clustering.ipynb`). But the model actually saved
   for inference refits a new `StandardScaler`/PCA on only the first 8
   columns, sliced positionally rather than by name (works today only
   because `spin_axis_clock` happens to be appended last). Consequence:
   every downstream nearest-centroid classifier
   (`src/tda/classify_pitches_to_csv.py`,
   `src/tda/assign_pitch_stuffplus_clusters.py`) operates in a different
   scaled feature space than the one that actually produced the
   clustering. It's an approximation of the original Mapper output, not a
   faithful reproduction.

2. **`spin_axis` is circular but treated as linear.** It (and its
   derived `spin_axis_clock`) goes into `StandardScaler` and Euclidean
   distance exactly like `release_speed`. 359° and 1° are functionally
   identical directions but ~358 standardized units apart under this
   metric. Should be encoded as $(\cos\theta, \sin\theta)$ to respect the
   circle topology — a real metric-space violation, not just a nitpick,
   given how much the deck emphasizes getting topological definitions
   precise.

3. **Hard nearest-centroid assignment contradicts the theory being
   presented.** True Mapper allows a point to belong to multiple
   simplices because cover sets overlap — that's what the nerve theorem
   depends on. Inference does `argmin` over centroid distances: single
   nearest cluster, full stop. Fine as engineering, but it means
   production is really "Mapper once, to define archetypes, then 1-NN
   forever after," not "Mapper, generatively." Worth stating explicitly
   rather than letting the audience assume otherwise.

4. **xwOBA model has very low explanatory power (R² ≈ 0.017), even after
   feature engineering** (`notebooks/ProStuff+.ipynb`). Defensible —
   "stuff" alone genuinely doesn't predict contact quality well, that's
   a known result in the public pitch-modeling literature — but needs to
   be stated proactively, not discovered by the committee. Miss/chase
   classifiers are much stronger (AUC ≈ 0.62–0.63); lead with those.

5. **Leakage in the final Stuff+ scores.** The cell that generates
   `pred_xw`/`pred_miss`/`pred_chase` for the final pitcher-level Stuff+
   aggregation reuses the same `df` the models were trained on
   (2022–2024, full dataset), not a held-out split. `train_test_split`
   was only used to report validation metrics during training; production
   Stuff+ values are computed by predicting on data that includes the 80%
   the models were fit on. This optimistically biases the final ratings,
   which matters for the later claim (`predictive_model_comparison.py`)
   that cluster-based Stuff+ predicts real outcomes — that comparison is
   partially validating a model against its own training signal.

6. **Multiple comparisons, unflagged.** ANOVA-style tests run across
   ~40–90 clusters × 5+ outcome metrics with no visible Bonferroni/FDR
   correction — at that many simultaneous tests, some "significant"
   cluster differences are expected by chance alone. Also worth
   independently checking: `src/validation/anova_real_data.py`
   implements the F-test by hand (comment says `# Remove scipy import`)
   rather than calling `scipy.stats.f_oneway` — hand-rolled statistics
   are a common source of quiet bugs (wrong degrees of freedom, pooled
   vs. unpooled variance). Worth a sanity check against scipy on a known
   case before presenting those p-values.

7. **Chase% uses a fixed rectangular strike zone**
   (`zone_z_min, zone_z_max = 1.6, 3.5` in `ProStuff+.ipynb`) instead of
   the actual per-pitch, batter-specific `sz_top`/`sz_bot` columns
   Statcast already provides. Easy, mechanical fix — a 6'5" and a 5'9"
   batter don't share a strike zone, so some chases are currently
   mislabeled.

## Genuine strengths worth stating with confidence

- Fitting the Mapper `clusterer` on the full standardized 9D space while
  using the 2D PCA lens only for cover binning is the textbook-correct
  Mapper construction (Singh–Mémoli–Cárlsson) — the lens was not
  conflated with the clustering metric, a common beginner error.
- LHP mirroring (flipping `pfx_x`/`release_pos_x`, reflecting
  `spin_axis`) to a common RHP frame is standard, defensible practice —
  avoids fragmenting the dataset by handedness while preserving movement
  semantics. Caveat: assumes symmetric effectiveness against batters,
  which platoon splits complicate.
- External, outcome-based validation of the clusters exists (ANOVA,
  variance comparison, predictive comparison against WAR) rather than
  just eyeballing the graph — real scientific hygiene, rarer than it
  should be in TDA application papers.
- Class-weight balancing on the miss/chase classifiers
  (`scale_pos_weight`) is the right call given ~11–19% positive rates,
  and precision/recall/specificity/Brier were reported rather than just
  accuracy — good instinct given the imbalance.

## Bottom line

Not any single bug — it's deciding what claim is actually being made
about the *topology*. Currently defensible as "clustering with a
philosophically motivated construction," but the deck is written as if
the graph structure itself does scientific work. Either find and present
a genuine topological finding (a loop, a branch point, something
connectivity reveals that flat clustering wouldn't), or reframe the MLB
section's claim to be about Mapper as a principled *clustering* method
rather than leaning on "topology" as the payoff. That's the gap between
what the code does and what the theory slides promise — and it's the
next thing to work on.
