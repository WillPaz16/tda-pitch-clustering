# Methodology review (2026-08-17)

A critical, PhD-committee-style pass over the actual pipeline logic (not
code hygiene — this is about whether the math/stats hold up). Written
before any "discovery" work (finding an actual interpretable topological
result) so the open questions are on record first. Nothing in the
pipeline has been changed as a result of this review yet.

## Why continuous features instead of pitch-type labels — quantified

Motivating point from the user (2026-08-17, not just an anecdote —
checked against this dataset directly): pitch-type tags conflate
pitchers whose actual stuff is nothing alike. E.g. a max-effort closer's
"four-seam fastball" and a soft-tossing starter's "four-seam fastball"
share a label but not much else physically. In this dataset: archetypes
labeled `FF` (four-seam) alone span **79.3–101.2 mph**, a 21.9 mph range
under one label. `SL` (slider) is worse — **34.1–94.2 mph**, a 60.1 mph
spread. That's the concrete numeric version of the motivating claim for
using continuous physical features (and topology) instead of validating
against or relying on pitch-type labels: the labels themselves don't
reliably separate physically distinct pitches, so a method that ignores
them and clusters on raw movement/velocity/spin is doing something a
label-based approach structurally cannot. This is also the flip side of
the slider/cutter bridge finding in docs/DISCOVERY_FINDINGS.md — labels
both over-split (Chapman's and Manaea's fastballs share a label but
aren't the same pitch) and under-split (many real sliders and cutters
are the same shape wearing different labels) relative to what the
physical features actually show.

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

1. **Feature-space mismatch between fit and inference.** ~~The clusters
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
   faithful reproduction.~~

   **FIXED (2026-08-18).** The model was refit (see item 2 below — both
   fixed together in the same refit since they touch the same fit code).
   There is now exactly one `StandardScaler`, fit once on one 9-feature
   matrix, used for the actual `mapper.map()` call and saved for
   inference — no second refit on a different column subset.
   `spin_axis_clock` is gone entirely (superseded by `spin_cos`/`spin_sin`
   from the circularity fix), which is what eliminated the two-feature-
   spaces problem: there's nothing left to accidentally slice
   inconsistently.

2. **`spin_axis` is circular but treated as linear.** ~~It (and its
   derived `spin_axis_clock`) goes into `StandardScaler` and Euclidean
   distance exactly like `release_speed`. 359° and 1° are functionally
   identical directions but ~358 standardized units apart under this
   metric. Should be encoded as $(\cos\theta, \sin\theta)$ to respect the
   circle topology — a real metric-space violation, not just a nitpick,
   given how much the deck emphasizes getting topological definitions
   precise.~~

   Checked directly (2026-08-17) before fixing — see
   `src/tda/spin_axis_circularity_check.py` (kept as a record of the
   pre-fix state, run against the old model). Found: only 20 of 3,932
   training archetypes (0.5%) sat within 20° of the 0/360 wraparound, and
   no cluster centroid was meaningfully distorted by it in the old
   model — but re-encoding `spin_axis` as `(cos, sin)` flipped the
   nearest-cluster assignment for 3 of those 20 near-wraparound points
   (15%), a small but real, verified effect, not hypothetical.

   **FIXED (2026-08-18).** `spin_axis` is now encoded as `(spin_cos,
   spin_sin)` throughout — in the Mapper fit itself
   (`notebooks/TDA_Pitch_Clustering.ipynb`) and in the shared production
   feature-prep helper (`src/tda/tda_classifier.py`'s
   `prepare_pitch_features()`). Averaging `spin_cos`/`spin_sin` per
   `(pitcher, pitch_type)` archetype is also the mathematically correct
   way to average a circular quantity, fixing the related (smaller)
   naive-arithmetic-mean-of-degrees issue in the same motion.
   `models/tda_mapper_model.pkl` was regenerated on the full 2025 season
   (721,799 pitches, 3,943 archetypes, up from 3,932) using the
   previously-tuned hyperparameters
   (`n_cubes=10, perc_overlap=0.3, eps=1, min_samples=4`) rather than
   re-running the full grid search — ponytail: the fix is about feature
   correctness, not hyperparameter optimality; upgrade path is to rerun
   the grid search cell against the corrected feature space if the new
   graph's node count/shape ever looks off. Verified post-refit:
   `classify_pitches_to_csv.py` runs cleanly against the new model, and
   `multi_membership_check.py`/`graph_topology_analysis.py` reproduce the
   same qualitative structure as before (crowded-continuum ambiguity
   unchanged at ~68%, since that's a separate, unaddressed design
   question — item 3 below; giant component + isolated slow-pitch
   outlier sub-components still present) — confirms the discovery-pass
   findings weren't artifacts of the bugs just fixed.

3. **Hard nearest-centroid assignment contradicts the theory being
   presented.** True Mapper allows a point to belong to multiple
   simplices because cover sets overlap — that's what the nerve theorem
   depends on. Inference does `argmin` over centroid distances: single
   nearest cluster, full stop. Fine as engineering, but it means
   production is really "Mapper once, to define archetypes, then 1-NN
   forever after," not "Mapper, generatively." Worth stating explicitly
   rather than letting the audience assume otherwise.

   **Checked directly (2026-08-18)** — see
   `src/tda/multi_membership_check.py`. Quantified how much genuine
   ambiguity the single-label choice discards: for every training
   archetype, computed the margin between its distance to the nearest
   and second-nearest cluster centroid, normalized by the typical
   (median) inter-centroid distance across all 57 clusters. Result:
   **64.6% of archetypes have a top-2 margin under 5%** of that typical
   spacing, 89.4% under 10%, 97.6% under 20% — the large majority of
   points sit almost equidistant between two (or more) clusters. The
   minimum inter-centroid distance across all cluster pairs is exactly
   0.0 — at least two clusters (`cube25_cluster1`/`cube26_cluster0`,
   already flagged in docs/DISCOVERY_FINDINGS.md as near-duplicates) have
   an effectively identical centroid. This makes the theoretical
   objection a practically significant one, not just a precision
   nitpick: the current single-label output is discarding real,
   common multi-membership almost everywhere, not in a rare edge case —
   consistent with the crowded, densely-overlapping continuum found
   during the discovery pass.

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

   **Checked directly (2026-08-17)** — see
   `src/validation/stuff_plus_leakage_check.py`. Used a natural
   experiment already available in the repo:
   `data/pitch_stuffplus_clusters.csv` was built from **2025** Statcast
   data, which the 2022–2024-trained xwOBA/miss/chase models never saw.
   If leakage were masking a model that hadn't actually learned anything
   generalizable, per-pitcher predictions on that genuinely unseen data
   shouldn't correlate with the official in-sample-era predictions.

   **A first pass at this got the wrong answer, worth recording.**
   Comparing the two pipelines' *combined Stuff+ scores* directly gave a
   striking negative correlation (Spearman r=−0.46, p<0.0001) — alarming
   on its face. That conclusion didn't survive inspection: the two
   pipelines use different Stuff+ formulas entirely (the official
   pipeline weights xwOBA/miss/chase equally, 1/3 each; the pipeline that
   produced the 2025 data uses the separately-optimized 0.72/0.11/0.17
   weights from `fitting_stuff_weights.py`), on top of different z-score
   normalization reference populations. Comparing a combined score
   computed two different ways isn't a valid leakage test regardless of
   how the correlation comes out.

   **Redone comparing the raw model predictions directly** (before any
   weighting/combination), which isolates whether the CatBoost models
   themselves generalize: xwOBA Spearman r=0.647, miss r=0.586, chase
   r=0.586 (n=122 pitchers, all p<0.0001) between the official
   2022–2024-era predictions and genuinely-unseen 2025 predictions for
   the same pitchers. That's strong, consistent, significant
   out-of-sample correlation across all three components. **This
   substantially de-risks the leakage concern**: whatever theoretical
   optimism the missing held-out split introduces into the exact final
   numbers, the underlying models demonstrably retain real predictive
   signal a full year removed from training — leakage did not mask a
   model that failed to learn anything. It doesn't rule out a modest
   optimistic bias in the precise Stuff+ values (that would need an
   actual held-out evaluation, not an out-of-sample correlation check),
   but the practical, presentation-relevant risk (that the whole
   downstream Stuff+-vs-outcomes story rests on a model with no real
   signal) is not supported.

   **Reconciled (2026-08-18)** — the two-different-weights split
   described above was itself a real inconsistency, not just a
   comparison-methodology mistake: `models/stuff_plus_per_pitch_type_metadata.json`
   (the weights `StuffPlusCalculator` reads at inference time) was still
   pinned at equal 1/3 weights, while `assign_pitch_stuffplus_clusters.py`
   separately hardcoded the optimized 0.72/0.11/0.17 weights. Since the
   underlying z-scores don't depend on the weights, `weights_used` (and
   `models/stuff_plus_per_pitch_type.csv` / `stuff_plus_pitcher_level.csv`,
   which only need the weighted-combination and rescaling steps redone)
   were updated in place to 0.72/0.11/0.17, and the `ProStuff+.ipynb`
   cell that originally generated this metadata now imports the same
   `OPTIMIZED_STUFFPLUS_WEIGHTS` constant from `stuff_plus_calculator.py`
   instead of hardcoding equal weights, so a future full rerun won't
   regress it.

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

   **Checked directly (2026-08-18)** — see
   `src/validation/anova_multiple_comparisons_check.py`. First,
   `data/anova_results_real_data.csv` turns out to be a single omnibus
   F-test per outcome metric (6 metrics: whiff%, chase%, groundball%,
   flyball%, xwOBA, velocity — each comparing all 48 clusters at once),
   not pairwise cluster-vs-cluster tests — a much smaller
   multiple-comparisons surface than "~40–90 clusters" implied.

   **A bigger problem surfaced while checking this, independent of the
   multiple-comparisons question**: the reported p-values are not real
   p-values. `anova_real_data.py`'s `manual_anova()` computes the
   F-statistic correctly (standard formula, verified), but converts it
   to a p-value with a hardcoded threshold —
   `p_value = 1.0 if f_stat < 3.0 else 0.001` — not the actual
   F-distribution CDF. The comment even says "Approximate p-value
   (rough)." The identical bug is duplicated in
   `cluster_outcome_validation.py`.

   Recomputed the real p-values from the correct F-statistics and known
   degrees of freedom (`scipy.stats.f.sf`): every metric's true p-value
   is astronomically smaller than the fake `0.001` placeholder (e.g.
   whiff% real p ≈ 6×10⁻¹²⁶ vs. reported `0.001`), and **every result
   survives Bonferroni correction across all 6 metrics comfortably**
   (corrected α ≈ 0.0083, real p-values are dozens of orders of
   magnitude below that). So: the specific "significant" conclusions
   already in this file are not wrong, and the multiple-comparisons risk
   for this particular set of 6 tests turns out to be low. But the fake
   p-value logic is a real, separate correctness bug that happened not
   to bite here only because the true effect sizes are enormous — it
   would silently produce a wrong significant/not-significant call on
   any more marginal comparison run through the same function elsewhere
   in the pipeline. Worth fixing (swap in `scipy.stats.f.sf` or
   `scipy.stats.f_oneway`, in both files) independent of anything about
   multiple comparisons.

7. **Chase% uses a fixed rectangular strike zone**
   (`zone_z_min, zone_z_max = 1.6, 3.5` in `ProStuff+.ipynb`) instead of
   the actual per-pitch, batter-specific `sz_top`/`sz_bot` columns
   Statcast already provides. Easy, mechanical fix — a 6'5" and a 5'9"
   batter don't share a strike zone, so some chases are currently
   mislabeled.

   **Checked directly (2026-08-17)** — see
   `src/validation/strike_zone_check.py`, run against one real day of
   Statcast data (2025-09-15, 2,729 pitches). At the aggregate level the
   effect is small: 17.41% chase rate under the fixed zone vs. 18.32%
   under the real per-batter zone, under a 1 percentage point gap. But
   at the individual-pitch level, **2.53% of pitches get a different
   chase/not-chase label** between the two definitions — real
   mislabeling in the training data, not hypothetical. The real strike
   zone top in this one day of data ranged from 2.56 to 4.18 feet across
   batters (bottom: 1.08 to 2.00 feet) — genuine batter-to-batter
   variation the fixed 1.6–3.5 window discards entirely. Small aggregate
   bias, real individual-pitch label noise; an easy, mechanical fix
   (swap the two constants for the `sz_top`/`sz_bot` columns, which are
   already being fetched) worth doing.

   **Side finding, unrelated to the strike zone question but worth
   recording — FIXED (2026-08-18):** `pybaseball.statcast()` failed in
   this environment when first found — the network call succeeded, but
   pybaseball's own postprocessing step crashed on a duplicate-column bug
   (`KeyError: "['pitcher.1', 'fielder_2.1'] not in index"`). Traced to a
   plain version bug: `pip install --upgrade pybaseball` (2.1.1 → 2.2.7)
   resolved it cleanly, verified with a real `statcast()` call. This
   unblocks real multi-season pulls going forward without needing the
   Baseball Savant CSV export workaround
   (`src/tda/tda_classifier.fetch_savant_csv`, still kept and used
   elsewhere for its own reasons — the CSV export bypasses pybaseball
   entirely and was what made the 2025-season Mapper refit and the
   strike-zone check above possible before this fix was found).

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
