# Deck alignment plan (2026-11-15)

Comparison of `presi.tex` (deck) with `docs/paper/paper_final.tex` (paper), from three read-only reviews: theory, results, and narrative/timing. Line numbers are deck lines (`P:`) or paper lines (`F:`) as of 2026-11-15. The aim is not an overhaul. The goal is to fix what is wrong, give time back, and tie the results to the theory.

**Status (2026-11-15):** A done (incl. A12, Eovaldi anonymized). B done: items 1–8 applied, with one change from the plan below. The Dey–Mémoli–Wang bound and the Rips definition went to the end of the Mapper part (new subsection "Checking a Mapper picture"), not onto the Mapper definition frame. B9 was not needed. C done (all 13; C7 uses only the paper's offspeed-overlap sentence, since the paper does not claim clustering separates offspeed from fastballs, and the lens axes are labeled in the caption rather than by regenerating the figure). D done: D1 and D2 in the main talk (the labels frame split into a definition frame with Fig. 11 and a cost frame with Fig. 12 and Jansen); D3 (flight table) and D4 (seams) in backup with pointers. E done: checked against Ester p. 228 (Def. 3 leaves n unbounded, but the remark after Def. 6 that every cluster contains a core point assumes at least one step); k >= 2 added in the paper (with one explaining sentence) and the deck (with a speaker note); ledger row M17 annotated for Will to confirm.

**Throughline:** *Mapper gives the picture; homology and nulls decide what is real; the answer is a continuum that the labels divide.* (Matches the paper's conclusion.)

## Where the time goes now

The deck has 53 frames and no divider frames (no `\AtBeginSection`), so the slide numbers in `CLAUDE.md` are out of date. The estimates below assume about 2 min per dense frame and 1 to 1.5 min per light frame.

| Part | Budget | Now | After the plan |
|---|---|---|---|
| Intro through graphs / simplicial complexes | 15 | ~20 | ~17 |
| PCA/SVD and Mapper | 30 | ~27.5 | ~26 |
| MLB results tied to the theory | 15 | ~27 | ~18 |
| **Total** | 60 | **~74.5** | **~61** |

Nearly all of the overrun is in the results section, and part of it is theory that sits there: the Dey–Mémoli–Wang theorem and the Rips definition.

## A. Fix: wrong or stale now (costs no time)

1. **2026 numbers.**
   - P:631 and the note at P:635: flight-space 2026 loop is **2.04** (was 1.53). The hint now appears in both seasons. Paper: "these data cannot tell this loop apart from chance" (F:658–659).
   - P:695–697: 2026 boundary pitches **150 of 1,565** (was 157 of 1,590); median shift **9.2** (was 9.6); one-SD share **33%** (was 34%).
2. **Conclusion, P:708.** "No loop that replicates across seasons" is now false. Use "no more gaps or loops than a Gaussian null".
3. **Claims the paper has since softened or dropped.**
   - The curveball bump is "also stable" (P:657). The paper claims neither small bump.
   - The slider–cutter "bridge … we dropped it" (P:558). The paper keeps a **band**: it is not structure beyond the continuum, but it describes where sliders and cutters sit, and 68% of SL/FC boundary pitches sit in it.
   - Future work, P:722: "require a minimum pitch count" is already done (Appendix B). Future work also still leads with the development planner, which is backup only.
4. **Vocabulary.**
   - "giant component" → "largest component" (P:538, P:560).
   - "changeups" family → "offspeed" (P:655, P:661).
   - Offspeed codes should include FO (P:502).
   - "features" for data columns → "coordinates" (P:298, P:313, P:564).
   - "loop" needs "independent cycle (cycle rank), not a self-loop" where b₁ is introduced (P:198 or P:205). A graph theorist will read "loop" as a self-loop.
   - "real" loop and "3 units" → multiples of *u* (median H₀ death), with long-lived bars at 3u and persistent loops at 2u, stated as conventions.
5. **Nerve theorem (P:177).** The headline is Bauer's closed convex version, which does not cover the frame's own circle-by-arcs example (arcs are not convex). Promote Hatcher's open-cover version (Cor. 4G.3), as the paper does (F:277–281). "Paracompact" is also undefined at P:183.
6. **Homotopy gloss (P:173).** "Each can be continuously deformed into the other" describes isotopy, not homotopy equivalence. Use "maps both ways whose composites are homotopic to the identities".
7. **Dey–Mémoli–Wang (P:641).**
   - "The nerve map" is never defined. Add the one line on φ from F:344.
   - Write H₁(|N(U)|), with the realization bars.
8. **P:268.** "Steps 1–3 build a cover of P" contradicts P:476 ("non-noise part"). Use "a cover of P apart from noise points" (F:361).
9. **Notation.** [ ] is used for both geometric simplices and abstract chains. Use { } for abstract simplices, as the paper does (F:201).
10. **Missing toy parameters.**
    - The DBSCAN toy (P:416–422) never states its ε = 0.5 but compares it with ε = 0.7.
    - The toy Mapper (P:277–278) never states ε = 0.25, M = 3.

    Take both from F:427 and F:433.
11. **Claim with no source in the paper.** P:280, "Gaussian with the circle's covariance gives b₁ = 0 in all 5 samples", has no number in `numbers.tex` and no counterpart in the paper. Verify it or drop it.
12. **Eovaldi backup (P:766–783).** The deck names him. The paper anonymizes him and uses different summary numbers (1.7 in. of IVB and a 7.6% neighbor median, vs. "≈2 in." and the 5-point gap). Decide which way, and match the numbers.

## B. Give time back: cuts and moves (about 13 min)

| # | Change | Saves |
|---|---|---|
| 1 | **Reeb graphs (P:224) to backup**, with Morse. Drop the phrase at P:279 and the note at P:250. The paper cut Reeb. | ~2–3 min |
| 2 | **Merge Covariance (P:311) into PCA.** Keep S = AᵀA/(m−1). Drop the PSD fact and the spectral-theorem sentence at P:318, per the advisor. | ~2 min |
| 3 | **Merge "Projecting" (P:333) with "Why k = 2" (P:348).** Drop the best rank-k block, which is deck-only and unused. | ~1–2 min |
| 4 | **Move the Dey–Mémoli–Wang statement to the Mapper definition** (P:240), with "we test this bound at the end". Shrink P:638 to two bullets. | ~1.5 min from results |
| 5 | **Move "Test 2: persistent loop" (P:596, the Rips definition) to the end of the Mapper part.** Results then shows only the diagram and table. | ~2 min from results |
| 6 | **Merge "86 loops: are they real?" (P:573) into Test 1 (P:587).** | ~1.5 min |
| 7 | **"Why DBSCAN is correct" (P:455) to backup.** A clause at P:411 keeps the point. | ~2 min |
| 8 | **Fold Future work (P:718) into Conclusions.** Restructure Conclusions as the paper's "for the mathematician / for the baseball reader". | ~1–2 min |
| 9 | Optional, only if Part 1 still runs long: merge Homology (P:188) and Betti (P:202) into one frame with the a, b, c, d example. | ~1.5 min |

## C. Make the connection: callbacks from the results to the theory (about +1 min total)

The theory slides already say "we'll come back to this", but the results slides rarely point back. One line or one thumbnail each:

1. **Slider–cutter hook as a bookend.** Open the talk (P:50) the way the paper opens: a slider and a cutter look alike but are graded on different scales. Close on it, since SL/FC is the most common boundary pair.
2. **G vs K → 86 vs 1.** Add a thumbnail of the a, b, c, d triangle to Test 1 (P:587): "G vs K, 85 times over".
3. **b₁ = E − V + C.** Use it once on the toy circle first: 8 − 8 + 1 = 1 (F:441).
4. **b₀.** Write "b₀ = 9 components" at P:538, and "long H₀ bars = gaps" at P:630.
5. **The nerve theorem's "warning for later" (P:184) is never paid off.** At P:648: "this is the warning: intersections of DBSCAN clusters need not be contractible".
6. **Toy circle → positive control.** At P:280: "this circle returns as our positive control". Better still, reuse `toy_mapper.pdf` at the cover, DBSCAN and graph steps. With overlap under 50%, no point lies in three intervals, so there are no triangles and nerve = graph. The real 2-D cover at 30% puts a point in up to four squares, so triangles appear. That sets up 86 → 1.
7. **PCA/SVD → lens.** Label the lens_scatter axes Av₁, Av₂. Add the paper's point that offspeed overlaps fastballs in the lens but clustering in ℝ⁹ separates them (F:582–586), a callback to "clustering uses all of ℝⁿ" (P:269).
8. **"Linear PCA can flatten curves" (P:353) → "4 lenses, incl. nonlinear Isomap"** at P:549.
9. **Cover knobs → the grid.** Port intervals 8/10/12 × overlap 20/30/40% × ε 0.8/1.0/1.2 at P:557: "Step 2's two knobs + Step 3's ε".
10. **DBSCAN noise → 630 of 3,943 shapes are noise** (P:538, F:599).
11. **Persistence → density peaks.** One phrase at P:652: "same idea, H₀ of the superlevel sets {density ≥ t}".
12. **Intrinsic dimension (P:664).** One line of math: the count within radius r grows like rᵈ (F:553). Add "below the shuffled null".
13. **Boundary pitches (P:683).** "A boundary pitch is defined by its neighbors in shape space, so this is the continuum showing up in grading" (F:712).

## D. Port from the paper (results block)

Ranked for a 15-min mathematical results block:

1. **`label_shift.pdf`** (paper Fig. 12): own-label vs. neighbor-label grade for every boundary pitch, with Jansen's cutter marked (86 among cutters, 106 among four-seamers). It turns the table-only frame at P:683 into a picture.
2. **`boundary_neighborhood.pdf`** (Fig. 11): the boundary-pitch definition as k-nearest neighbors in one picture.
3. **Table 1, `tab:flight`** (paper): nine coordinates vs. flight space in both seasons, with nulls. It replaces the stale bullet at P:631 and shows the two-season replication honestly.
4. **`label_seams.pdf`** (Fig. 13): the four boundary pairs in the break plane. Use it if time allows.

Senga, Wembanyama / Lum et al., tunneling, role-based grading and Moneyball are paper §8 framing. They are optional for a math audience, and best kept for a baseball-audience version of the talk.

## E. Found in the paper, not only the deck

**DBSCAN chain length (F:411, also P:394).** The density-reachability chain q = p₁, …, p_k = p allows k = 1, so every point is density-reachable from itself. Then a lone non-core point p gives a cluster {p}:

- maximality holds, since only p is reachable from p;
- connectivity holds, through o = p.

So noise would be empty. Fix: require k ≥ 2, or require a cluster to contain a core point (as Ester's Lemmas 1–2 effectively do). **Check Ester's exact wording before changing either file.**

## Decisions for Will

- [ ] Apply section A (fixes)? Recommended first; it costs no talk time.
- [ ] Which cuts in B? 1–8 recommended; 9 only if needed.
- [ ] Which callbacks in C? All are one line each.
- [ ] Which ports in D? 1–3 recommended.
- [ ] Eovaldi in the deck backup: name him or anonymize him to match the paper?
- [ ] DBSCAN k ≥ 2 / core-point fix in the paper and the deck (after checking Ester).
- [ ] P:280 Gaussian-circle claim: verify or drop.
