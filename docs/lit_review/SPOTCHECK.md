# Check 3: human spot-check sheet

For each item, open the PDF in `Lit Review/`, go to the **PDF page** (the page number your PDF viewer shows), and confirm the source says what the claim says, with the same hypotheses. Mark ✅ or ❌ and add a note. Items marked DECK FIX are where the slide currently disagrees with the source.

Random sample seed: 20260928 (reproducible).

## Required (21): manual items, slide fixes, theorems/definitions

- [ ] **T01** Definition of a topological space (open-set axioms)
    - `` → page TBD (Will, check 3)
    - ⚠️ manual: page number needed from your copy
- [ ] **T02** Subspace topology definition
    - `algebraic-topology.pdf` → wording source, PDF p3
    - ⚠️ manual: read the page image (scan/OCR symbols)
- [ ] **T08** Definition of an n-simplex
    - `Hatcher_AlgebraicTopology.pdf` → Sec. 2.1, p. 103 (PDF p112)
    - `Mohnhaupt_Bsc.pdf` → Def. 2.1.2, p. 4 (PDF p14)
    - `Borsuk1948_FundMath35.pdf` → Sec. 2, p. 219 (simplex = minimal convex set spanned by linearly independent points) [VISUAL]
    - ⚠️ manual: read the page image (scan/OCR symbols)
- [ ] **T09** Face of a simplex
    - `Mohnhaupt_Bsc.pdf` → Def. 2.1.2, p. 4 (PDF p14)
    - `Borsuk1948_FundMath35.pdf` → Sec. 2, p. 219 (face) [VISUAL]
    - ⚠️ manual: read the page image (scan/OCR symbols)
- [ ] **T10** Definition of a simplicial complex (closed under faces; intersections are faces)
    - `Topological Persistence and Simplification.pdf` → Sec. 2, p. 513 (PDF p3)
    - `Mohnhaupt_Bsc.pdf` → Def. 2.1.3, p. 4 (PDF p14)
    - `Borsuk1948_FundMath35.pdf` → Sec. 2, p. 219 (simplicial complex) [VISUAL]
    - ⚠️ manual: read the page image (scan/OCR symbols)
- [ ] **T11** Nerve of a cover
    - `Hatcher_AlgebraicTopology.pdf` → Sec. 3.3, p. 256 (PDF p265)
    - `Mohnhaupt_Bsc.pdf` → Def. 2.1.8, p. 5 (PDF p15)
    - `Borsuk1948_FundMath35.pdf` → Sec. 4, p. 224 (nerve credited to Alexandroff, Math. Ann. 98 (1928) p. 634, fn. 4) [VISUAL]
    - ⚠️ manual: read the page image (scan/OCR symbols)
- [ ] **T12** Nerve theorem statement and hypotheses  — *DECK FIX*
    - `A unified view on the functorial nerve theorem and its variations.pdf` → Thm 3.9, p. 20 (PDF p20)
    - `A unified view on the functorial nerve theorem and its variations.pdf` → Table 1, p. 5 (PDF p5)
    - `Hatcher_AlgebraicTopology.pdf` → Cor. 4G.3, p. 459 (PDF p468)
    - `Borsuk1948_FundMath35.pdf` → Sec. 9 Corollary 3, p. 234 (scan 10 left); decomposition defined Sec. 4 p. 224; intro p. 217 describes 'regular' (AR sets with AR intersections) [VISUAL, image-only scan]
    - ⚠️ manual: read the page image (scan/OCR symbols)
- [ ] **T17** Pullback cover and topological graph / Mapper as nerve of pullback cover  — *DECK FIX*
    - `mapperPBG.pdf` → Sec. 2.1, no printed page on PDF (PDF p3)
- [ ] **T19** Persistent homology and persistence diagrams
    - `Topological Persistence and Simplification.pdf` → Sec. 3, Eq. (3), p. 517 (PDF p7)
    - `Computing Persistent Homology.pdf` → Abstract, p. 249 (PDF p1)
- [ ] **T21** Nerve map induces a surjection on H1 for path-connected covers (Thm 8); Mapper version b1(Mapper) <= b1(X) (Thm 18)
    - `DeyMemoliWang2017_arXiv1703.07387.pdf` → Thm 8, arXiv v1 (PDF p6)
    - `DeyMemoliWang2017_arXiv1703.07387.pdf` → Thm 18, arXiv v1 (PDF p9)
    - `DeyMemoliWang2017_LIPIcs_SoCG.pdf` → Thm 18 (published), 36:8 (PDF p8)
- [ ] **T22** Quotient onto the Reeb space is surjective on H1
- [ ] **M07** PCA is never computed by eigendecomposition; SVD used for numerical stability  — *DECK FIX*
    - `Principal component analysis a review and recent developments.pdf` → Sec. 1, p. 2 (PDF p2)
- [ ] **M09** Eigenvalues of covariance = sigma_i^2/(m-1)
    - `Principal component analysis a review and recent developments.pdf` → Sec. 2, Eqs. (2.2)-(2.3), p. 3 (PDF p3)
- [ ] **M10** Truncated SVD gives best rank-k approximation (Frobenius)  — *DECK FIX*
    - `Axler2024_LADR4e.pdf` → Thm 7.92, p. 284 (PDF p298)
- [ ] **M11** Proportion / cumulative variance explained
    - `Principal component analysis a review and recent developments.pdf` → Sec. 2, Eq. (2.6), p. 4 (PDF p4)
- [ ] **M15** DBSCAN chosen for its popularity across Mapper applications  — *DECK FIX*
    - `s41060-025-00971-0.pdf` → Sec. 2 (clustering usage summary), p. 10 (PDF p10)
- [ ] **M16** DBSCAN: eps-neighborhood; directly density-reachable (core condition)  — *DECK FIX*
    - `Ester1996_KDD96_AAAI.pdf` → Def. 1, p. 227 (PDF p2)
    - `Ester1996_KDD96_AAAI.pdf` → Def. 2, p. 228 (PDF p3)
    - ⚠️ manual: read the page image (scan/OCR symbols)
- [ ] **M17** DBSCAN: density-reachable and density-connected  — *DECK FIX*
    - `Ester1996_KDD96_AAAI.pdf` → Def. 3, p. 228 (PDF p3)
    - `Ester1996_KDD96_AAAI.pdf` → Def. 4, p. 228 (PDF p3)
    - ⚠️ manual: read the page image (scan/OCR symbols)
- [ ] **M18** DBSCAN: cluster (maximality + connectivity) and noise  — *DECK FIX*
    - `Ester1996_KDD96_AAAI.pdf` → Def. 5, p. 228 (PDF p3)
    - `Ester1996_KDD96_AAAI.pdf` → Def. 6, p. 228 (PDF p3)
    - ⚠️ manual: read the page image (scan/OCR symbols)
- [ ] **M19** DBSCAN lemma: cluster = set density-reachable from any core point  — *DECK FIX*
    - `Ester1996_KDD96_AAAI.pdf` → Lemma 2, p. 228 (PDF p3)
    - ⚠️ manual: read the page image (scan/OCR symbols)
- [ ] **R08** Applied Mapper work rarely checks graph features against null models  — *DECK FIX*
    - `CarriereMichelOudot2018_JMLR.pdf` → Abstract, p. 1 (PDF p1)
    - `s41060-025-00971-0.pdf` → Sec. 3, p. 16 (PDF p16)

## Random 20% sample of the rest (4)

- [ ] **M01** Lens/filter function definition
    - `mapperPBG.pdf` → Sec. 3, no printed page on PDF (PDF p4)
- [ ] **M21** Mapper graph construction (nodes = clusters; edges = nonempty intersections)
    - `mapperPBG.pdf` → Sec. 2.1, no printed page on PDF (PDF p3)
- [ ] **R01** Mapper stability/parameter-selection approach (grid + resampling)
    - `CarriereMichelOudot2018_JMLR.pdf` → Abstract, p. 1 (PDF p1)
- [ ] **T14** Definition of the Reeb graph
    - `Structure and Stability of the One-Dimensional Mapper.pdf` → Sec. 1, p. 1334 (PDF p2)
