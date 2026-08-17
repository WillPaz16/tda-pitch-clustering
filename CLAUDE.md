# Project notes

TDA Pitch Clustering — Will Paz's graduate thesis. Combines the Mapper
algorithm (Topological Data Analysis) with a CatBoost-based Stuff+ model to
study MLB pitch quality and outcomes. See [docs/REORG_NOTES.md](docs/REORG_NOTES.md)
for the codebase layout and reorganization history.

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
