"""
Figures for the thesis talk (docs/presentation/figures/*.pdf).

Every figure is built from saved model/analysis outputs, and every number a
slide quotes is printed here so the deck can be checked against it.

  lens_scatter     PCA lens of the 3,943 archetypes, three pitch families
  mapper_graph     the saved Mapper graph, nodes colored by mean velocity
  nerve_b1         graph b1 vs full-nerve b1: real grid fits vs null models
  intrinsic_dim    MLE intrinsic dimension: real vs spin-pair shuffle null
  persistence      H1 persistence diagrams: real archetypes vs Gaussian null
  reeb_torus       textbook example: height function on a torus + Reeb graph
  pitch_design     one four-seamer's design neighborhood: shape vs model score,
                   and model score vs actual 2025 SwStr% (illustrative)

Colors follow the dataviz reference palette: 3 categorical slots (validated
all-pairs, light mode) with direct labels, and a one-hue blue sequential ramp.
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from ripser import ripser
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "tda"))
from tda_classifier import load_tda_model
from mapper_stability import gaussian_null

OUT = ROOT / "docs" / "presentation" / "figures"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # validated all-pairs (light)
OTHER = "#b9b8b2"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
FAMILY = {**dict.fromkeys(["FF", "SI", "FC"], "Fastballs"),
          **dict.fromkeys(["SL", "ST", "SV", "CU", "KC", "CS"], "Breaking balls"),
          **dict.fromkeys(["CH", "FS", "FO", "SC"], "Offspeed")}
FAMILIES = ["Fastballs", "Breaking balls", "Offspeed"]

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight"})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (name + ".pdf"))
    plt.close(fig)
    print("  wrote figures/%s.pdf" % name)


def lens_scatter(orig, cols, ptype):
    X = StandardScaler().fit_transform(orig[cols].values.astype(float))
    pca = PCA(2).fit(X)
    Z = pca.transform(X)
    print("lens_scatter: explained variance PC1 %.3f, PC2 %.3f (sum %.3f)" % (*pca.explained_variance_ratio_, pca.explained_variance_ratio_.sum()))
    fam = np.array([FAMILY.get(t, "Other") for t in ptype])
    print("  family counts:", dict(Counter(fam)))
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    m = fam == "Other"
    ax.scatter(Z[m, 0], Z[m, 1], s=8, c=OTHER, lw=0, label="Other (EP, FA, KN, ...)")
    for f, c in zip(FAMILIES, SERIES):
        m = fam == f
        ax.scatter(Z[m, 0], Z[m, 1], s=8, c=c, lw=0, alpha=0.75, label=f)
        off = {"Fastballs": (0, 22), "Breaking balls": (0, 18), "Offspeed": (-10, -26)}[f]
        ax.annotate(f, np.median(Z[m], axis=0), xytext=off, textcoords="offset points", fontsize=10, fontweight="bold",
                    color=INK, ha="center", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, lw=1.2, alpha=0.9))
    ax.set_xlabel("PC1 (%.0f%% of variance): fastball → breaking ball" % (100 * pca.explained_variance_ratio_[0]))
    ax.set_ylabel("PC2 (%.0f%%): delivery" % (100 * pca.explained_variance_ratio_[1]))
    ax.legend(frameon=False, fontsize=8, loc="lower right", markerscale=2)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    save(fig, "lens_scatter")


def mapper_graph(mc, speed, ptype):
    g = mc["graph"]
    G = nx.Graph(); G.add_nodes_from(g["nodes"]); G.add_edges_from((s, t) for s, ts in g["links"].items() for t in ts)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    print("mapper_graph: %d nodes, %d edges, %d components (giant %d nodes)" % (G.number_of_nodes(), G.number_of_edges(), len(comps), len(comps[0])))
    pos = nx.kamada_kawai_layout(G.subgraph(comps[0]))
    for i, c in enumerate(comps[1:]):            # small components in a column on the right
        for j, n in enumerate(sorted(c)):
            pos[n] = np.array([1.45 + 0.12 * j, 0.9 - 0.25 * i])
    v = {n: speed[g["nodes"][n]].mean() for n in G}
    size = {n: 12 + 4 * np.sqrt(len(g["nodes"][n])) for n in G}
    maj = {n: Counter(ptype[g["nodes"][n]]).most_common(1)[0][0] for n in G}
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#9b9a95", width=0.8)
    sc = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[v[n] for n in G], cmap=BLUES, vmin=75, vmax=96,
                                node_size=[size[n] for n in G], edgecolors=SURF, linewidths=1.5)
    big = [n for n in comps[0] if len(g["nodes"][n]) >= 150] + [n for c in comps[1:] for n in c]
    for n in big:
        ax.annotate(maj[n], pos[n], xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7, color=INK)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01); cb.set_label("mean velocity (mph; slower clipped to 75)", color=INK2); cb.outline.set_visible(False)
    ax.text(1.5, 1.08, "isolated\ncomponents", fontsize=8, color=INK2, ha="center", transform=ax.transData)
    ax.axis("off")
    save(fig, "mapper_graph")


def nerve_b1():
    d = pd.read_csv(ROOT / "data" / "nerve_homology.csv")
    groups = [("grid", "Real data\n(27 parameter fits)"), ("null_gauss", "Gaussian null\n(5 draws)"), ("null_shuffle", "Shuffle null\n(5 draws)")]
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    rng = np.random.default_rng(0)
    for i, (k, lab) in enumerate(groups):
        s = d[d.kind == k]
        for col, c, dx, name in (("graph_b1", SERIES[0], -0.15, "graph (1-skeleton)"), ("nerve_b1", SERIES[1], 0.15, "full nerve")):
            ax.scatter(i + dx + rng.uniform(-0.05, 0.05, len(s)), s[col], s=22, c=c, edgecolors=SURF, lw=1, zorder=3, label=name if i == 0 else None)
        print("nerve_b1: %-12s graph b1 %d-%d (median %g) | nerve b1 %d-%d (median %g)" % (k, s.graph_b1.min(), s.graph_b1.max(), s.graph_b1.median(), s.nerve_b1.min(), s.nerve_b1.max(), s.nerve_b1.median()))
    sm = d[d.kind == "saved_model"].iloc[0]
    ax.annotate("our model: %d → %d" % (sm.graph_b1, sm.nerve_b1), (0.15, sm.nerve_b1), xytext=(0.32, 24), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in groups], fontsize=8)
    ax.set_ylabel("first Betti number $b_1$"); ax.set_ylim(-5, 155)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "nerve_b1")


def intrinsic_dim():
    d = pd.read_csv(ROOT / "data" / "intrinsic_dimension.csv")
    m = d.groupby(["scope", "kind"]).mle.mean().unstack()
    rows = ["all"] + [s for s in m.index if s != "all"]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for y, s in enumerate(reversed(rows)):
        a, b = m.loc[s, "real"], m.loc[s, "null_shuffle_spinpair"]
        ax.plot([a, b], [y, y], color=GRID, lw=2.5, zorder=1)
        ax.scatter(a, y, s=40, c=SERIES[0], edgecolors=SURF, lw=1, zorder=3, label="real archetypes" if y == 0 else None)
        ax.scatter(b, y, s=40, c=SERIES[1], edgecolors=SURF, lw=1, zorder=3, label="fair null (shuffle, spin pair kept)" if y == 0 else None)
        print("intrinsic_dim: %-4s real %.2f | fair null %.2f | gap %.2f" % (s, a, b, b - a))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(["all pitches" if s == "all" else s for s in reversed(rows)], fontsize=8)
    ax.set_xlabel("MLE intrinsic dimension (ambient dimension = 9)"); ax.set_xlim(4, 9)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "intrinsic_dim")


def persistence(orig, cols):
    X_raw = orig[cols].values.astype(float)
    sc = StandardScaler().fit(X_raw)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_raw), 1000, replace=False)
    real = sc.transform(X_raw)[idx]
    gauss = sc.transform(gaussian_null(X_raw, rng))[idx]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.0), sharex=True, sharey=True)
    for ax, (lab, Y) in zip(axes, (("Real archetypes", real), ("Gaussian null", gauss))):
        dg = ripser(Y, maxdim=1)["dgms"]
        h0 = dg[0][np.isfinite(dg[0][:, 1])]
        ax.scatter(h0[:, 0], h0[:, 1], s=8, c=SERIES[0], lw=0, label="$H_0$ (components)")
        ax.scatter(dg[1][:, 0], dg[1][:, 1], s=10, c=SERIES[1], lw=0, label="$H_1$ (loops)")
        hi = 4.3
        ax.plot([0, hi], [0, hi], color=INK2, lw=0.8)
        ax.set_title(lab, fontsize=10, color=INK); ax.set_xlabel("birth")
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        print("persistence: %-16s max H1 persistence %.3f over %d classes" % (lab, (dg[1][:, 1] - dg[1][:, 0]).max(), len(dg[1])))
    axes[0].set_ylabel("death"); axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "persistence")


def reeb_torus():
    """Upright torus, height function f = z: critical values min, 2 saddles, max."""
    R, r = 1.0, 0.38
    u, w = np.meshgrid(np.linspace(0, 2 * np.pi, 80), np.linspace(0, 2 * np.pi, 40))
    x = (R + r * np.cos(w)) * np.cos(u); y = r * np.sin(w); z = (R + r * np.cos(w)) * np.sin(u)
    fig = plt.figure(figsize=(6.2, 3.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(x, y, z, color="#cde2fb", edgecolor="#86b6ef", lw=0.2, alpha=0.9, shade=False)
    for c in (-R - r, -R + r, R - r, R + r):
        ax.plot([-1.6, 1.6], [0, 0], [c, c], color=SERIES[1], lw=0.9, ls="--")
    ax.set_proj_type("ortho"); ax.set_box_aspect((1, 0.5, 1)); ax.view_init(0, -90); ax.axis("off")
    ax.set_title("$X$ = torus, $f$ = height", fontsize=10, color=INK)
    ax2 = fig.add_subplot(1, 2, 2)
    lv = {"min": -R - r, "s1": -R + r, "s2": R - r, "max": R + r}
    ax2.plot([0, 0], [lv["min"], lv["s1"]], color=SERIES[0], lw=2)
    ax2.plot([0, 0], [lv["s2"], lv["max"]], color=SERIES[0], lw=2)
    t = np.linspace(0, np.pi, 50)
    for s in (1, -1):
        ax2.plot(s * 0.45 * np.sin(t), lv["s1"] + (lv["s2"] - lv["s1"]) * (1 - np.cos(t)) / 2, color=SERIES[0], lw=2)
    for k, val in lv.items():
        ax2.scatter(0, val, s=30, c=INK, zorder=3)
        ax2.axhline(val, color=SERIES[1], lw=0.9, ls="--")
    ax2.text(0.6, 0, "the two arcs:\nlevel sets with\n2 components", fontsize=8, color=INK2, va="center")
    ax2.set_xlim(-1, 1.4); ax2.axis("off"); ax2.set_title("Reeb graph $R_f(X)$", fontsize=10, color=INK)
    save(fig, "reeb_torus")


def toy_mapper():
    """Mapper by hand on a noisy circle, lens = height: the graph is a loop (the circle's Reeb graph)."""
    from sklearn.cluster import DBSCAN
    rng = np.random.default_rng(1)
    t = rng.uniform(0, 2 * np.pi, 240); r = 1 + rng.normal(0, 0.04, 240)
    P = np.c_[r * np.cos(t), r * np.sin(t)]
    f = P[:, 1]
    n, ov = 5, 0.3
    w = (f.max() - f.min()) / (1 + (n - 1) * (1 - ov)); step = w * (1 - ov)
    nodes = []                                   # (interval, member indices)
    for i in range(n):
        lo = f.min() + i * step
        idx = np.flatnonzero((f >= lo) & (f <= lo + w))
        lab = DBSCAN(eps=0.25, min_samples=3).fit_predict(P[idx])
        nodes += [(i, idx[lab == c]) for c in sorted(set(lab) - {-1})]
    G = nx.Graph(); G.add_nodes_from(range(len(nodes)))
    G.add_edges_from((a, b) for a in range(len(nodes)) for b in range(a + 1, len(nodes)) if np.intersect1d(nodes[a][1], nodes[b][1]).size)
    b1 = G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)
    print("toy_mapper: %d nodes, %d edges, b1 = %d" % (G.number_of_nodes(), G.number_of_edges(), b1))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.0))
    for i in range(n):
        lo = f.min() + i * step
        a1.axhspan(lo, lo + w, color=SERIES[0], alpha=0.07, lw=0)
        a1.plot([-1.42 + 0.06 * (i % 2)] * 2, [lo, lo + w], color=SERIES[0], lw=2, solid_capstyle="butt")
    a1.scatter(P[:, 0], P[:, 1], s=7, c=INK2, lw=0)
    a1.set_title("$P$, lens $f(x, y) = y$, 5 intervals", fontsize=9, color=INK)
    a1.set_aspect("equal"); a1.axis("off")
    pos = {k: P[m].mean(axis=0) for k, (_, m) in enumerate(nodes)}
    nx.draw_networkx_edges(G, pos, ax=a2, edge_color=INK2, width=1.5)
    nx.draw_networkx_nodes(G, pos, ax=a2, node_color=SERIES[0], node_size=[3 * len(m) for _, m in nodes], edgecolors=SURF, linewidths=1.5)
    a2.set_title("Mapper graph: $b_1 = %d - %d + 1 = %d$" % (G.number_of_edges(), G.number_of_nodes(), b1), fontsize=9, color=INK)
    a2.set_aspect("equal"); a2.set_xlim(-1.5, 1.5); a2.set_ylim(-1.3, 1.3); a2.axis("off")
    save(fig, "toy_mapper")


def dbscan_toy():
    """Core, border and noise points for one (eps, M)."""
    from sklearn.cluster import DBSCAN
    rng = np.random.default_rng(3)
    blob = rng.normal(0, 0.28, (16, 2))
    far = blob[np.argmax(blob[:, 0])]
    P = np.vstack([blob, far + [0.42, 0.05], [[1.25, 0.75], [-1.05, -0.85]]])
    eps, M = 0.5, 4
    db = DBSCAN(eps=eps, min_samples=M).fit(P)   # sklearn counts p itself, as N_eps(p) does
    core = np.zeros(len(P), bool); core[db.core_sample_indices_] = True
    noise = db.labels_ == -1; border = ~core & ~noise
    print("dbscan_toy: %d core, %d border, %d noise (eps=%.1f, M=%d)" % (core.sum(), border.sum(), noise.sum(), eps, M))
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    for k, ls in ((np.flatnonzero(core)[np.argmax(P[core, 0])], "-"), (np.flatnonzero(border)[0], "--")):
        ax.add_patch(plt.Circle(P[k], eps, fill=False, color=INK2, lw=0.8, ls=ls))
    ax.scatter(*P[core].T, s=36, c=SERIES[0], edgecolors=SURF, lw=1, zorder=3, label="core ($|N_\\varepsilon(p)| \\geq M$)")
    ax.scatter(*P[border].T, s=36, facecolors="white", edgecolors=SERIES[0], lw=1.6, zorder=3, label="border (in a cluster, not core)")
    ax.scatter(*P[noise].T, s=40, marker="x", c=SERIES[1], lw=1.8, zorder=3, label="noise")
    ax.set_title("$\\varepsilon = %.1f$, $M = %d$" % (eps, M), fontsize=9, color=INK)
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "dbscan_toy")


def pitch_design():
    plt.rcParams["font.size"] = 13   # shown at reduced width on the slide
    d = pd.read_csv(ROOT / "data" / "pitch_design_example.csv")
    me = d[d.index == 0].iloc[0]; nb = d.iloc[1:]
    top = nb[nb.score >= nb.score.quantile(2 / 3)]
    shift = top[["HB_in", "IVB_in", "mph", "axis_deg"]].median() - me[["HB_in", "IVB_in", "mph", "axis_deg"]].astype(float)
    rho = d[["score", "swstr"]].corr(method="spearman").iloc[0, 1]
    print("pitch_design: %s %s, n=%d, score %.1f, SwStr %.3f; %d neighbors; top third minus self: %s; Spearman(score, SwStr) %.2f"
          % (me["name"], me.pitch_type, me.n, me.score, me.swstr, len(nb), shift.round(1).to_dict(), rho))
    first = lambda s: s.split(", ")[1].split()[0][0] + ". " + s.split(", ")[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"wspace": 0.5})
    sc = a1.scatter(nb.HB_in, nb.IVB_in, c=nb.score, cmap=BLUES, s=46, edgecolors=SURF, lw=1, zorder=3)
    a1.scatter(me.HB_in, me.IVB_in, marker="*", s=200, c=SERIES[1], edgecolors=SURF, lw=1, zorder=4)
    a1.annotate("", (me.HB_in + shift.HB_in, me.IVB_in + shift.IVB_in), (me.HB_in, me.IVB_in),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2), zorder=5)
    a1.annotate(first(me["name"]), (me.HB_in, me.IVB_in), xytext=(-6, -14), textcoords="offset points", fontsize=11, color=INK, ha="right")
    a1.set_xlabel("horizontal break (in, + = arm side)"); a1.set_ylabel("induced vertical break (in)")
    cb = fig.colorbar(sc, ax=a1, fraction=0.05, pad=0.02); cb.set_label("model score", fontsize=11); cb.ax.tick_params(labelsize=10)
    a1.grid(color=GRID, lw=0.6); a1.set_axisbelow(True)
    se = np.sqrt(d.swstr * (1 - d.swstr) / d.n) * 100
    a2.errorbar(nb.score, nb.swstr * 100, yerr=1.96 * se[1:], fmt="o", ms=5, color=SERIES[0], ecolor=GRID, elinewidth=1.5, zorder=3)
    a2.errorbar([me.score], [me.swstr * 100], yerr=[1.96 * se[0]], fmt="*", ms=12, color=SERIES[1], ecolor=GRID, elinewidth=1.5, zorder=4)
    a2.set_xlabel("model score (pooled, 100 = avg)"); a2.set_ylabel("actual 2025 SwStr% (95% CI)")
    a2.text(0.03, 0.97, "Spearman $\\rho$ = %.2f" % rho, transform=a2.transAxes, fontsize=11, color=INK2, va="top")
    a2.grid(color=GRID, lw=0.6); a2.set_axisbelow(True)
    save(fig, "pitch_design")
    plt.rcParams["font.size"] = 10


def development_path(pitcher=695611, ptype="SL"):
    """The planner's path for one pitch: feasible shapes, route, season milestones, value."""
    plt.rcParams["font.size"] = 13   # shown at reduced width on the slide
    pop = pd.read_csv(ROOT / "data" / f"development_population_{pitcher}_{ptype}.csv")
    plan = pd.read_csv(ROOT / "data" / f"development_plan_{pitcher}_{ptype}.csv")
    hb, ivb, v = -12 * pop.raw_pfx_x, 12 * pop.raw_pfx_z, 100 * pop.V
    first = lambda s: s.split(", ")[1].split()[0][0] + ". " + s.split(", ")[0]
    xs = [hb[0]] + list(plan.hb_in); ys = [ivb[0]] + list(plan.ivb_in); vs = [v[0]] + list(100 * plan.V)
    tags = ["now"] + ["season %d\nlike %s %s" % (k, first(n), t) for k, n, t in zip(plan.season_budget, plan.name, plan.pitch_type)]
    print("development_path: %s %s; %d feasible shapes; values %s; gains %s" % (
        pitcher, ptype, len(pop) - 1, [round(x, 2) for x in vs], [round(x - vs[0], 2) for x in vs[1:]]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.55})
    sc = a1.scatter(hb[1:], ivb[1:], c=v[1:], cmap=BLUES, s=40, edgecolors=SURF, lw=0.8, zorder=2)
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        a1.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6), zorder=4)
    a1.scatter(xs[0], ys[0], marker="*", s=320, c=SERIES[1], edgecolors=SURF, lw=1, zorder=5)
    a1.scatter(xs[1:], ys[1:], s=90, facecolors="white", edgecolors=INK, lw=1.6, zorder=5)
    for (x, y), tag, (dx, dy, va) in zip(zip(xs, ys), tags, ((-18, -12, "top"), (4, -14, "top"), (6, 12, "bottom"))):
        a1.annotate(tag, (x, y), xytext=(dx, dy), textcoords="offset points", ha="center", va=va, fontsize=11, color=INK)
    a1.set_xlim(min(xs) - 6, max(xs) + 6); a1.set_ylim(min(ys) - 6, max(ys) + 6)   # zoom on the path's neighbourhood
    cb = fig.colorbar(sc, ax=a1, fraction=0.045, pad=0.02); cb.set_label("value (pp SwStr)", fontsize=11); cb.ax.tick_params(labelsize=10)
    a1.set_xlabel("horizontal break (in, + = arm side)"); a1.set_ylabel("induced vertical break (in)")
    a1.grid(color=GRID, lw=0.6); a1.set_axisbelow(True)
    labels = ["now"] + ["season %d" % k for k in plan.season_budget]
    a2.plot(range(len(vs)), vs, color=SERIES[0], lw=2, marker="o", ms=8, zorder=3)
    for i, val in enumerate(vs[1:], 1):
        a2.annotate("%+.2f" % (val - vs[0]), (i, val), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=11, color=INK)
    a2.set_xticks(range(len(vs))); a2.set_xticklabels(labels)
    a2.set_ylabel("value (pp SwStr)"); a2.set_xlim(-0.4, len(vs) - 0.6); a2.set_ylim(min(vs) - 0.6, max(vs) + 0.9)
    a2.grid(axis="y", color=GRID, lw=0.6); a2.set_axisbelow(True)
    save(fig, "development_path")
    plt.rcParams["font.size"] = 10


def main():
    mc = load_tda_model(ROOT / "models" / "tda_mapper_model.pkl")
    orig, cols = mc["original_data"], list(mc["stuff_columns"])
    ptype = orig.index.get_level_values("pitch_type").values.astype(str)
    speed = orig["release_speed"].values
    lens_scatter(orig, cols, ptype)
    mapper_graph(mc, speed, ptype)
    nerve_b1()
    intrinsic_dim()
    persistence(orig, cols)
    reeb_torus()
    toy_mapper()
    dbscan_toy()
    pitch_design()
    development_path()


if __name__ == "__main__":
    main()
