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


if __name__ == "__main__":
    main()
