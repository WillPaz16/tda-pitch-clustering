"""
Every number the paper quotes, generated from saved outputs into
docs/paper/numbers.tex as LaTeX macros. The paper never types a result by
hand, so text and data cannot drift apart (as "55%" and "3,940" did in the
deck). Re-run after any analysis changes:
  python3 src/viz/paper_numbers.py
"""
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import spearmanr
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'data'
N_PITCHES_2025 = 721_799    # full-season pull used for the fit (provenance check, DISCOVERY_FINDINGS.md)
N_PITCHES_2026 = 746_715    # 2026-03-20 .. 10-26 pull used for the replication
N_PITCHES_2024 = 745_350    # 2024-03-20 .. 11-02 pull (development tests)
N_PITCHES_2023 = 732_562    # 2023-03-30 .. 11-01 pull (development tests)


def fmt(x, nd=0):
    """LaTeX number: thousands separators, fixed decimals."""
    return f"{x:,.{nd}f}".replace(',', '{,}').replace('-', '\\ensuremath{-}')   # true minus, text or math


def main():
    m = {}
    mc = pickle.load(open(ROOT / 'models' / 'tda_mapper_model.pkl', 'rb'))
    g = mc['graph']
    G = nx.Graph(); G.add_nodes_from(g['nodes']); G.add_edges_from((s, t) for s, ts in g['links'].items() for t in ts)
    m['NPitches'] = fmt(N_PITCHES_2025)
    m['NPitchesTwentySix'] = fmt(N_PITCHES_2026)
    m['NShapes'] = fmt(len(mc['original_data']))
    m['NFeatures'] = fmt(len(mc['stuff_columns']))
    m['NNodes'], m['NEdges'] = fmt(G.number_of_nodes()), fmt(G.number_of_edges())
    m['NComponents'] = fmt(nx.number_connected_components(G))
    ev = mc['pca'].explained_variance_ratio_
    m['PCone'], m['PCtwo'] = fmt(100 * ev[0], 1), fmt(100 * ev[1], 1)

    nh = pd.read_csv(D / 'nerve_homology.csv')
    sm = nh[nh.kind == 'saved_model'].iloc[0]
    m['GraphBone'], m['NerveBone'] = fmt(sm.graph_b1), fmt(sm.nerve_b1)
    m['NerveRank'] = fmt(sm.graph_b1 - sm.nerve_b1)   # rank of the boundary map, by equation (2)
    meta = json.loads((ROOT / 'models' / 'xwoba_model_metadata.json').read_text())
    m['StuffTrainN'] = fmt(meta['n_train_samples'])
    grid = nh[nh.kind == 'grid']
    m['GridGraphBoneMin'], m['GridGraphBoneMax'] = fmt(grid.graph_b1.min()), fmt(grid.graph_b1.max())
    m['GridNerveBoneMin'], m['GridNerveBoneMax'] = fmt(grid.nerve_b1.min()), fmt(grid.nerve_b1.max())

    idim = pd.read_csv(D / 'intrinsic_dimension.csv').groupby(['scope', 'kind'])[['mle', 'twonn']].mean()
    m['DimMLE'], m['DimTwoNN'] = fmt(idim.loc[('all', 'real'), 'mle'], 2), fmt(idim.loc[('all', 'real'), 'twonn'], 2)
    m['DimNullMLE'] = fmt(idim.loc[('all', 'null_shuffle_spinpair'), 'mle'], 2)

    dp = pd.read_csv(D / 'density_peaks.csv')['persistence_rel'].values
    m['PeakBB'], m['PeakCH'] = fmt(dp[1], 2), fmt(dp[2], 2)

    for tag, word in [('2025', 'TwentyFive'), ('2026', 'TwentySix')]:
        s = json.loads((D / f'label_sensitivity_{tag}_relative_summary.json').read_text())
        m[f'LS{word}Boundary'] = fmt(s['boundary'])
        m[f'LS{word}Eligible'] = fmt(s['eligible'])
        m[f'LS{word}BoundaryPct'] = fmt(100 * s['boundary_frac'])
        m[f'LS{word}Shift'] = fmt(s['median_abs_shift'], 1)
        m[f'LS{word}OneSDPct'] = fmt(100 * s['frac_ge_1sd'])
        m[f'LS{word}TwoSEPct'] = fmt(100 * s['frac_gt_2se'])
        m[f'LS{word}CrossPct'] = fmt(100 * s['frac_cross_100'])
        m[f'LS{word}PctChange'] = fmt(s['median_pct_change'])

    pdv = pd.read_csv(D / 'pitch_development_results.csv')
    pr = pdv[(pdv.scope == 'primary') & (pdv.outcome == 'swstr')].set_index('direction')
    m['DevN'] = fmt(pr.loc['better', 'n'])
    for d, word in [('better', 'Better'), ('random', 'Random'), ('worse', 'Worse')]:
        m[f'Dev{word}Beta'] = fmt(100 * pr.loc[d, 'beta'], 2)
        m[f'Dev{word}Lo'], m[f'Dev{word}Hi'] = fmt(100 * pr.loc[d, 'lo'], 2), fmt(100 * pr.loc[d, 'hi'], 2)
        m[f'Dev{word}P'] = fmt(pr.loc[d, 'p'], 3)
    ex = pdv[pdv.scope == 'exploratory_joint'].set_index('direction')
    m['DevExRandomBeta'], m['DevExRandomP'] = fmt(100 * ex.loc['random', 'beta'], 2), fmt(ex.loc['random', 'p'], 3)
    m['DevExBetterBeta'], m['DevExBetterP'] = fmt(100 * ex.loc['better', 'beta'], 2), fmt(ex.loc['better', 'p'], 3)

    ty = pd.read_csv(D / 'typicality_results.csv')
    tp = ty[(ty.scope == 'primary') & (ty.k == 20) & (ty.outcome == 'swstr')].iloc[0]
    ts = ty[(ty.scope == 'secondary') & (ty.k == 20) & (ty.outcome == 'swstr')].iloc[0]
    m['TypN'] = fmt(tp.n)
    m['TypBeta'], m['TypLo'], m['TypHi'] = fmt(100 * tp.beta, 2), fmt(100 * tp.lo, 2), fmt(100 * tp.hi, 2)
    m['TypSecBeta'], m['TypSecP'] = fmt(100 * ts.beta, 2), fmt(ts.p, 2)

    yg = pd.read_csv(D / 'development_young_2026.csv')
    m['YoungPlanned'], m['YoungClear'] = fmt(len(yg)), fmt(int(yg['clear'].sum()))
    jr = yg[(yg.pitcher == 695611) & (yg.pitch_type == 'SL')].iloc[0]
    m['JumpAge'], m['JumpN'] = fmt(jr.age), fmt(jr.n)
    jp = pd.read_csv(D / 'development_plan_695611_SL.csv')
    jpop = pd.read_csv(D / 'development_population_695611_SL.csv')
    v0 = jpop.V.iloc[0]
    m['JumpGainOne'], m['JumpGainEnd'] = fmt(100 * (jp.V.iloc[0] - v0), 2), fmt(100 * (jp.V.iloc[-1] - v0), 2)
    m['JumpEndSeason'] = fmt(jp.season_budget.iloc[-1])
    m['JumpSweepStart'] = fmt(abs(12 * jpop.raw_pfx_x.iloc[0]), 1)   # glove-side sweep, inches
    m['JumpSweepOne'], m['JumpSweepEnd'] = fmt(-jp.hb_in.iloc[0], 1), fmt(abs(jp.hb_in.iloc[-1]), 1)

    m['NPitchesTwentyFour'], m['NPitchesTwentyThree'] = fmt(N_PITCHES_2024), fmt(N_PITCHES_2023)
    sc = json.loads((D / 'shape_counts_2025.json').read_text())
    m['MedianPitches'] = fmt(sc['median_pitches'])
    m['PctLtFive'], m['PctLtTwenty'] = fmt(100 * sc['frac_lt5']), fmt(100 * sc['frac_lt20'])
    m['NSlow'], m['SlowMedian'], m['SlowQ'] = fmt(sc['n_slow']), fmt(sc['slow_median_pitches']), fmt(sc['slow_q25_pitches'])
    ss = pd.read_csv(D / 'sample_size_check.csv').set_index('min_n')
    for k, word in [(1, 'One'), (10, 'Ten'), (20, 'Twenty')]:
        r = ss.loc[k]
        m[f'SS{word}Shapes'], m[f'SS{word}Slow'], m[f'SS{word}Comp'] = fmt(r.shapes), fmt(r.slow), fmt(r.n_components)
        m[f'SS{word}Iso'] = 'none in a node' if pd.isna(r.slow_isolated) else fmt(100 * r.slow_isolated) + r'\%'
        m[f'SS{word}Rho'] = fmt(r.ordering_rho, 2)

    # Section 7: stability and nulls (data/mapper_stability.csv)
    st = pd.read_csv(D / 'mapper_stability.csv')
    order = lambda g: 100 * np.mean(g['ordering_rho'].dropna() >= 0.7)
    slow = lambda g: 100 * np.mean(g['slow_isolated'].dropna() >= 0.5)
    real = st[st.kind.isin(['grid', 'subsample'])].groupby(['lens', 'kind'])
    nul = st[st.kind.str.startswith('null')].groupby(['lens', 'kind'])
    m['StabOrderMin'] = fmt(real.apply(order).min())
    pca = st[(st.lens == 'pca') & st.kind.isin(['grid', 'subsample'])].groupby('kind').apply(slow)
    m['StabSlowPCALo'], m['StabSlowPCAHi'] = fmt(pca.min()), fmt(pca.max())
    m['StabSlowLo'], m['StabSlowHi'] = fmt(real.apply(slow).min()), fmt(real.apply(slow).max())
    m['NullOrderMin'] = fmt(nul.apply(order).min())
    m['NullSlowLo'], m['NullSlowHi'] = fmt(nul.apply(slow).min()), fmt(nul.apply(slow).max())
    for kind, word in [('null_gauss', 'Gauss'), ('null_shuffle', 'Shuffle')]:
        g = nh[nh.kind == kind]
        m[f'{word}GraphBoneMin'], m[f'{word}GraphBoneMax'] = fmt(g.graph_b1.min()), fmt(g.graph_b1.max())
        m[f'{word}NerveBoneMin'], m[f'{word}NerveBoneMax'] = fmt(g.nerve_b1.min()), fmt(g.nerve_b1.max())
    # persistent homology (data/persistence_check.csv)
    ph = pd.read_csv(D / 'persistence_check.csv')
    rr, gg, ss = ph[ph.kind == 'real'].iloc[0], ph[ph.kind == 'null_gauss'], ph[ph.kind == 'null_shuffle']
    m['PHRealLoops'] = fmt(rr.h1_long_loops_mean)
    m['PHRealLongBars'] = fmt(rr.h0_long_bars)
    m['PHGaussLongBarsLo'], m['PHGaussLongBarsHi'] = fmt(gg.h0_long_bars.min()), fmt(gg.h0_long_bars.max())
    m['PHShuffleLongBarsLo'], m['PHShuffleLongBarsHi'] = fmt(ss.h0_long_bars.min()), fmt(ss.h0_long_bars.max())
    m['PHRealHzeroMedian'], m['PHGaussHzeroMedian'] = fmt(rr.h0_median, 2), fmt(gg.h0_median.mean(), 2)
    # intrinsic dimension per type
    m['DimNullTwoNN'] = fmt(idim.loc[('all', 'null_shuffle_spinpair'), 'twonn'], 2)
    gap = (idim.xs('null_shuffle_spinpair', level='kind')['mle'] - idim.xs('real', level='kind')['mle']).drop('all')
    m['DimTypeGapLo'], m['DimTypeGapHi'] = fmt(gap.min(), 1), fmt(gap.max(), 1)
    # density peaks
    m['PeakCurve'], m['PeakSweep'] = fmt(dp[3], 3), fmt(dp[4], 3)
    m['PeakP'] = fmt(pd.read_csv(D / 'density_peaks.csv')['p_value'].iloc[1], 2)
    pc = pd.read_csv(D / 'density_peaks_pc1.csv')
    pcs = pc[pc.bandwidth >= 0.15]
    m['PCPeaks'] = fmt(pcs.peaks_sig.min())
    m['PCBandLo'], m['PCBandHi'] = fmt(pcs.bandwidth.min(), 2), fmt(pcs.bandwidth.max(), 2)
    # labels and outcomes
    mm = [pd.read_csv(D / f, usecols=['n_memberships']) for f in ('pitch_stuffplus_clusters.csv', 'classified_pitches_2026-04-01_2026-04-07.csv')]
    m['MultiTwentyFive'], m['MultiTwentySix'] = fmt(100 * (mm[0].n_memberships >= 2).mean()), fmt(100 * (mm[1].n_memberships >= 2).mean())
    an = pd.read_csv(D / 'pitch_level_anova_results.csv')
    an = an[an.level == 'pitch'].pivot(index='metric', columns='labeling', values='omega2')
    m['OmegaWhiffLabels'], m['OmegaWhiffCentroid'], m['OmegaWhiffMapper'] = (fmt(100 * an.loc['whiff', c], 1) for c in ['pitch_type', 'centroid_primary', 'mapper_primary'])
    m['OmegaGBLabels'], m['OmegaGBCentroid'], m['OmegaGBMapper'] = (fmt(100 * an.loc['gb', c], 1) for c in ['pitch_type', 'centroid_primary', 'mapper_primary'])
    gr = pd.read_csv(D / 'graph_reliability_correlation.csv')
    m['ReliabilityRho'] = fmt(spearmanr(gr.train_n, gr.roundtrip_accuracy).correlation, 2)

    out = ROOT / 'docs' / 'paper' / 'numbers.tex'
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ['% GENERATED by src/viz/paper_numbers.py -- do not edit by hand'] + \
            [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in m.items()]
    out.write_text('\n'.join(lines) + '\n')
    print(f"wrote {out} ({len(m)} macros)")
    for k, v in m.items():
        print(f"  \\{k} = {v}")


if __name__ == "__main__":
    main()
