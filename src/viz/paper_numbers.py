"""
Every number the paper quotes, generated from saved outputs into
docs/paper/numbers.tex as LaTeX macros. The paper never types a result by
hand, so text and data cannot drift apart (as "55%" and "3,940" did in the
deck). Re-run after any analysis changes:
  python3 src/viz/paper_numbers.py

Paper (docs/paper/paper_final.tex): writes docs/paper/numbers.tex, every number the paper
quotes. A few blocks (development tests, typicality, multi-membership, outcome
ANOVA, reliability) produce macros paper_final.tex does not use; they are kept
for the deck and earlier drafts.
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
N_PITCHES_2026 = 727_721    # 2026-03-20 .. 10-26 pull, spring training (game_type S, 18,994 pitches) removed; regular season + postseason
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
    # what the components hold (Section 7.2): the giant one, slow-only ones, other small ones, and DBSCAN noise
    WORD = dict(enumerate(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']))
    spd = mc['original_data']['release_speed'].values
    comps = [sorted({i for n in c for i in g['nodes'][n]}) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    small = comps[1:]
    slow_only = [c for c in small if (spd[c] < 70).all()]
    m['GiantShapes'] = fmt(len(comps[0]))
    m['NSlowComps'], m['NOtherComps'] = WORD[len(slow_only)], WORD[len(small) - len(slow_only)]
    m['NoiseShapes'] = fmt(len(spd) - len({i for c in comps for i in c}))
    ev = mc['pca'].explained_variance_ratio_
    m['PCone'], m['PCtwo'] = fmt(100 * ev[0], 1), fmt(100 * ev[1], 1)
    # linear check on dimension (Section 7.3.3): full PCA of the standardized shapes
    from sklearn.decomposition import PCA
    cols9 = list(mc['stuff_columns'])
    full = PCA().fit(mc['scaler'].transform(mc['original_data'][cols9].values.astype(float)))
    m['PCASixPct'] = fmt(100 * full.explained_variance_ratio_[:6].sum(), 1)
    top2 = lambda k: set(np.array(cols9)[np.argsort(-np.abs(full.components_[k]))[:2]])
    assert top2(8) == {'pfx_x', 'spin_sin'} and top2(7) == {'pfx_z', 'spin_cos'} \
        and top2(6) == {'release_speed', 'release_spin_rate'}, "smallest PCs changed; update the Section 7.3.3 text"

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
    assert m['LSTwentyFivePctChange'] == m['LSTwentySixPctChange'], "Section 8.1 says 'in both seasons'; update the paper text"

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
    # what the long-lived H0 bars split off (Section 7.3.2): H0 deaths of a Rips filtration are the
    # edge lengths of a minimum spanning tree, so cutting each long edge shows the piece it separates
    from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
    from scipy.spatial.distance import pdist, squareform
    Zs = mc['scaler'].transform(mc['original_data'][mc['stuff_columns']].values.astype(float))
    mst = minimum_spanning_tree(squareform(pdist(Zs))).toarray()
    unit = np.median(mst[mst > 0])
    pieces = []
    for i, j in zip(*np.nonzero(mst > 3 * unit)):
        cut = mst.copy(); cut[i, j] = 0
        _, lab = connected_components(cut, directed=False)
        pieces.append(np.bincount(lab).min())
    assert len(pieces) == rr.h0_long_bars, "MST long edges disagree with persistence_check.csv"
    m['PHLongSingles'], m['PHLongMaxGroup'] = fmt(sum(p == 1 for p in pieces)), fmt(max(pieces))
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

    # flight-space robustness check (src/tda/flight_space_check.py)
    fm = pd.read_csv(D / 'flight_space_2025_pca_mapper.csv')
    fb = fm[(fm.kind == 'grid') & (fm.n_cubes == 10) & (fm.perc_overlap == 0.3) & (fm.eps == 1.0)].iloc[0]
    fg = fm[fm.kind == 'grid']; fn = fm[fm.kind.str.startswith('null')]
    m['FlComp'], m['FlGraphBone'], m['FlNerveBone'] = fmt(fb.n_components), fmt(fb.graph_b1), fmt(fb.nerve_b1)
    m['FlGridCompMax'] = fmt(fg.n_components.max())
    m['FlGridNerveMin'], m['FlGridNerveMax'] = fmt(fg.nerve_b1.min()), fmt(fg.nerve_b1.max())
    m['FlNullNerveMin'], m['FlNullNerveMax'] = fmt(fn.nerve_b1.min()), fmt(fn.nerve_b1.max())
    m['FlSlowIso'] = fmt(100 * np.mean(fm[fm.kind.isin(['grid', 'subsample'])].slow_isolated.dropna() >= 0.5))
    fp = pd.read_csv(D / 'flight_space_2025_persistence.csv')
    m['FlLongBars'] = fmt(fp[fp.kind == 'real'].h0_long_bars.iloc[0])
    fpg = fp[fp.kind == 'null_gauss']
    m['FlGaussLongBarsLo'], m['FlGaussLongBarsHi'] = fmt(fpg.h0_long_bars.min()), fmt(fpg.h0_long_bars.max())
    fd = pd.read_csv(D / 'flight_space_2025_idim.csv').groupby(['scope', 'kind'])[['mle', 'twonn']].mean()
    m['FlDimMLE'], m['FlDimNullMLE'] = fmt(fd.loc[('all', 'real'), 'mle'], 2), fmt(fd.loc[('all', 'null_shuffle_spinpair'), 'mle'], 2)

    allf = pd.concat([pd.read_csv(f) for f in sorted(D.glob('flight_space_20*_*_mapper.csv'))])
    realf = allf[allf.kind.isin(['grid', 'subsample'])]
    m['FlRuns'] = fmt(len(list(D.glob('flight_space_20*_*_mapper.csv'))))
    m['FlAllNerveMax'] = fmt(realf.nerve_b1.max())
    m['FlAllSlowIso'] = fmt(100 * (realf.slow_isolated.dropna() >= 0.5).mean())
    fp6 = pd.read_csv(D / 'flight_space_2026_persistence.csv')
    m['FlSixLongBars'], m['FlSixLoops'] = fmt(fp6[fp6.kind == 'real'].h0_long_bars.iloc[0]), fmt(fp6[fp6.kind == 'real'].h1_long_loops_mean.iloc[0])
    g6 = fp6[fp6.kind == 'null_gauss']
    m['FlSixGaussLo'], m['FlSixGaussHi'] = fmt(g6.h0_long_bars.min()), fmt(g6.h0_long_bars.max())
    fd6 = pd.read_csv(D / 'flight_space_2026_idim.csv').groupby(['scope', 'kind'])['mle'].mean()
    m['FlSixDim'], m['FlSixDimNull'] = fmt(fd6[('all', 'real')], 2), fmt(fd6[('all', 'null_shuffle_spinpair')], 2)
    # 2026 flight-space Mapper (PCA lens) and largest-loop persistence, for the feature-space table
    fm6 = pd.read_csv(D / 'flight_space_2026_pca_mapper.csv')
    fb6 = fm6[(fm6.kind == 'grid') & (fm6.n_cubes == 10) & (fm6.perc_overlap == 0.3) & (fm6.eps == 1.0)].iloc[0]
    fg6, fn6 = fm6[fm6.kind == 'grid'], fm6[fm6.kind.str.startswith('null')]
    m['FlSixComp'], m['FlSixGridCompMax'] = fmt(fb6.n_components), fmt(fg6.n_components.max())
    m['FlSixGridNerveMin'], m['FlSixGridNerveMax'] = fmt(fg6.nerve_b1.min()), fmt(fg6.nerve_b1.max())
    m['FlSixNullNerveMin'], m['FlSixNullNerveMax'] = fmt(fn6.nerve_b1.min()), fmt(fn6.nerve_b1.max())
    m['FlSixSlowIso'] = fmt(100 * np.mean(fm6[fm6.kind.isin(['grid', 'subsample'])].slow_isolated.dropna() >= 0.5))
    nn = nh[nh.kind.isin(['null_gauss', 'null_shuffle'])]
    m['NullNerveBoneMin'], m['NullNerveBoneMax'] = fmt(nn.nerve_b1.min()), fmt(nn.nerve_b1.max())
    # boundary-pitch label pairs, 2025
    lb = pd.read_csv(D / 'label_sensitivity_2025_relative.csv')
    lb = lb[lb.boundary_share >= 0.5]
    pairs = lb.apply(lambda r: ''.join(sorted([r.pitch_type, r.alt_label])), axis=1).value_counts()
    for pr in ('FCSL', 'CHFS', 'CUKC', 'SLST'):
        m[f'LSPair{pr}'] = fmt(pairs[pr])
    assert list(pairs.index[:4]) == ['FCSL', 'CHFS', 'CUKC', 'SLST'], "boundary-pair order changed; update the paper text"
    # named FC/FF example (Section 8.2): Kenley Jansen's cutter
    m['LSPairFCFF'] = fmt(pairs['FCFF'])
    kj = lb[(lb.name == 'Jansen, Kenley') & (lb.pitch_type == 'FC')].iloc[0]
    assert kj.alt_label == 'FF' and kj.grade_own < 100 < kj.grade_alt, "Jansen example changed; update the paper text"
    m['JansenNbFF'] = fmt(round(20 * kj.boundary_share))
    m['JansenOwnGrade'], m['JansenAltGrade'] = fmt(kj.grade_own), fmt(kj.grade_alt)
    # slider-cutter band (Section 7.3.1): do the SL/FC boundary pitches sit in the band's nodes?
    # band = top-5 betweenness nodes of the giant component with a SL/FC majority (as in mapper_stability.top_betweenness)
    from collections import Counter
    pt_all = mc['original_data'].index.get_level_values('pitch_type').astype(str).values
    key_of = {k: i for i, k in enumerate(zip(mc['original_data'].index.get_level_values(0), pt_all))}
    giant = max(nx.connected_components(G), key=len)
    btw = nx.betweenness_centrality(G.subgraph(giant))
    band = {i for n in sorted(btw, key=btw.get, reverse=True)[:5]
            if Counter(pt_all[mc['graph']['nodes'][n]]).most_common(1)[0][0] in ('SL', 'FC') for i in mc['graph']['nodes'][n]}

    def in_band(df):      # (shapes matched to the Mapper model, of which in the band)
        hits = [h for h in (key_of.get((r.pitcher_id, r.pitch_type)) for r in df.itertuples()) if h is not None]
        return len(hits), sum(h in band for h in hits)
    n_sf, k_sf = in_band(lb[lb.apply(lambda r: {r.pitch_type, r.alt_label} == {'SL', 'FC'}, axis=1)])
    allsf = pd.read_csv(D / 'label_sensitivity_2025_relative.csv')
    n_all, k_all = in_band(allsf[allsf.pitch_type.isin(['SL', 'FC'])])
    m['BandBoundaryIn'], m['BandBoundaryN'], m['BandAllPct'] = fmt(k_sf), fmt(n_sf), fmt(100 * k_all / n_all)
    m['BandBoundaryPct'] = fmt(100 * k_sf / n_sf)
    # worked example (anonymized): the boundary pitch whose grade crosses 100 with |shift| closest to the median
    NAMES = dict(FF='four-seam fastball', SI='sinker', FC='cutter', SL='slider', ST='sweeper', SV='slurve',
                 CU='curveball', KC='knuckle curve', CH='changeup', FS='splitter')
    cross = lb[(lb.grade_own - 100) * (lb.grade_alt - 100) < 0]
    ex = cross.loc[(cross['shift'].abs() - lb['shift'].abs().median()).abs().idxmin()]
    m['ExOwnName'], m['ExAltName'] = NAMES[ex.pitch_type], NAMES[ex.alt_label]
    m['ExOwnGrade'], m['ExAltGrade'] = fmt(ex.grade_own), fmt(ex.grade_alt)
    bn = json.loads((D / 'boundary_neighborhood.json').read_text())   # thesis_figures.py boundary_neighborhood
    assert (bn['own'], bn['alt']) == (ex.pitch_type, ex.alt_label), "neighborhood figure is for a different example; rerun it"
    m['ExNbOwn'], m['ExNbAlt'] = fmt(bn['n_own']), fmt(bn['n_alt'])
    # illustrative pitch-design example (src/tda/pitch_design_neighborhood.py), anonymized in the paper
    pd_ex = pd.read_csv(D / 'pitch_design_example.csv')
    me, nbr = pd_ex.iloc[0], pd_ex.iloc[1:]
    top = nbr[nbr.score >= nbr.score.quantile(2 / 3)]
    m['DesignNeighbors'] = fmt(len(nbr))
    m['DesignSwStr'], m['DesignNbSwStr'] = fmt(100 * me.swstr, 1), fmt(100 * nbr.swstr.median(), 1)
    m['DesignIVBGain'] = fmt(top.IVB_in.median() - me.IVB_in, 1)
    assert (nbr.score > me.score).all() and (nbr.swstr > me.swstr).all(), "design example is no longer last; update the paper text"
    # the paper's running example (src/tda/circle_example.py)
    ce = pd.read_csv(D / 'circle_example.csv')
    cn = ce[ce.kind == 'null_gauss']
    m['CircNullGraphBoneMax'] = fmt(cn.graph_b1.max())
    m['CircNullCompLo'], m['CircNullCompHi'] = fmt(cn.n_components.min()), fmt(cn.n_components.max())
    # Figure 7, the persistent loop on a 40-point noisy circle (src/viz/thesis_figures.py, rips_toy)
    rt = json.loads((D / 'rips_toy.json').read_text())
    m['RipsBirth'], m['RipsDeath'] = fmt(rt['birth'], 3), fmt(rt['death'], 3)
    m['RipsPers'], m['RipsNext'] = fmt(rt['persistence'], 3), fmt(rt['next_persistence'], 3)
    for word, p in [('PH', ph), ('Fl', fp), ('FlSix', fp6)]:
        g = p[p.kind == 'null_gauss'].h1_max_persistence
        m[f'{word}MaxPers'] = fmt(p[p.kind == 'real'].h1_max_persistence.iloc[0], 2)
        m[f'{word}GaussMaxPersLo'], m[f'{word}GaussMaxPersHi'] = fmt(g.min(), 2), fmt(g.max(), 2)

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
