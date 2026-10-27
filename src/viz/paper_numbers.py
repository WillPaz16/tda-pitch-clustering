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
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'data'
N_PITCHES_2025 = 721_799    # full-season pull used for the fit (provenance check, DISCOVERY_FINDINGS.md)
N_PITCHES_2026 = 746_715    # 2026-03-20 .. 10-26 pull used for the replication


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
