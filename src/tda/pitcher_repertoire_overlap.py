"""
Pitcher Repertoire Overlap in the Slider/Cutter Bridge Region

R&D-flavored follow-up to the verified finding in docs/DISCOVERY_FINDINGS.md
that the graph's real topological crossroads is a slider/cutter bridge band
(85-90mph, mild glove-side break) sitting between the curveball region and
the four-seam fastball region.

A network statistic and a movement chart are useful to a mathematician; a
player-development analyst wants a name (or ID) and an actionable question.
This checks: are there specific real pitchers whose two DIFFERENT labeled
pitch types (e.g. their own slider AND their own cutter) both land in this
bridge region, or in adjacent/overlapping nodes generally? That's the
concrete, actionable version of "this pitcher's two secondary offerings may
not be meaningfully differentiated in movement/velocity space" -- a real
conversation pitching coaches and pitch-design staffs have.

Analysis/reporting script. See docs/DISCOVERY_FINDINGS.md for write-up.

Note: pitcher identity is reported as Statcast/MLBAM pitcher_id, not name --
name resolution via pybaseball.playerid_reverse_lookup() was unavailable in
this environment (a Chadwick register parsing error, not a code issue with
this script). IDs can be cross-referenced manually on Baseball Savant.
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_PATH = _ROOT / 'models' / 'tda_mapper_model.pkl'

# the verified slider/cutter bridge nodes from the betweenness-centrality check
BRIDGE_NODES = ['cube24_cluster0', 'cube32_cluster0', 'cube43_cluster0',
                 'cube42_cluster0', 'cube51_cluster0']


def load_model(model_path=_DEFAULT_MODEL_PATH):
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def build_networkx_graph(mapper_graph):
    G = nx.Graph()
    for node in mapper_graph['nodes']:
        G.add_node(node)
    for src, targets in mapper_graph['links'].items():
        for t in targets:
            G.add_edge(src, t)
    return G


def main():
    mc = load_model()
    graph = mc['graph']
    orig = mc['original_data']
    cs = mc['cluster_summary'].set_index('cluster')

    # map each training archetype (pitcher_id, pitch_type) row index -> ALL clusters it
    # belongs to (a point can legitimately be a member of multiple overlapping nodes)
    idx_to_clusters = defaultdict(set)
    for cid, members in graph['nodes'].items():
        for m in members:
            idx_to_clusters[m].add(cid)

    # pitchers with >=2 DISTINCT pitch types landing in the bridge region.
    # Note: a single archetype can legitimately appear as a member of multiple
    # overlapping bridge nodes (that's Mapper's overlapping cover working as
    # intended) -- so this must be deduplicated by distinct pitch_type label
    # per pitcher, not by raw (pitcher, node) membership rows, or it massively
    # overcounts "different pitches converging" when it's really the same
    # single archetype straddling adjacent cover cells.
    bridge_members = defaultdict(dict)  # pitcher_id -> {pitch_type: set(clusters)}
    for cid in BRIDGE_NODES:
        for m in graph['nodes'].get(cid, []):
            pitcher_id, pitch_type = orig.index[m]
            bridge_members[pitcher_id].setdefault(pitch_type, set()).add(cid)

    print("=== Pitchers with 2+ DISTINCT of their OWN pitch types in the slider/cutter bridge region ===\n")
    multi = {p: v for p, v in bridge_members.items() if len(v) >= 2}
    print(f"{len(multi)} pitchers found (out of {len(bridge_members)} total pitchers with any pitch in the bridge region)\n")
    for pid, pitch_map in sorted(multi.items())[:30]:
        details = []
        for pt, cids in pitch_map.items():
            reps = ", ".join(cids)
            avg_speed = np.mean([cs.loc[c, 'release_speed'] for c in cids])
            details.append(f"{pt}@{avg_speed:.1f}mph (in {reps})")
        print(f"  pitcher_id={pid}: " + " | ".join(details))
    if len(multi) > 30:
        print(f"  ... and {len(multi) - 30} more (full list in the saved CSV)")

    single = {p: v for p, v in bridge_members.items() if len(v) == 1}
    print(f"\n(For comparison: {len(single)} pitchers have only ONE distinct pitch type in the "
          f"bridge region, appearing in multiple overlapping nodes -- that's the Mapper cover "
          f"overlap mechanism, not repertoire overlap, and is excluded from the count above.)")

    # broader check: pitchers whose two different pitch types land in graph-ADJACENT
    # clusters generally (not just the bridge nodes specifically)
    G = build_networkx_graph(graph)
    print("\n=== Pitchers whose two different pitch types land in the SAME or graph-adjacent clusters (any region) ===\n")
    pitcher_pitch_clusters = defaultdict(list)
    for m, (pitcher_id, pitch_type) in enumerate(orig.index):
        if m in idx_to_clusters:
            # a point can belong to multiple nodes; represent membership as the set
            pitcher_pitch_clusters[pitcher_id].append((pitch_type, idx_to_clusters[m]))

    same_or_adjacent = []
    for pid, pitches in pitcher_pitch_clusters.items():
        if len(pitches) < 2:
            continue
        for i in range(len(pitches)):
            for j in range(i + 1, len(pitches)):
                pt_i, cids_i = pitches[i]
                pt_j, cids_j = pitches[j]
                if pt_i == pt_j:
                    continue  # same labeled pitch type, not interesting here
                # true if ANY of pitch i's clusters equals or is adjacent to ANY of pitch j's
                is_same = bool(cids_i & cids_j)
                is_adjacent = any(G.has_edge(a, b) for a in cids_i for b in cids_j if a != b)
                if not (is_same or is_adjacent):
                    continue
                cid_i, cid_j = sorted(cids_i)[0], sorted(cids_j)[0]  # representative for reporting
                same_or_adjacent.append({
                    'pitcher_id': pid, 'pitch_a': pt_i, 'cluster_a': cid_i,
                    'pitch_b': pt_j, 'cluster_b': cid_j,
                    'same_cluster': is_same,
                })

    df = pd.DataFrame(same_or_adjacent)
    print(f"Total pitcher-pitch-pairs with two differently-labeled pitches in the same/adjacent cluster: {len(df)}")
    if len(df):
        print(f"  of which same cluster (movement essentially indistinguishable): {df['same_cluster'].sum()}")
        print(f"  of which graph-adjacent but distinct clusters (very similar, blended at the boundary): {(~df['same_cluster']).sum()}")
        print("\nPitch-type pairs involved (most common combinations):")
        pair_counts = df.apply(lambda r: tuple(sorted([r['pitch_a'], r['pitch_b']])), axis=1).value_counts()
        print(pair_counts.head(15).to_string())

    out_path = _ROOT / 'data' / 'pitcher_repertoire_overlap.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved full pair table to {out_path}")


if __name__ == "__main__":
    main()
