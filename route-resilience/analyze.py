"""
analyze.py
----------
Phase III — Network Analysis & Stress Testing.

For the healed graph, we compute:
    1.  Betweenness centrality     — identifies the "gatekeeper" nodes
    2.  Criticality score          — normalised betweenness per node
    3.  Baseline global efficiency — harmonic mean of inverse shortest paths
    4.  Node ablation simulation   — remove nodes from highest to lowest
                                     centrality, recompute efficiency, build
                                     a Resilience Index curve
    5.  "Criticality worth" per *edge* — sum of centrality of its endpoints,
                                         used by the dashboard heatmap.

The Resilience Index used here is the ratio of post-perturbation
efficiency to baseline efficiency.  An index of 1.0 means the network is
unaffected; 0.0 means it has collapsed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from skeletonize import Edge, Node


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class AnalyzeConfig:
    sample_size: int = 200     # for approximate betweenness on large graphs
    top_k_ablations: int = 30   # how many of the top nodes to ablate


# --------------------------------------------------------------------------- #
# Build NetworkX graph
# --------------------------------------------------------------------------- #
def to_networkx(nodes: List[Node], edges: List[Edge]) -> nx.Graph:
    G = nx.Graph()
    for n in nodes:
        G.add_node(n.id, y=n.y, x=n.x, kind=n.kind)
    for e in edges:
        # Edge weight = 1 / length (lengths used as "cost" for routing)
        G.add_edge(e.a, e.b,
                   length=e.geom_length,
                   weight=e.geom_length,
                   pixels=e.pixels,
                   is_bridge=getattr(e, "is_bridge", False))
    return G


# --------------------------------------------------------------------------- #
# Centrality
# --------------------------------------------------------------------------- #
def betweenness(G: nx.Graph, cfg: AnalyzeConfig) -> Dict[int, float]:
    if G.number_of_nodes() == 0:
        return {}
    if G.number_of_nodes() > 2000:
        return nx.betweenness_centrality(G, k=cfg.sample_size, seed=0,
                                          normalized=True, weight="length")
    return nx.betweenness_centrality(G, normalized=True, weight="length")


def criticality_scores(G: nx.Graph,
                       betw: Dict[int, float]) -> Tuple[Dict[int, float], List[int]]:
    """Return (node_id -> score in [0,1]) and the list of node ids sorted
    descending by score (most critical first)."""
    if not betw:
        return {}, []
    mx = max(betw.values())
    if mx <= 0:
        return {n: 0.0 for n in betw}, []
    norm = {n: v / mx for n, v in betw.items()}
    return norm, sorted(norm, key=norm.get, reverse=True)


def edge_criticality(G: nx.Graph, node_score: Dict[int, float]) -> Dict[Tuple[int, int], float]:
    """Criticality of an edge = mean score of its two endpoints."""
    out: Dict[Tuple[int, int], float] = {}
    for u, v in G.edges():
        a = node_score.get(u, 0.0)
        b = node_score.get(v, 0.0)
        out[(u, v)] = 0.5 * (a + b)
    return out


# --------------------------------------------------------------------------- #
# Global efficiency & resilience index
# --------------------------------------------------------------------------- #
def global_efficiency(G: nx.Graph) -> float:
    """Efficiency = average of 1 / d(i, j) over all pairs (Harmonic mean)."""
    if G.number_of_nodes() < 2:
        return 0.0
    n = G.number_of_nodes()
    s = 0.0
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="length"))
    for src, dests in lengths.items():
        for d, L in dests.items():
            if src == d:
                continue
            if L > 0:
                s += 1.0 / L
    return s / (n * (n - 1))


def resilience_curve(G: nx.Graph, ordered_nodes: List[int], cfg: AnalyzeConfig
                     ) -> List[dict]:
    """Sequentially remove nodes from highest to lowest centrality and
    measure the remaining global efficiency after each removal."""
    base_eff = global_efficiency(G)
    if base_eff == 0:
        return []
    Gc = G.copy()
    curve: List[dict] = []
    removed = 0
    for nid in ordered_nodes[:cfg.top_k_ablations]:
        if nid not in Gc:
            continue
        Gc.remove_node(nid)
        removed += 1
        eff = global_efficiency(Gc)
        curve.append({
            "step": removed,
            "removed_node": nid,
            "efficiency": eff,
            "resilience_index": eff / base_eff,
            "components": nx.number_connected_components(Gc),
        })
    return curve


def top_gatekeepers(ordered: List[int], nodes: List[Node], k: int = 10
                    ) -> List[dict]:
    nid_to_node = {n.id: n for n in nodes}
    out = []
    for nid in ordered[:k]:
        n = nid_to_node.get(nid)
        if n is None:
            continue
        out.append({"id": n.id, "y": n.y, "x": n.x, "kind": n.kind})
    return out


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    G = nx.Graph()
    # Two paths, sharing a single bottleneck node in the middle
    for i in range(10):
        G.add_node(i)
    for i in range(4):
        G.add_edge(i, i + 1, length=1.0)
    for i in range(5, 9):
        G.add_edge(i, i + 1, length=1.0)
    G.add_edge(4, 5, length=1.0)
    b = betweenness(G, AnalyzeConfig())
    print("betweenness:", {k: round(v, 3) for k, v in b.items()})
    eff = global_efficiency(G)
    print("efficiency:", round(eff, 4))
    order = sorted(b, key=b.get, reverse=True)
    curve = resilience_curve(G, order, AnalyzeConfig(top_k_ablations=5))
    for c in curve:
        print(c)
