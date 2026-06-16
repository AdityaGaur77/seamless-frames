"""
heal.py
-------
Phase II — step 2.

Given a set of nodes and edges from `skeletonize.extract_skeleton`, the
network is typically fragmented (canopy, shadow, occlusion, segmentation
errors).  We bridge the gaps with a *gated* MST over endpoints:

  1.  All endpoint nodes are potential bridge candidates.
  2.  Each candidate pair (u, v) is scored by
          (a) Euclidean distance d(u, v)  — must be < d_max
          (b) angular alignment           — the new segment's direction
                                            must match both endpoints'
                                            tangent directions
                                            (angle < ang_max degrees)
          (c) straightness                — no other endpoint inside the
                                            corridor of the proposed link
  3.  We build a sparse graph of valid candidates and run a Minimum
      Spanning Tree (MST) on it.  Adding the MST edges back to the
      original graph maximises the largest connected component without
      making the network look like spaghetti.

This is the core of the "topological healing" described in Phase II of
the problem statement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx
import numpy as np

from skeletonize import Edge, Node


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class HealConfig:
    d_max: int = 35           # px, max Euclidean gap to bridge
    ang_max: float = 35.0     # degrees, max angular mismatch per endpoint
    corridor: int = 4         # px, half-width of the "no other endpoint" check
    weight: str = "distance"  # MST weight key


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tangent_directions(node: Node, edges: List[Edge]) -> List[Tuple[float, float]]:
    """Return BOTH forward and backward tangent directions of every edge
    incident to *node*.

    A *dead-end* endpoint only has one edge; for it the bridge direction is
    the *opposite* of the outgoing direction.  Using both signs of every
    incident edge is what makes the angular gate accept natural extensions
    of dead-ends.  Junctions and bends automatically get all their incident
    directions in this set.
    """
    out: List[Tuple[float, float]] = []
    for e in edges:
        if e.a == node.id:
            first = e.pixels[1] if len(e.pixels) > 1 else e.pixels[0]
        elif e.b == node.id:
            first = e.pixels[-2] if len(e.pixels) > 1 else e.pixels[-1]
        else:
            continue
        dy = first[0] - node.y
        dx = first[1] - node.x
        n = math.hypot(dy, dx) + 1e-9
        d = (dy / n, dx / n)
        out.append(d)
        out.append((-d[0], -d[1]))
    return out


def _angle(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Unsigned angle (degrees) between two vectors in [0, 180]."""
    a1 = math.atan2(v1[0], v1[1])
    a2 = math.atan2(v2[0], v2[1])
    d = abs(a1 - a2)
    if d > math.pi:
        d = 2 * math.pi - d
    return math.degrees(d)


# --------------------------------------------------------------------------- #
# Build candidate graph
# --------------------------------------------------------------------------- #
def _candidate_edges(nodes: List[Node], edges: List[Edge],
                     cfg: HealConfig
                     ) -> List[Tuple[int, int, float]]:
    """Return (u, v, distance) for every endpoint pair that passes the gate."""
    endpoints = [n for n in nodes if n.kind == "endpoint"]
    tangents = {n.id: _tangent_directions(n, edges) for n in nodes}
    cands: List[Tuple[int, int, float]] = []

    for i, u in enumerate(endpoints):
        for v in endpoints[i + 1:]:
            dy = v.y - u.y
            dx = v.x - u.x
            d = math.hypot(dy, dx)
            if d > cfg.d_max or d < 2.0:
                continue
            du = (dy / d, dx / d)
            dv = (-du[0], -du[1])

            # Angular gate — bridge direction at u must align with one of
            # u's tangent directions (forward OR backward).  Using the
            # *tangent* set (both signs) lets dead-end endpoints accept
            # bridges that simply extend the existing edge.
            t_u = tangents.get(u.id, [])
            t_v = tangents.get(v.id, [])
            ok_u = any(_angle(du, w) < cfg.ang_max for w in t_u) \
                if t_u else True
            ok_v = any(_angle(dv, w) < cfg.ang_max for w in t_v) \
                if t_v else True
            if not (ok_u and ok_v):
                continue

            # Straightness / corridor gate — no other endpoint in the tube
            ax, ay = u.x, u.y
            bx, by = v.x, v.y
            bad = False
            for n3 in nodes:
                if n3.id in (u.id, v.id) or n3.kind != "endpoint":
                    continue
                px, py = n3.x, n3.y
                t = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / (d * d + 1e-9)
                t = max(0.0, min(1.0, t))
                cx = ax + t * (bx - ax)
                cy = ay + t * (by - ay)
                if math.hypot(px - cx, py - cy) < cfg.corridor:
                    bad = True
                    break
            if bad:
                continue
            cands.append((u.id, v.id, d))
    return cands


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def heal_graph(nodes: List[Node], edges: List[Edge],
               cfg: HealConfig | None = None
               ) -> Tuple[List[Node], List[Edge], dict]:
    """Add bridge edges between endpoints using a gated MST.

    Returns
    -------
    nodes   : unchanged
    edges   : original edges + the new bridge edges (with geom_length = d)
    info    : dict with stats
    """
    if cfg is None:
        cfg = HealConfig()
    if not nodes:
        return nodes, edges, {"added": 0, "lcc_before": 0, "lcc_after": 0}

    # Build the working graph (NetworkX) to measure connectivity before / after
    G = nx.Graph()
    for n in nodes:
        G.add_node(n.id, y=n.y, x=n.x, kind=n.kind)
    for e in edges:
        G.add_edge(e.a, e.b, weight=e.geom_length, length=e.geom_length,
                   pixels=e.pixels, is_bridge=False)
    lcc_before = (
        len(max(nx.connected_components(G), key=len))
        if G.number_of_edges() else 1
    )

    cands = _candidate_edges(nodes, edges, cfg)
    if not cands:
        return nodes, edges, {"added": 0, "candidates": 0,
                              "lcc_before": lcc_before,
                              "lcc_after": lcc_before}

    cand_G = nx.Graph()
    for u, v, w in cands:
        cand_G.add_edge(u, v, weight=w, distance=w)
    mst = nx.minimum_spanning_tree(cand_G, weight="weight")

    added = 0
    for u, v in mst.edges():
        d = mst[u][v]["weight"]
        chain = [(nodes[u].y, nodes[u].x),
                 (nodes[v].y, nodes[v].x)]
        edges.append(Edge(a=u, b=v, pixels=chain, geom_length=float(d)))
        G.add_edge(u, v, weight=d, length=d, pixels=chain, is_bridge=True)
        added += 1

    lcc_after = (
        len(max(nx.connected_components(G), key=len))
        if G.number_of_edges() else 1
    )
    info = {
        "added": added,
        "candidates": len(cands),
        "lcc_before": lcc_before,
        "lcc_after": lcc_after,
        "lcc_pct_increase": round(
            100.0 * (lcc_after - lcc_before) / max(1, lcc_before), 2
        ),
    }
    return nodes, edges, info


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from skeletonize import extract_skeleton

    # Two 4-px-thick road segments separated by a 20-px gap.
    m = np.zeros((100, 200), np.uint8)
    m[48:52, 10:90] = 255
    m[48:52, 110:190] = 255
    skel, nodes, edges = extract_skeleton(m)
    print(f"before heal: nodes={len(nodes)} edges={len(edges)}")
    nodes, edges, info = heal_graph(
        nodes, edges, HealConfig(d_max=50, ang_max=40, corridor=4)
    )
    print(f"after  heal: nodes={len(nodes)} edges={len(edges)}  info={info}")
