"""
skeletonize.py
--------------
Phase II — step 1.

Convert the binary road mask into a 1-pixel-wide skeleton and extract
the topologically important primitives: nodes (intersections, endpoints)
and edges (sequences of pixels connecting two nodes).

This is the foundation of the routable graph that Phase III analyses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    """A single topological node in the road graph.

    `kind` is one of:
        - "endpoint"   : a dead-end (1 neighbour in the skeleton)
        - "junction"   : T / + intersection (≥3 neighbours)
        - "bend"       : minor angle change (2 neighbours but not colinear)
    """
    id: int
    y: int
    x: int
    kind: str = "endpoint"

    def as_tuple(self) -> Tuple[int, int]:
        return (self.y, self.x)


@dataclass
class Edge:
    """An edge stores the *pixel chain* connecting two nodes; we keep the
    chain so the dashboard can colour it and the analyser can compute
    its real geometric length (not just Euclidean)."""
    a: int
    b: int
    pixels: List[Tuple[int, int]] = field(default_factory=list)
    geom_length: float = 0.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _neighbours(skel: np.ndarray, y: int, x: int) -> List[Tuple[int, int]]:
    """8-neighbour lookup, kept in-bounds."""
    h, w = skel.shape
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                out.append((ny, nx))
    return out


def _classify(skel: np.ndarray, y: int, x: int) -> str:
    n = len(_neighbours(skel, y, x))
    if n == 0:
        return "isolated"
    if n == 1:
        return "endpoint"
    if n >= 3:
        return "junction"
    return "bend"


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def extract_skeleton(mask: np.ndarray,
                     min_edge_pixels: int = 8
                     ) -> Tuple[np.ndarray, List[Node], List[Edge]]:
    """Thin a binary road mask, then walk the skeleton to recover nodes
    and edges.

    Returns
    -------
    skel     : (H, W) 0/1 ndarray, 1-pixel-wide road skeleton
    nodes    : list of Node
    edges    : list of Edge
    """
    mask_bin = (mask > 127).astype(np.uint8)
    if mask_bin.sum() == 0:
        return np.zeros_like(mask_bin, dtype=np.uint8), [], []

    skel = skeletonize(mask_bin).astype(np.uint8)

    # ----- 1.  Find node candidates -----
    candidate_pixels: Dict[Tuple[int, int], str] = {}
    ys, xs = np.where(skel > 0)
    for y, x in zip(ys, xs):
        k = _classify(skel, y, x)
        if k in ("endpoint", "junction"):
            candidate_pixels[(int(y), int(x))] = k

    # Bends: keep them as minor nodes (helps routing accuracy)
    for (y, x), k in list(candidate_pixels.items()):
        if k == "endpoint":
            pass  # always kept
        elif k == "junction":
            pass  # always kept
    # Optionally also add bends as nodes:
    for y, x in zip(ys, xs):
        if (y, x) in candidate_pixels:
            continue
        k = _classify(skel, y, x)
        if k == "bend":
            nbrs = _neighbours(skel, y, x)
            # Only add bends where the two neighbours are roughly opposite
            # (i.e. we are not at a perfectly straight line).  This keeps
            # the graph clean.
            (y1, x1), (y2, x2) = nbrs[0], nbrs[1]
            v1 = np.array([y1 - y, x1 - x], float)
            v2 = np.array([y2 - y, x2 - x], float)
            cosang = (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            if abs(cosang) < 0.85:
                candidate_pixels[(int(y), int(x))] = "bend"

    # Build Node objects
    nodes: List[Node] = []
    node_at: Dict[Tuple[int, int], int] = {}
    for (y, x), k in candidate_pixels.items():
        nid = len(nodes)
        nodes.append(Node(id=nid, y=y, x=x, kind=k))
        node_at[(y, x)] = nid

    # ----- 2.  Walk the skeleton to recover edges -----
    # We repeatedly pick an unused node, walk along the skeleton, and
    # stop when we hit another node (or run out of pixels).
    visited_edge_pix: set[Tuple[int, int]] = set()
    edges: List[Edge] = []

    # Start from each node, walk each unvisited neighbour
    for nid, node in enumerate(nodes):
        for ny, nx in _neighbours(skel, node.y, node.x):
            if (ny, nx) in visited_edge_pix:
                continue
            chain: List[Tuple[int, int]] = [(node.y, node.x), (ny, nx)]
            visited_edge_pix.add((node.y, node.x))
            visited_edge_pix.add((ny, nx))
            cur_y, cur_x = ny, nx
            prev_y, prev_x = node.y, node.x
            target_id = -1
            while True:
                if (cur_y, cur_x) in node_at and (cur_y, cur_x) != (node.y, node.x):
                    target_id = node_at[(cur_y, cur_x)]
                    break
                nxt = None
                for ay, ax in _neighbours(skel, cur_y, cur_x):
                    if (ay, ax) == (prev_y, prev_x):
                        continue
                    if (ay, ax) in visited_edge_pix:
                        # If we hit an already-traversed pixel, treat it
                        # as a connection to whatever node is at that pixel
                        if (ay, ax) in node_at:
                            target_id = node_at[(ay, ax)]
                            break
                        continue
                    nxt = (ay, ax)
                    break
                if nxt is None:
                    break
                chain.append(nxt)
                visited_edge_pix.add(nxt)
                prev_y, prev_x = cur_y, cur_x
                cur_y, cur_x = nxt
            if target_id == -1 or len(chain) < min_edge_pixels:
                continue
            # Compute geometric length
            length = 0.0
            for i in range(1, len(chain)):
                py, px = chain[i - 1]
                cy, cx = chain[i]
                length += float(np.hypot(cy - py, cx - px))
            edges.append(Edge(a=nid, b=target_id, pixels=chain,
                              geom_length=length))

    return skel, nodes, edges


# --------------------------------------------------------------------------- #
# Sanity plot (used by pipeline when --plot is on)
# --------------------------------------------------------------------------- #
def render_overlay(skel: np.ndarray, nodes: List[Node],
                   edges: List[Edge], size: int = 1024) -> np.ndarray:
    """Return a BGR image with the skeleton + nodes + edges drawn on a
    white background.  Used for the static summary panel of the dashboard."""
    h, w = skel.shape
    scale = max(1, size // max(h, w))
    img = np.full((h * scale, w * scale, 3), 255, dtype=np.uint8)
    skel_big = cv2.resize(skel, (w * scale, h * scale),
                          interpolation=cv2.INTER_NEAREST)
    img[skel_big > 0] = (40, 40, 40)
    for e in edges:
        pts = np.array([(p[1] * scale, p[0] * scale) for p in e.pixels],
                       dtype=np.int32)
        cv2.polylines(img, [pts], False, (180, 100, 30), 1, cv2.LINE_AA)
    for n in nodes:
        col = (0, 0, 255) if n.kind == "endpoint" else \
              (0, 200, 0) if n.kind == "junction" else (200, 200, 0)
        cv2.circle(img, (n.x * scale, n.y * scale),
                   max(3, scale), col, -1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Build a tiny mask: a cross (uint8 0/255, what extract_skeleton expects)
    m = np.zeros((80, 80), np.uint8)
    m[10:70, 38:42] = 255
    m[38:42, 10:70] = 255
    skel, nodes, edges = extract_skeleton(m)
    print(f"skel px={skel.sum()}  nodes={len(nodes)}  edges={len(edges)}")
    for n in nodes[:5]:
        print(f"  node {n.id} {n.kind} @ ({n.y},{n.x})")
    for e in edges[:5]:
        print(f"  edge {e.a}->{e.b}  len={e.geom_length:.1f}  px={len(e.pixels)}")
