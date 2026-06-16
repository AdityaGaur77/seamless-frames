"""
pipeline.py
-----------
End-to-end orchestrator.

Glues together:
    dataset   → model (seg) → mask   → skeletonize → heal → analyze → JSON
                                                              ↓
                                                       dashboard (Streamlit)

CLI usage
---------
    # Train (uses data/synth by default)
    python train.py --epochs 30 --batch 4

    # Run the full pipeline on a single tile or on the whole dataset
    python pipeline.py --ckpt checkpoints/best.pt --src data/synth \
        --out reports/run.json --plot reports/preview.png

    # Or run on a single PNG image
    python pipeline.py --ckpt checkpoints/best.pt --img some_tile.png \
        --out reports/single.json
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize as sk_skeletonize

from analyze import (AnalyzeConfig, betweenness, criticality_scores,
                     edge_criticality, global_efficiency, resilience_curve,
                     to_networkx, top_gatekeepers)
from heal import HealConfig, heal_graph
from model import AttentionUNet
from skeletonize import Edge, Node, extract_skeleton, render_overlay


# --------------------------------------------------------------------------- #
def load_model(ckpt_path: str, base: int = 16, device: str = "cpu") -> AttentionUNet:
    model = AttentionUNet(in_ch=3, base=base).to(device)
    if Path(ckpt_path).exists():
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        model.eval()
        print(f"[ok] loaded {ckpt_path}")
    else:
        print(f"[warn] no checkpoint at {ckpt_path}, using random weights")
        model.eval()
    return model


def predict_mask(model: AttentionUNet, img_bgr: np.ndarray,
                 device: str = "cpu", thresh: float = 0.5) -> np.ndarray:
    """Run the model on a single BGR image and return a binary mask."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    # Resize to a multiple of 32 (encoder downsamples 4x)
    nh, nw = (h // 32) * 32, (w // 32) * 32
    if (nh, nw) != (h, w):
        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img
    t = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
    t = t.to(device)
    with torch.no_grad():
        out, _, _ = model(t)
    prob = torch.sigmoid(out)[0, 0].cpu().numpy()
    if (nh, nw) != (h, w):
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return (prob > thresh).astype(np.uint8) * 255


# --------------------------------------------------------------------------- #
def process_mask(mask: np.ndarray,
                 heal_cfg: HealConfig = None,
                 an_cfg: AnalyzeConfig = None
                 ) -> Dict:
    """Run the full Phase II + III stack on a binary mask.

    Returns a JSON-serialisable report describing the topology and the
    resilience analysis of the recovered graph.
    """
    if heal_cfg is None:
        heal_cfg = HealConfig()
    if an_cfg is None:
        an_cfg = AnalyzeConfig()

    skel, nodes, edges = extract_skeleton(mask)
    if len(nodes) == 0:
        return {
            "skel_pixels": int(skel.sum()),
            "nodes": 0, "edges": 0,
            "heal": {"added": 0},
            "centrality": {},
            "gatekeepers": [],
            "resilience": [],
            "summary": "empty mask",
        }

    nodes_h, edges_h, heal_info = heal_graph(nodes, edges, heal_cfg)
    G = to_networkx(nodes_h, edges_h)
    betw = betweenness(G, an_cfg)
    node_score, ordered = criticality_scores(G, betw)
    ec = edge_criticality(G, node_score)
    curve = resilience_curve(G, ordered, an_cfg)
    base_eff = global_efficiency(G)

    nodes_json = [
        {"id": n.id, "y": int(n.y), "x": int(n.x), "kind": n.kind,
         "centrality": float(node_score.get(n.id, 0.0))}
        for n in nodes_h
    ]
    edges_json = []
    for e in edges_h:
        key = (e.a, e.b) if (e.a, e.b) in ec else (e.b, e.a)
        edges_json.append({
            "a": e.a, "b": e.b,
            "length": float(e.geom_length),
            "criticality": float(ec.get(key, 0.0)),
            "is_bridge": bool(e.pixels and len(e.pixels) <= 2),
            "n_pixels": len(e.pixels),
        })

    return {
        "skel_pixels": int(skel.sum()),
        "nodes": len(nodes_h),
        "edges": len(edges_h),
        "heal": heal_info,
        "base_efficiency": float(base_eff),
        "centrality_summary": {
            "max": float(max(node_score.values())) if node_score else 0.0,
            "mean": float(np.mean(list(node_score.values()))) if node_score else 0.0,
            "n_hotspots": int(sum(1 for v in node_score.values() if v > 0.4)),
        },
        "gatekeepers": top_gatekeepers(ordered, nodes_h, k=10),
        "resilience": curve[:50],
        "nodes_list": nodes_json,
        "edges_list": edges_json,
        "summary": _summarise(nodes_h, edges_h, heal_info, node_score, curve, base_eff),
    }


def _summarise(nodes, edges, heal_info, node_score, curve, base_eff) -> str:
    if not nodes:
        return "Empty graph."
    top3 = sorted(node_score.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_str = ", ".join(f"#{nid} (s={s:.2f})" for nid, s in top3)
    ri = curve[0]["resilience_index"] if curve else 1.0
    return (f"{len(nodes)} nodes / {len(edges)} edges.  "
            f"Heal added {heal_info['added']} bridges "
            f"(LCC {heal_info['lcc_before']}->{heal_info['lcc_after']}, "
            f"+{heal_info['lcc_pct_increase']}%).  "
            f"Top gatekeepers: {top_str}.  "
            f"Resilience after #1 ablation: {ri:.2f}.")


# --------------------------------------------------------------------------- #
def run_on_image(model: AttentionUNet, img_path: Path, out_dir: Path,
                 device: str = "cpu", save_overlay: bool = True) -> Dict:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": f"could not read {img_path}"}
    pred = predict_mask(model, img, device=device)
    report = process_mask(pred)
    report["source"] = str(img_path)

    if save_overlay:
        skel, nodes, edges = extract_skeleton(pred)
        from heal import heal_graph
        nodes, edges, _ = heal_graph(nodes, edges)
        overlay = render_overlay(skel, nodes, edges, size=1024)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / (img_path.stem + "_graph.png")
        cv2.imwrite(str(out_png), overlay)
        report["overlay"] = str(out_png)

    return report


def run_on_dataset(model: AttentionUNet, src: Path, out_dir: Path,
                   device: str = "cpu", limit: int | None = None) -> Dict:
    src = Path(src)
    img_dir = src / "images"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(img_dir.glob("*.png"))
    if limit:
        files = files[:limit]

    all_reports = []
    aggregate = {
        "n_tiles": 0, "total_nodes": 0, "total_edges": 0,
        "bridges_added": 0, "lcc_pct_total": 0.0,
        "scenes": {},
    }
    for f in files:
        r = run_on_image(model, f, out_dir, device=device, save_overlay=False)
        all_reports.append(r)
        aggregate["n_tiles"] += 1
        aggregate["total_nodes"] += r.get("nodes", 0)
        aggregate["total_edges"] += r.get("edges", 0)
        aggregate["bridges_added"] += r.get("heal", {}).get("added", 0)
        aggregate["lcc_pct_total"] += r.get("heal", {}).get("lcc_pct_increase", 0.0)
        # scene lookup from meta (supports both 'name' and 'id' keys)
        meta_path = src / "meta.json"
        if meta_path.exists():
            with open(meta_path) as fh:
                meta_list = json.load(fh)
            meta = {}
            for m in meta_list:
                key = m.get("name") or m.get("id")
                if key:
                    meta[key] = m
            # synth_data.py writes "terrain"; legacy "scene" is also accepted
            scene = (meta.get(f.stem, {}).get("terrain")
                     or meta.get(f.stem, {}).get("scene", "unknown"))
            aggregate["scenes"].setdefault(scene, 0)
            aggregate["scenes"][scene] += 1

    aggregate["lcc_pct_avg"] = (aggregate["lcc_pct_total"] / max(1, aggregate["n_tiles"]))
    out_file = out_dir / "report.json"
    with open(out_file, "w") as f:
        json.dump({"aggregate": aggregate, "tiles": all_reports}, f, indent=2)
    print(f"[ok] wrote {out_file}")
    return {"aggregate": aggregate, "tiles": all_reports}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--img", default=None, help="single image path")
    ap.add_argument("--src", default=None, help="dataset root (with images/)")
    ap.add_argument("--out", default="reports/run.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out).parent
    model = load_model(args.ckpt, base=args.base, device=args.device)

    if args.img:
        r = run_on_image(model, Path(args.img), out_dir, device=args.device)
        with open(args.out, "w") as f:
            json.dump(r, f, indent=2)
        print(json.dumps({k: v for k, v in r.items()
                          if k not in ("nodes_list", "edges_list")}, indent=2))
    elif args.src:
        run_on_dataset(model, Path(args.src), out_dir, device=args.device,
                       limit=args.limit)
    else:
        print("Provide either --img or --src")


if __name__ == "__main__":
    main()
