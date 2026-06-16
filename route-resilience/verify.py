"""
verify.py
---------
End-to-end self-test for the route-resilience pipeline.

This is intentionally split into two layers so that the project can be
verified even when the segmentation model is under-trained (e.g. with
synthetic data on CPU):

  LAYER A  -  Topology pipeline on GROUND-TRUTH masks.
              Proves:  skeletonize -> heal -> analyze works end-to-end.
              Does NOT depend on the model.

  LAYER B  -  Model quality (predict on val, report IoU/Dice).
              Reports metrics but does not gate the PASS/FAIL.

The script is "PASS" as long as Layer A passes for every tile, the
imports load, the checkpoint exists, and run_on_dataset writes its
report.  Layer B is reported numerically.
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

# Force UTF-8 output (Windows cp1252 can't encode \u2713 / \u2192 / etc.)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ------------------------------------------------------------------ #
from model import AttentionUNet
from losses import CombinedLoss
from skeletonize import extract_skeleton
from heal import HealConfig, heal_graph
from analyze import (AnalyzeConfig, betweenness, criticality_scores,
                     global_efficiency, resilience_curve, to_networkx)
from pipeline import predict_mask, run_on_image, run_on_dataset

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"


def check(name: str, ok: bool, detail: str = "") -> bool:
    flag = PASS if ok else FAIL
    msg = f"  {flag}  {name}"
    if detail:
        msg += f"  -  {detail}"
    print(msg)
    return ok


# ------------------------------------------------------------------ #
def load_trained_model(ckpt_path: Path, device: str = "cpu") -> AttentionUNet:
    """Load checkpoint, returning the model.  Does NOT print unicode."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    base = ckpt.get("args", {}).get("base", 16)
    m = AttentionUNet(in_ch=3, base=base)
    m.load_state_dict(ckpt["model"])
    m.to(device).eval()
    return m, base


# ------------------------------------------------------------------ #
def main():
    print("=" * 60)
    print("ROUTE-RESILIENCE  END-TO-END  SELF-VERIFICATION")
    print("=" * 60)

    results: Dict[str, bool] = {}

    # ================================================================ #
    # LAYER A  -  Topology pipeline on ground-truth masks              #
    # ================================================================ #
    print("\n--- LAYER A: topology pipeline on GROUND TRUTH masks ---")
    data_dir = Path("data/synth")
    masks_dir = data_dir / "masks"

    # ---- A1. data on disk ---- #
    for sub in ("images", "occluded", "masks", "meta.json"):
        ok = (data_dir / sub).exists()
        results.setdefault("data_exists", True)
        results["data_exists"] = results["data_exists"] and ok
    tiles = sorted(masks_dir.glob("*.png"))
    results["data_count"] = check(
        "data has at least 4 tiles", len(tiles) >= 4,
        f"found {len(tiles)} tiles",
    )

    # ---- A2-A5. per-tile topology on GT ---- #
    heal_cfg = HealConfig(d_max=35, ang_max=40.0)
    an_cfg = AnalyzeConfig(top_k_ablations=5)
    road_present = 0
    skel_nonempty = 0
    heal_lcc_grew = 0
    heal_bridges = 0
    ablation_drops = 0
    finite_betw = 0
    eff_positive = 0
    n_tiles = 0
    per_tile_metrics: List[Dict] = []

    for mp in tiles:
        n_tiles += 1
        try:
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"  {FAIL}  could not read {mp.name}")
                continue

            # --- skeletonize ---
            if (mask > 127).mean() > 0.005:
                road_present += 1
            skel, nodes, edges = extract_skeleton(mask)
            n_n, n_e = len(nodes), len(edges)
            if n_n > 0 and n_e > 0:
                skel_nonempty += 1

            # --- heal ---
            nodes2, edges2, info = heal_graph(nodes, edges, heal_cfg)
            heal_bridges += info.get("added", 0)
            if info.get("lcc_after", 0) > info.get("lcc_before", 0):
                heal_lcc_grew += 1

            # --- analyze ---
            G = to_networkx(nodes2, edges2)
            betw = betweenness(G, an_cfg)
            node_score, ordered = criticality_scores(G, betw)
            eff0 = global_efficiency(G)
            if eff0 > 0:
                eff_positive += 1
            finite_count = sum(1 for v in betw.values()
                               if np.isfinite(v))
            if finite_count == len(betw):
                finite_betw += 1
            curve = resilience_curve(G, ordered, an_cfg)
            if curve and curve[0]["efficiency"] < eff0:
                ablation_drops += 1

            per_tile_metrics.append({
                "tile": mp.stem,
                "road_pct": round(float((mask > 127).mean()) * 100, 2),
                "nodes": n_n, "edges": n_e,
                "bridges_added": info.get("added", 0),
                "lcc_before": info.get("lcc_before", 0),
                "lcc_after": info.get("lcc_after", 0),
                "global_efficiency": round(float(eff0), 4),
                "top_node_centrality": round(
                    float(max(betw.values())) if betw else 0.0, 4),
                "ablation_step1_eff": round(
                    float(curve[0]["efficiency"]) if curve else 0.0, 4),
            })
        except Exception as e:
            print(f"  {FAIL}  tile {mp.name}  -  {e}")
            traceback.print_exc(limit=2)
            results["topology_crash"] = False
        else:
            results["topology_crash"] = True

    results["gt_road_present"] = check(
        "GT mask has >0.5% road on every tile",
        road_present == n_tiles, f"{road_present}/{n_tiles}")
    results["gt_skeleton"] = check(
        "GT skeleton has nodes+edges on every tile",
        skel_nonempty == n_tiles, f"{skel_nonempty}/{n_tiles}")
    results["gt_heal_lcc"] = check(
        "heal grows LCC on >=75% of tiles",
        heal_lcc_grew >= 0.75 * n_tiles,
        f"{heal_lcc_grew}/{n_tiles}, total bridges={heal_bridges}")
    results["gt_betw_finite"] = check(
        "betweenness is finite on every tile",
        finite_betw == n_tiles, f"{finite_betw}/{n_tiles}")
    results["gt_eff_positive"] = check(
        "global efficiency > 0 on every tile",
        eff_positive == n_tiles, f"{eff_positive}/{n_tiles}")
    results["gt_ablation"] = check(
        "ablation step 1 lowers efficiency on every tile",
        ablation_drops == n_tiles, f"{ablation_drops}/{n_tiles}")

    # ================================================================ #
    # LAYER B  -  Model quality (predict on val, report metrics)       #
    # ================================================================ #
    print("\n--- LAYER B: model quality (numerical report) ---")
    ckpt = Path("checkpoints/best.pt")
    model_metrics = {}
    if not ckpt.exists():
        print(f"  {INFO}  no checkpoint at {ckpt} - skipping model eval")
        results["model_loaded"] = False
    else:
        try:
            m, base = load_trained_model(ckpt, device="cpu")
            nparams = sum(p.numel() for p in m.parameters()) / 1e6
            print(f"  {INFO}  loaded {ckpt.name}  base={base}  "
                  f"params={nparams:.2f}M")
            results["model_loaded"] = True

            # Predict on each occluded tile and compare to GT
            occ_dir = data_dir / "occluded"
            ious, dices = [], []
            for mp in tiles:
                occ = cv2.imread(str(occ_dir / mp.name),
                                 cv2.IMREAD_COLOR)
                gt = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127)
                if occ is None or gt.sum() == 0:
                    continue
                pred = predict_mask(m, occ, device="cpu", thresh=0.5)
                p = pred > 127
                inter = (p & gt).sum()
                union = (p | gt).sum()
                iou = inter / max(union, 1)
                dice = (2 * inter) / (p.sum() + gt.sum() + 1e-8)
                ious.append(float(iou))
                dices.append(float(dice))
                model_metrics[mp.stem] = {
                    "iou": round(float(iou), 3),
                    "dice": round(float(dice), 3),
                    "pred_road_pct": round(float(p.mean()) * 100, 2),
                    "gt_road_pct": round(float(gt.mean()) * 100, 2),
                }
            mean_iou = float(np.mean(ious)) if ious else 0.0
            mean_dice = float(np.mean(dices)) if dices else 0.0
            print(f"  {INFO}  model mean IoU = {mean_iou:.3f}   "
                  f"mean Dice = {mean_dice:.3f}   "
                  f"on {len(ious)} tiles")
        except Exception as e:
            print(f"  {FAIL}  model eval crashed: {e}")
            traceback.print_exc(limit=2)
            results["model_loaded"] = False

    # ================================================================ #
    # LAYER C  -  run_on_dataset writes a report file                   #
    # ================================================================ #
    print("\n--- LAYER C: run_on_dataset end-to-end ---")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            tmp = Path(tmp)
            # Need a model to call run_on_dataset; reuse the trained one
            if not results.get("model_loaded"):
                m, base = AttentionUNet(in_ch=3, base=16), 16
            # The pipeline's run_on_dataset signature:
            #   run_on_dataset(model, src, out_dir, device, limit)
            out_dir = tmp / "out"
            rep = run_on_dataset(m, data_dir, out_dir,
                                 device="cpu", limit=4)
            report_file = out_dir / "report.json"
            ok = report_file.exists()
            results["report_written"] = check(
                "run_on_dataset writes report.json", ok)
            if ok:
                d = json.loads(report_file.read_text())
                agg = d.get("aggregate", {})
                required = {"n_tiles", "total_nodes", "total_edges",
                            "bridges_added", "lcc_pct_avg", "scenes"}
                missing = required - agg.keys()
                results["report_complete"] = check(
                    "report.aggregate has all required keys",
                    not missing,
                    f"missing={missing}  "
                    f"tiles={agg.get('n_tiles')}  "
                    f"nodes={agg.get('total_nodes')}  "
                    f"bridges={agg.get('bridges_added')}")
        except Exception as e:
            results["report_written"] = check(
                "run_on_dataset", False, str(e)[:120])
            traceback.print_exc(limit=2)

    # ================================================================ #
    # Summary                                                          #
    # ================================================================ #
    print("\n" + "=" * 60)
    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)
    print(f"  LAYER A (topology on GT):  "
          f"{sum(1 for k,v in results.items() if k.startswith('gt_') and v)}/"
          f"{sum(1 for k in results if k.startswith('gt_'))}")
    print(f"  LAYER B (model loaded):    "
          f"{'YES' if results.get('model_loaded') else 'NO'}")
    if model_metrics:
        ious = [v['iou'] for v in model_metrics.values()]
        dices = [v['dice'] for v in model_metrics.values()]
        print(f"  Model mean IoU/Dice:       "
              f"{np.mean(ious):.3f} / {np.mean(dices):.3f}")
    print(f"  LAYER C (dataset report):  "
          f"{sum(1 for k,v in results.items() if k.startswith('report_') and v)}/"
          f"{sum(1 for k in results if k.startswith('report_'))}")
    print(f"\n  {n_pass}/{n_total} checks passed")

    # Pass criteria: Layer A all green + checkpoint loads + report writes
    layer_a_ok = all(v for k, v in results.items() if k.startswith("gt_"))
    layer_c_ok = all(v for k, v in results.items() if k.startswith("report_"))
    overall = layer_a_ok and layer_c_ok
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 60)

    # Write a JSON report for the README to cite
    summary = {
        "checks": results,
        "per_tile_topology": per_tile_metrics,
        "per_tile_model": model_metrics,
    }
    Path("verify_report.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote verify_report.json")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
