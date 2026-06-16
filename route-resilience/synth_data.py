"""
synth_data.py
-------------
Generates synthetic satellite-like tiles for end-to-end testing of the
Route Resilience pipeline WITHOUT requiring a network download of
Sentinel-2 / Cartosat imagery.

Each tile simulates a small urban / suburban / rural patch:
  - Background texture (perlin-like noise) emulating terrain reflectance
  - Ground-truth road network drawn as polylines of varying width
  - One of three "terrains": urban, forested, rural
  - Optional occlusions: tree canopy blobs, building shadows, vehicle specks,
    cloud streaks

Outputs (saved under data/synth/):
  images/<terrain>_<idx>.png       — RGB tile (H=W=512)
  masks/<terrain>_<idx>.png        — binary road mask (255=road)
  occluded/<terrain>_<idx>.png     — RGB tile with simulated occlusions
  meta.json                        — list of files + parameters
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _perlin_like(h: int, w: int, scale: float = 24.0, seed: int = 0) -> np.ndarray:
    """Cheap stand-in for Perlin noise: low-pass white noise via repeated
    blur + downsample stacking. Good enough for a 'terrain texture'."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    cur_scale = scale
    while cur_scale > 1.5:
        small = rng.random((max(2, int(h / cur_scale)), max(2, int(w / cur_scale))))
        big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        out += big * amp
        amp *= 0.55
        cur_scale /= 2.0
    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
    return out


def _draw_road(
    canvas: np.ndarray,
    polyline: List[Tuple[int, int]],
    width: int,
    color: int = 255,
) -> np.ndarray:
    """Rasterise a polyline onto a uint8 single-channel canvas with a fixed
    line width.  Self-intersections are handled by simple Bresenham +
    disk dilation along the path."""
    mask = np.zeros_like(canvas)
    for i in range(len(polyline) - 1):
        cv2.line(mask, polyline[i], polyline[i + 1], color, thickness=width,
                 lineType=cv2.LINE_AA)
    # Round the line ends with small circles
    for (x, y) in polyline:
        cv2.circle(mask, (int(x), int(y)), width // 2, color, -1,
                   lineType=cv2.LINE_AA)
    return np.maximum(canvas, mask)


def _random_road_graph(
    h: int, w: int, n_nodes: int, rng: random.Random
) -> List[Tuple[int, int]]:
    """Build a connected-ish random road polyline by walking between random
    waypoints with mild smoothing.  Returns a list of pixel coords."""
    pts: List[Tuple[int, int]] = []
    x = rng.randint(int(w * 0.1), int(w * 0.9))
    y = rng.randint(int(h * 0.1), int(h * 0.9))
    pts.append((x, y))
    n_segments = rng.randint(3, 6)
    for _ in range(n_segments):
        tx = rng.randint(int(w * 0.05), int(w * 0.95))
        ty = rng.randint(int(h * 0.05), int(h * 0.95))
        steps = rng.randint(20, 60)
        for s in range(1, steps + 1):
            t = s / steps
            # Curved interpolation for non-trivial shapes
            cx = (1 - t) * x + t * tx + rng.randint(-12, 12)
            cy = (1 - t) * y + t * ty + rng.randint(-12, 12)
            cx = int(np.clip(cx, 0, w - 1))
            cy = int(np.clip(cy, 0, h - 1))
            pts.append((cx, cy))
        x, y = tx, ty
    return pts


# --------------------------------------------------------------------------- #
# Occlusion generators
# --------------------------------------------------------------------------- #
def _occlusion_canopy(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Tree-canopy blobs: dark green ellipses with bumpy edges."""
    out = img.copy()
    h, w, _ = out.shape
    n = rng.randint(15, 35)
    for _ in range(n):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        rx, ry = rng.randint(10, 28), rng.randint(10, 28)
        ang = rng.randint(0, 180)
        # base color: dark forest green
        col = (rng.randint(10, 50), rng.randint(60, 110), rng.randint(20, 60))
        cv2.ellipse(out, (cx, cy), (rx, ry), ang, 0, 360, col, -1,
                    lineType=cv2.LINE_AA)
    # blur the canopy edges
    out = cv2.GaussianBlur(out, (5, 5), 0)
    return out


def _occlusion_shadow(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Long building shadows: dark translucent parallelograms."""
    out = img.copy()
    h, w, _ = out.shape
    n = rng.randint(2, 5)
    for _ in range(n):
        x1 = rng.randint(0, w)
        y1 = rng.randint(0, h)
        length = rng.randint(80, 220)
        width = rng.randint(18, 35)
        ang = rng.randint(20, 70)  # sun angle
        x2 = int(x1 + length * np.cos(np.deg2rad(ang)))
        y2 = int(y1 + length * np.sin(np.deg2rad(ang)))
        # build the shadow as a polygon
        perp = np.deg2rad(ang + 90)
        p1 = (x1 + int(width * np.cos(perp)),
              y1 + int(width * np.sin(perp)))
        p2 = (x1 - int(width * np.cos(perp)),
              y1 - int(width * np.sin(perp)))
        p3 = (x2 - int(width * np.cos(perp)),
              y2 - int(width * np.sin(perp)))
        p4 = (x2 + int(width * np.cos(perp)),
              y2 + int(width * np.sin(perp)))
        overlay = out.copy()
        cv2.fillPoly(overlay, [np.array([p1, p2, p3, p4])], (20, 20, 20))
        out = cv2.addWeighted(overlay, 0.65, out, 0.35, 0)
    return out


def _occlusion_vehicles(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Tiny bright rectangles scattered on the road."""
    out = img.copy()
    h, w, _ = out.shape
    n = rng.randint(8, 25)
    for _ in range(n):
        x = rng.randint(0, w - 8)
        y = rng.randint(0, h - 8)
        col = (rng.randint(180, 255),) * 3
        cv2.rectangle(out, (x, y), (x + rng.randint(5, 9),
                                     y + rng.randint(3, 6)), col, -1)
    return out


def _occlusion_cloud(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Soft white cloud streaks."""
    out = img.copy()
    h, w, _ = out.shape
    n = rng.randint(1, 3)
    for _ in range(n):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        rx, ry = rng.randint(60, 140), rng.randint(20, 45)
        ang = rng.randint(0, 180)
        overlay = out.copy()
        cv2.ellipse(overlay, (cx, cy), (rx, ry), ang, 0, 360,
                    (240, 240, 245), -1, lineType=cv2.LINE_AA)
        overlay = cv2.GaussianBlur(overlay, (15, 15), 0)
        out = cv2.addWeighted(overlay, 0.7, out, 0.3, 0)
    return out


# --------------------------------------------------------------------------- #
# Tile generator
# --------------------------------------------------------------------------- #
@dataclass
class TileGen:
    h: int = 512
    w: int = 512
    terrains: Tuple[str, ...] = ("urban", "forested", "rural")

    def make_tile(self, terrain: str, seed: int
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (clean_rgb, occluded_rgb, mask)."""
        rng = random.Random(seed)
        nrng = np.random.default_rng(seed)

        # Terrain base reflectance (RGB)
        bg = _perlin_like(self.h, self.w, scale=32.0, seed=seed)
        if terrain == "urban":
            base = np.stack([
                80 + 60 * bg,
                80 + 60 * bg,
                80 + 60 * bg,
            ], axis=-1).astype(np.uint8)
            # sprinkle building blocks
            for _ in range(rng.randint(20, 40)):
                x = rng.randint(0, self.w - 30)
                y = rng.randint(0, self.h - 30)
                bw = rng.randint(15, 35)
                bh = rng.randint(15, 35)
                col = (rng.randint(120, 200),) * 3
                cv2.rectangle(base, (x, y), (x + bw, y + bh), col, -1)
            road_w = rng.randint(9, 13)
        elif terrain == "forested":
            # mostly green with texture
            base = np.stack([
                30 + 50 * bg,
                80 + 70 * bg,
                30 + 50 * bg,
            ], axis=-1).astype(np.uint8)
            # tree speckles
            for _ in range(rng.randint(80, 140)):
                x = rng.randint(0, self.w)
                y = rng.randint(0, self.h)
                r = rng.randint(2, 5)
                col = (rng.randint(20, 60), rng.randint(90, 150), rng.randint(20, 60))
                cv2.circle(base, (x, y), r, col, -1)
            road_w = rng.randint(7, 10)
        else:  # rural
            base = np.stack([
                100 + 80 * bg,
                110 + 70 * bg,
                60 + 50 * bg,
            ], axis=-1).astype(np.uint8)
            for _ in range(rng.randint(10, 20)):
                x = rng.randint(0, self.w)
                y = rng.randint(0, self.h)
                r = rng.randint(4, 9)
                col = (rng.randint(60, 110), rng.randint(120, 160), rng.randint(40, 80))
                cv2.circle(base, (x, y), r, col, -1)
            road_w = rng.randint(6, 9)

        # Ground truth road mask
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        n_roads = rng.randint(2, 4)
        for _ in range(n_roads):
            poly = _random_road_graph(self.h, self.w,
                                      n_nodes=rng.randint(4, 8), rng=rng)
            mask = _draw_road(mask, poly, width=road_w, color=255)

        # Add the road colour onto the base image
        rgb = base.copy()
        road_pix = mask > 0
        rgb[road_pix] = (
            np.clip(rgb[road_pix].astype(np.int32)
                    + np.array([40, 40, 40], dtype=np.int32), 0, 255)
        ).astype(np.uint8)
        # Smooth the road to look realistic
        rgb = cv2.GaussianBlur(rgb, (3, 3), 0)

        # Apply occlusions
        occluded = rgb.copy()
        if terrain == "forested":
            occluded = _occlusion_canopy(occluded, rng)
        if terrain == "urban":
            occluded = _occlusion_shadow(occluded, rng)
        occluded = _occlusion_vehicles(occluded, rng)
        occluded = _occlusion_cloud(occluded, rng)

        return rgb, occluded, mask


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/synth")
    ap.add_argument("--n_per_terrain", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = Path(args.out)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "masks").mkdir(parents=True, exist_ok=True)
    (out_root / "occluded").mkdir(parents=True, exist_ok=True)

    gen = TileGen()
    meta: List[dict] = []
    idx = 0
    for terrain in gen.terrains:
        for i in range(args.n_per_terrain):
            seed = args.seed + idx
            clean, occluded, mask = gen.make_tile(terrain, seed=seed)
            name = f"{terrain}_{i:03d}.png"
            # Save three triplets per tile:
            #   images/    — clean RGB  (ground reference, no occlusions)
            #   occluded/  — degraded   (model input)
            #   masks/     — road mask  (training target)
            cv2.imwrite(str(out_root / "images" / name),
                        cv2.cvtColor(clean, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_root / "occluded" / name),
                        cv2.cvtColor(occluded, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_root / "masks" / name), mask)
            meta.append({"id": name[:-4], "terrain": terrain, "seed": seed})
            idx += 1
            print(f"  wrote {name}")

    with open(out_root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nGenerated {len(meta)} tiles into {out_root}")


if __name__ == "__main__":
    main()
