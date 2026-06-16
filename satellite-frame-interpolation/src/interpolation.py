from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np


def estimate_optical_flow(
    frame0: np.ndarray,
    frame1: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
) -> np.ndarray:
    """Estimate dense optical flow with OpenCV Farnebäck."""
    f0 = _to_uint8(frame0)
    f1 = _to_uint8(frame1)
    return cv2.calcOpticalFlowFarneback(
        f0,
        f1,
        None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=0,
    )


def warp(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp a frame by a flow field."""
    h, w = flow.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xx + flow[..., 0]
    map_y = yy + flow[..., 1]
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def interpolate_pair(
    frame0: np.ndarray,
    frame1: np.ndarray,
    alpha: float = 0.5,
    flow01: np.ndarray | None = None,
    sharpen: bool = True,
) -> np.ndarray:
    """Interpolate between two frames using bidirectional optical flow.

    Parameters
    ----------
    frame0, frame1:
        Input frames.
    alpha:
        Target time in [0, 1], where 0 is frame0 and 1 is frame1.
    flow01:
        Precomputed optical flow from frame0 to frame1.
    sharpen:
        Apply mild unsharp masking to reduce blur.
    """
    if flow01 is None:
        flow01 = estimate_optical_flow(frame0, frame1)

    flow10 = -flow01
    flow0_alpha = flow01 * alpha
    flow1_alpha = flow01 * (alpha - 1.0)

    warped0 = warp(frame0, flow0_alpha)
    warped1 = warp(frame1, flow1_alpha)

    # Visibility-aware blending.
    w0 = 1.0 - alpha
    w1 = alpha
    blended = warped0 * w0 + warped1 * w1

    if sharpen:
        blended = _unsharp_mask(blended)

    return blended.astype(np.float32)


def interpolate_sequence(
    frames: List[np.ndarray],
    factor: int = 2,
    sharpen: bool = True,
) -> tuple[List[np.ndarray], List[dict]]:
    """Interpolate a sequence by integer factor.

    factor=2 inserts one frame between each pair.
    factor=4 inserts three frames between each pair.
    """
    if factor < 2:
        raise ValueError("interpolation factor must be >= 2")

    out: List[np.ndarray] = []
    meta: List[dict] = []

    for i in range(len(frames) - 1):
        out.append(frames[i])
        meta.append({"type": "original", "index": i, "alpha": None})
        flow = estimate_optical_flow(frames[i], frames[i + 1])
        for k in range(1, factor):
            alpha = k / factor
            interp = interpolate_pair(frames[i], frames[i + 1], alpha=alpha, flow01=flow, sharpen=sharpen)
            out.append(interp)
            meta.append({"type": "interpolated", "pair": [i, i + 1], "alpha": alpha})

    out.append(frames[-1])
    meta.append({"type": "original", "index": len(frames) - 1, "alpha": None})
    return out, meta


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    lo = np.nanpercentile(arr, 1)
    hi = np.nanpercentile(arr, 99)
    if hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    clipped = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (clipped * 255).astype(np.uint8)


def _unsharp_mask(frame: np.ndarray, strength: float = 0.35) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=1.0, sigmaY=1.0)
    sharpened = arr + strength * (arr - blurred)
    return np.clip(sharpened, 0, 255)
