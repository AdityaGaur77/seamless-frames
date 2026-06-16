from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def _prepare_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} != {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        raise ValueError("No finite overlapping pixels for metric calculation.")
    return a[mask], b[mask]


def mse(pred: np.ndarray, truth: np.ndarray) -> float:
    p, t = _prepare_pair(pred, truth)
    return float(mean_squared_error(t, p))


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, truth)))


def psnr(pred: np.ndarray, truth: np.ndarray, data_range: float = 255.0) -> float:
    p, t = _prepare_pair(pred, truth)
    if float(np.max(t) - np.min(t)) <= 1e-6:
        return float("inf") if float(np.max(p) - np.min(p)) <= 1e-6 else 0.0
    return float(peak_signal_noise_ratio(t, p, data_range=data_range))


def ssim(pred: np.ndarray, truth: np.ndarray, data_range: float = 255.0) -> float:
    p, t = _prepare_pair(pred, truth)
    if p.size < 49:
        win_size = 3
    else:
        win_size = 11 if p.shape[0] >= 11 and p.shape[1] >= 11 else min(p.shape)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        return float("nan")
    try:
        return float(structural_similarity(t, p, data_range=data_range, win_size=win_size))
    except ValueError:
        return float("nan")


def _gradients(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(frame)
    return gx, gy


def _gradient_magnitude(frame: np.ndarray) -> np.ndarray:
    gx, gy = _gradients(frame)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def fsim_lite(pred: np.ndarray, truth: np.ndarray) -> float:
    """Lightweight FSIM-style metric for cloud-edge preservation.

    This is not a full phase-congruency implementation, but it captures the same
    intuition: compare local phase/edge strength and gradient magnitude.
    """
    p = np.asarray(pred, dtype=np.float32)
    t = np.asarray(truth, dtype=np.float32)
    if p.shape != t.shape:
        raise ValueError(f"Shape mismatch: {p.shape} != {t.shape}")

    gp = _gradient_magnitude(p)
    gt = _gradient_magnitude(t)

    # Phase-congruency-like feature: normalized local gradient strength.
    pc_p = gp / (np.max(gp) + 1e-6)
    pc_t = gt / (np.max(gt) + 1e-6)

    pc_sim = (2 * pc_p * pc_t + 0.85) / (pc_p * pc_p + pc_t * pc_t + 0.85)
    gm_sim = (2 * gp * gt + 90.0) / (gp * gp + gt * gt + 90.0)

    score = np.sqrt(pc_sim * gm_sim)
    return float(np.nanmean(score))


def gradient_difference(pred: np.ndarray, truth: np.ndarray) -> float:
    gp = _gradient_magnitude(pred)
    gt = _gradient_magnitude(truth)
    return float(np.mean(np.abs(gp - gt)))


def compute_frame_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    data_range: float = 255.0,
    use_fsims: bool = True,
) -> Dict[str, float]:
    """Compute image-quality metrics between predicted and ground-truth frames."""
    result = {
        "mse": mse(pred, truth),
        "rmse": rmse(pred, truth),
        "psnr": psnr(pred, truth, data_range=data_range),
        "ssim": ssim(pred, truth, data_range=data_range),
        "gradient_difference": gradient_difference(pred, truth),
    }
    if use_fsims:
        result["fsim_lite"] = fsim_lite(pred, truth)
    return result


def temporal_consistency(pred_seq: Sequence[np.ndarray], truth_seq: Sequence[np.ndarray]) -> float:
    """Mean absolute temporal-gradient error across aligned sequences."""
    if len(pred_seq) != len(truth_seq):
        raise ValueError("Sequences must have the same length.")
    if len(pred_seq) < 2:
        return 0.0

    diffs: List[float] = []
    for p0, p1, t0, t1 in zip(pred_seq[:-1], pred_seq[1:], truth_seq[:-1], truth_seq[1:]):
        dp = np.asarray(p1) - np.asarray(p0)
        dt = np.asarray(t1) - np.asarray(t0)
        diffs.append(float(np.mean(np.abs(dp - dt))))
    return float(np.mean(diffs))


def summarize_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {}
    keys = records[0].keys()
    summary: Dict[str, float] = {}
    for key in keys:
        values = [float(r[key]) for r in records if np.isfinite(r[key])]
        summary[key] = float(np.mean(values)) if values else float("nan")
    return summary
