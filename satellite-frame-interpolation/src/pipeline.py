from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from .io_nc import load_nc, save_png


def _load_nc_files(input_dir: Path, variable_name: str) -> Tuple[List[Path], List[np.ndarray]]:
    paths = sorted(input_dir.glob("*.nc"))
    if not paths:
        raise FileNotFoundError(f"No .nc files found in {input_dir}")
    frames = [load_nc(p, variable_name) for p in paths]
    return paths, frames


def _sanitize(value):
    if isinstance(value, (np.floating, float)):
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        if np.isnan(value):
            return None
    return value


def run_pipeline(config_path: str | Path = "config.yaml") -> dict:
    """Run the full demo pipeline: load, interpolate, validate, save, and report."""
    from .config import load_config
    from .interpolation import interpolate_sequence
    from .metrics import compute_frame_metrics, summarize_metrics, temporal_consistency
    from .report import write_report

    cfg = load_config(config_path)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    paths, frames = _load_nc_files(cfg.input_dir, cfg.variable_name)
    if len(frames) < 2:
        raise ValueError("At least two frames are required for interpolation.")

    interpolated, meta = interpolate_sequence(frames, factor=cfg.interpolation_factor, sharpen=True)

    # Save interpolated frames and PNGs.
    interp_dir = cfg.output_dir / "interpolated"
    interp_dir.mkdir(parents=True, exist_ok=True)
    image_dir = cfg.output_dir / "images" / "interpolated"
    image_dir.mkdir(parents=True, exist_ok=True)
    original_image_dir = cfg.output_dir / "images" / "original"
    original_image_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame in enumerate(frames):
        save_png(original_image_dir / f"frame_{idx:04d}.png", frame)

    for idx, frame in enumerate(interpolated):
        save_png(image_dir / f"frame_{idx:04d}.png", frame)
        save_nc(
            interp_dir / f"frame_{idx:04d}.nc",
            frame,
            variable_name=cfg.variable_name,
            time_offset_minutes=None,
            reference_path=paths[min(idx // cfg.interpolation_factor, len(paths) - 1)],
        )

    # Validate interpolated frames against ground truth where available.
    metric_records = []
    data_range = float(cfg.metrics.get("psnr_data_range", 255.0))
    use_fsims = bool(cfg.metrics.get("use_fsims", True))

    for idx, item in enumerate(meta):
        if item["type"] == "interpolated":
            pair = item["pair"]
            alpha = item["alpha"]
            gt_idx = pair[0] + int(round(alpha * (pair[1] - pair[0])))
            if gt_idx < len(frames):
                metrics = compute_frame_metrics(
                    interpolated[idx], frames[gt_idx], data_range=data_range, use_fsims=use_fsims
                )
                metrics.update(
                    {
                        "frame_index": idx,
                        "pair": pair,
                        "alpha": alpha,
                        "type": "interpolated",
                    }
                )
                metric_records.append(metrics)

    summary = summarize_metrics(metric_records)
    if len(interpolated) == len(frames):
        summary["temporal_consistency"] = temporal_consistency(interpolated, frames)
    else:
        summary["temporal_consistency"] = None

    report = {
        "input_dir": str(cfg.input_dir),
        "output_dir": str(cfg.output_dir),
        "variable_name": cfg.variable_name,
        "interpolation_factor": cfg.interpolation_factor,
        "num_input_frames": len(frames),
        "num_output_frames": len(interpolated),
        "metrics": summary,
        "frame_metrics": metric_records,
        "meta": meta,
    }
    write_report(report, cfg.output_dir)

    return report
