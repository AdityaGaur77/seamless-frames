from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def _gaussian_cloud(shape: tuple[int, int], center: tuple[float, float], sigma: float, amplitude: float) -> np.ndarray:
    y, x = np.indices(shape)
    cy, cx = center
    g = np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma * sigma)))
    return amplitude * g


def generate_synthetic_frame(
    height: int = 128,
    width: int = 128,
    frame_idx: int = 0,
    speed: float = 3.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic geostationary TIR-like frame.

    The scene contains a warm background and several moving cloud blobs. The motion
    is intentionally smooth and mostly linear so the optical-flow baseline can be
    evaluated without external data.
    """
    rng = np.random.default_rng(seed + frame_idx)
    y = np.linspace(220, 235, height, dtype=np.float32)[:, None]
    x = np.linspace(225, 245, width, dtype=np.float32)[None, :]
    base = 230.0 + 0.03 * x + 0.02 * y

    t = frame_idx
    centers = [
        (height * (0.25 + 0.02 * np.sin(0.4 * t)), width * (0.20 + speed * t / width)),
        (height * (0.55 + 0.03 * np.cos(0.3 * t)), width * (0.60 + 0.7 * speed * t / width)),
        (height * (0.75 + 0.02 * np.sin(0.2 * t)), width * (0.35 + 1.1 * speed * t / width)),
    ]

    frame = base.copy()
    for i, center in enumerate(centers):
        sigma = 8 + i * 2
        amp = 8 + i * 1.5
        cloud = _gaussian_cloud((height, width), center, sigma, amp)
        frame += cloud

    # Add small high-frequency cloud texture and mild noise.
    texture = rng.normal(0, 0.35, size=(height, width))
    frame += texture
    return np.clip(frame, 190, 260).astype(np.float32)


def save_synthetic_frame(path: Path, frame: np.ndarray, frame_idx: int, time_step_minutes: float = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w", format="NETCDF4") as ds:
        h, w = frame.shape
        ds.createDimension("y", h)
        ds.createDimension("x", w)
        ds.createDimension("time", 1)

        ds.platform = "synthetic"
        ds.satellite = "demo_geostationary"
        ds.Conventions = "CF-1.8"

        time = ds.createVariable("time", "f4", ("time",))
        time.units = f"minutes since {frame_idx:04d}"
        time.long_name = "Synthetic acquisition time"
        time[:] = [frame_idx * time_step_minutes]

        tir = ds.createVariable("TIR1", "f4", ("y", "x"), zlib=True)
        tir.long_name = "Thermal infrared brightness temperature"
        tir.units = "K"
        tir.standard_name = "toa_brightness_temperature"
        tir[:] = frame

        lat = ds.createVariable("lat", "f4", ("y", "x"), zlib=True)
        lat.long_name = "Latitude"
        lat.units = "degrees_north"
        lat[:] = np.linspace(-10, 10, h)[:, None].repeat(w, axis=1)

        lon = ds.createVariable("lon", "f4", ("y", "x"), zlib=True)
        lon.long_name = "Longitude"
        lon.units = "degrees_east"
        lon[:] = np.linspace(60, 100, w)[None, :].repeat(h, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic satellite NetCDF frames for demo.")
    parser.add_argument("--out-dir", default="data/demo", type=Path)
    parser.add_argument("--num-frames", default=4, type=int)
    parser.add_argument("--height", default=128, type=int)
    parser.add_argument("--width", default=128, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cloud-speed", default=3.0, type=float)
    parser.add_argument("--time-step-minutes", default=20, type=float)
    args = parser.parse_args()

    for i in range(args.num_frames):
        frame = generate_synthetic_frame(
            height=args.height,
            width=args.width,
            frame_idx=i,
            speed=args.cloud_speed,
            seed=args.seed,
        )
        save_synthetic_frame(
            args.out_dir / f"frame_{i:04d}.nc",
            frame,
            frame_idx=i,
            time_step_minutes=args.time_step_minutes,
        )

    print(f"Generated {args.num_frames} synthetic frames in {args.out_dir}")


if __name__ == "__main__":
    main()
