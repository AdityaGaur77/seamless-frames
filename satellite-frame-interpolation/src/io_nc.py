from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from netCDF4 import Dataset

VARIABLE_CANDIDATES = [
    "TIR1",
    "thermal_infrared",
    "brightness_temperature",
    "radiance",
    "data",
]


def _find_variable(ds: Dataset, requested: str | None = None) -> str:
    candidates = []
    if requested:
        candidates.append(requested)
    candidates.extend(VARIABLE_CANDIDATES)

    for name in candidates:
        if name in ds.variables:
            return name

    for name, var in ds.variables.items():
        dims = getattr(var, "dimensions", ())
        if len(dims) >= 2:
            return name

    raise KeyError("No 2D satellite variable found in NetCDF file.")


def _extract_2d(var) -> np.ndarray:
    arr = np.asarray(var[:])
    while arr.ndim > 2:
        arr = arr[-1]
    while arr.ndim < 2:
        arr = arr[np.newaxis, :]
    return arr.astype(np.float32)


def load_nc(path: str | Path, variable_name: str | None = None) -> np.ndarray:
    """Load a 2D satellite frame from a NetCDF file."""
    with Dataset(path, "r") as ds:
        var_name = _find_variable(ds, variable_name)
        return _extract_2d(ds.variables[var_name])


def _infer_attrs(ds: Dataset, var_name: str) -> dict:
    attrs = {}
    for dim in ds.dimensions:
        attrs[f"dim_{dim}"] = len(ds.dimensions[dim])

    var = ds.variables[var_name]
    for attr in ("units", "long_name", "standard_name", "scale_factor", "add_offset"):
        if hasattr(var, attr):
            attrs[attr] = getattr(var, attr)

    for attr in ("Conventions", "source", "institution", "platform", "satellite"):
        if hasattr(ds, attr):
            attrs[attr] = getattr(ds, attr)
    return attrs


def save_nc(
    path: str | Path,
    frame: np.ndarray,
    variable_name: str = "TIR1",
    time_offset_minutes: float | None = None,
    reference_path: str | Path | None = None,
) -> None:
    """Save a 2D frame to NetCDF with simple metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 2:
        raise ValueError("save_nc expects a 2D frame.")

    attrs = {"long_name": "Thermal infrared interpolated frame", "units": "K"}
    if reference_path:
        with Dataset(reference_path, "r") as ds:
            attrs.update(_infer_attrs(ds, _find_variable(ds, None)))

    with Dataset(path, "w", format="NETCDF4") as ds:
        y_dim, x_dim = frame.shape
        ds.createDimension("y", y_dim)
        ds.createDimension("x", x_dim)
        if time_offset_minutes is not None:
            ds.createDimension("time", 1)
            time_var = ds.createVariable("time", "f4", ("time",))
            time_var.units = "minutes since interpolated frame"
            time_var[:] = [time_offset_minutes]

        var = ds.createVariable(variable_name, "f4", ("y", "x"), zlib=True)
        var.long_name = attrs.get("long_name", "Thermal infrared interpolated frame")
        var.units = attrs.get("units", "K")
        var[:] = frame

        for key, value in attrs.items():
            if key not in ("long_name", "units"):
                try:
                    setattr(var, key, value)
                except TypeError:
                    setattr(var, key, str(value))


def save_png(path: str | Path, frame: np.ndarray) -> None:
    """Save a normalized frame as an 8-bit PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    norm = normalize_frame(frame)
    Image.fromarray(norm.astype(np.uint8), mode="L").save(path)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize a frame to 0..255 using percentile clipping."""
    arr = np.asarray(frame, dtype=np.float32)
    lo, hi = np.nanpercentile(arr, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    clipped = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (clipped * 255).astype(np.uint8)
