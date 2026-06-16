from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class AppConfig:
    input_dir: Path
    output_dir: Path
    variable_name: str
    interpolation_factor: int
    time_step_minutes: float
    demo: Dict[str, Any]
    metrics: Dict[str, Any]
    dashboard: Dict[str, Any]


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load YAML configuration and normalize paths."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return AppConfig(
        input_dir=Path(raw.get("input_dir", "data/demo")),
        output_dir=Path(raw.get("output_dir", "demo_output")),
        variable_name=raw.get("variable_name", "TIR1"),
        interpolation_factor=int(raw.get("interpolation_factor", 2)),
        time_step_minutes=float(raw.get("time_step_minutes", 20)),
        demo=dict(raw.get("demo", {})),
        metrics=dict(raw.get("metrics", {})),
        dashboard=dict(raw.get("dashboard", {})),
    )
