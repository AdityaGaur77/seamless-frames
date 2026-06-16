from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_ready(v) for v in obj]
    return obj


def write_report(report: dict, output_dir: str | Path) -> None:
    """Write JSON and HTML reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "metrics.json"
    html_path = output_dir / "report.html"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(report), f, indent=2)

    metrics = report.get("metrics", {})
    frame_metrics = report.get("frame_metrics", [])

    metric_rows = []
    for row in frame_metrics:
        metric_rows.append(
            "<tr>"
            + "".join(
                f"<td>{row.get(k, '')}</td>"
                for k in [
                    "frame_index",
                    "pair",
                    "alpha",
                    "mse",
                    "rmse",
                    "psnr",
                    "ssim",
                    "gradient_difference",
                    "fsim_lite",
                ]
            )
            + "</tr>"
        )

    summary_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.items())

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Satellite Frame Interpolation Report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; color: #111; }}
    h1, h2 {{ color: #123; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.45rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f3f6fb; }}
    .card {{ border: 1px solid #d9e2ef; border-radius: 10px; padding: 1rem; margin: 1rem 0; }}
    code {{ background: #f4f4f4; padding: 0.1rem 0.25rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Satellite Frame Interpolation Report</h1>
  <div class="card">
    <p><strong>Input directory:</strong> <code>{report.get('input_dir')}</code></p>
    <p><strong>Output directory:</strong> <code>{report.get('output_dir')}</code></p>
    <p><strong>Variable:</strong> <code>{report.get('variable_name')}</code></p>
    <p><strong>Interpolation factor:</strong> {report.get('interpolation_factor')}</p>
    <p><strong>Input frames:</strong> {report.get('num_input_frames')}</p>
    <p><strong>Output frames:</strong> {report.get('num_output_frames')}</p>
  </div>

  <h2>Summary Metrics</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{summary_rows}</tbody>
  </table>

  <h2>Per-frame Metrics</h2>
  <table>
    <thead>
      <tr>
        <th>Frame</th><th>Pair</th><th>Alpha</th><th>MSE</th><th>RMSE</th><th>PSNR</th><th>SSIM</th><th>Grad Diff</th><th>FSIM-lite</th>
      </tr>
    </thead>
    <tbody>{''.join(metric_rows)}</tbody>
  </table>
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")
