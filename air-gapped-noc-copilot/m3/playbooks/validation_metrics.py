"""Validation metrics for Phase 6 scenario reports.

Reads per-scenario JSON files written by `fault_injection_runner.py` and
computes the metrics required by the project specification:

  - prediction accuracy and lead time
  - explanation quality (grounded, no hallucination, operator-relevant)
  - air-gap integrity

Also produces a single dashboard summary JSON that can be rendered by
the NOC UI or by the offline report generator.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

LOG = logging.getLogger("noc_copilot.metrics")


def _load_reports(runs_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sorted(runs_dir.glob("scenario_*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError as e:
            LOG.warning("Could not parse %s: %s", p, e)
    return out


def _lead_time(report: Dict[str, Any]) -> float:
    return float(report.get("validation", {}).get("lead_time_minutes", 0.0) or 0.0)


def _grounding(report: Dict[str, Any]) -> float:
    return float(report.get("validation", {}).get("grounding_recall_at_k", 0.0) or 0.0)


def _fabrication(report: Dict[str, Any]) -> float:
    return float(report.get("validation", {}).get("fabrication_rate", 1.0) or 0.0)


def _action_applicability(report: Dict[str, Any]) -> float:
    return 1.0 if report.get("validation", {}).get("first_action_match") else 0.0


def _issue_type_accuracy(report: Dict[str, Any]) -> float:
    return 1.0 if report.get("validation", {}).get("predicted_issue_type_match") else 0.0


def _air_gap_integrity(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "scenarios_run": len(reports),
        "copilot_unavailable_count": sum(1 for r in reports if r.get("copilot_unavailable_reason")),
        "all_responses_grounded": all(r.get("validation", {}).get("answer_grounded") for r in reports),
        "all_evidence_chunk_ids_resolve": all(_fabrication(r) == 0.0 for r in reports),
    }


def summarise(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not reports:
        return {"scenarios_run": 0}
    n = len(reports)
    return {
        "scenarios_run": n,
        "mean_lead_time_minutes": round(sum(_lead_time(r) for r in reports) / n, 2),
        "min_lead_time_minutes": round(min(_lead_time(r) for r in reports), 2),
        "max_lead_time_minutes": round(max(_lead_time(r) for r in reports), 2),
        "mean_grounding_recall_at_k": round(sum(_grounding(r) for r in reports) / n, 3),
        "mean_fabrication_rate": round(sum(_fabrication(r) for r in reports) / n, 3),
        "mean_action_applicability": round(sum(_action_applicability(r) for r in reports) / n, 3),
        "mean_issue_type_accuracy": round(sum(_issue_type_accuracy(r) for r in reports) / n, 3),
        "air_gap_integrity": _air_gap_integrity(reports),
        "per_scenario": [
            {
                "scenario_id": r.get("scenario_id"),
                "name": r.get("name"),
                "lead_time_minutes": _lead_time(r),
                "grounding_recall_at_k": _grounding(r),
                "fabrication_rate": _fabrication(r),
                "action_applicability": _action_applicability(r),
                "issue_type_accuracy": _issue_type_accuracy(r),
                "copilot_unavailable_reason": r.get("copilot_unavailable_reason"),
            }
            for r in reports
        ],
    }


def render_markdown(summary: Dict[str, Any]) -> str:
    if summary.get("scenarios_run", 0) == 0:
        return "# Phase 6 Validation\n\n_No scenario reports found._\n"
    lines = [
        "# Phase 6 Validation Summary",
        "",
        f"- Scenarios run: **{summary['scenarios_run']}**",
        f"- Mean lead time: **{summary['mean_lead_time_minutes']} min** (min {summary['min_lead_time_minutes']}, max {summary['max_lead_time_minutes']})",
        f"- Mean grounding recall@8: **{summary['mean_grounding_recall_at_k']}**",
        f"- Mean fabrication rate: **{summary['mean_fabrication_rate']}**",
        f"- Mean action applicability: **{summary['mean_action_applicability']}**",
        f"- Mean issue-type accuracy: **{summary['mean_issue_type_accuracy']}**",
        "",
        "## Per-scenario",
        "",
        "| # | Name | Lead (min) | Grounding@8 | Fabrication | Action OK | Type OK |",
        "|---|------|-----------:|------------:|------------:|----------:|--------:|",
    ]
    for s in summary.get("per_scenario", []):
        lines.append(
            f"| {s['scenario_id']} | {s['name']} | {s['lead_time_minutes']:.1f} | {s['grounding_recall_at_k']:.2f} | {s['fabrication_rate']:.2f} | {int(s['action_applicability'])} | {int(s['issue_type_accuracy'])} |"
        )
    air = summary.get("air_gap_integrity", {})
    lines += [
        "",
        "## Air-gap integrity",
        "",
        f"- All responses grounded: **{air.get('all_responses_grounded')}**",
        f"- All evidence chunk IDs resolve: **{air.get('all_evidence_chunk_ids_resolve')}**",
        f"- Copilot unavailable count: **{air.get('copilot_unavailable_count')}**",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Compute Phase 6 validation metrics")
    p.add_argument("--runs", type=Path, required=True, help="Directory of scenario_*.json reports")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    reports = _load_reports(args.runs)
    summary = summarise(reports)
    md = render_markdown(summary)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
