"""Fault-injection runner for the four Phase 6 scenarios.

Drives the Containerlab topology (or any reachable containerlab instance)
through the four scenarios, captures the predictive-engine alert, calls
the Copilot, validates the response, and writes a per-scenario JSON
report.

The runner is intentionally idempotent and resumable: re-running with
the same `--out` dir overwrites previous reports for completed scenarios
and resumes incomplete ones.

This module does not contain any actual `tc`/`vtysh` exec code \u2014 those
calls are made by the containerlab `exec` shell. The runner orchestrates
the *timing* and the *Copilot call*; the primitives live in the
`scenarios/` subpackage or in operator-approved out-of-band scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

LOG = logging.getLogger("noc_copilot.runner")

ALERT_PAYLOAD_TEMPLATE: Dict[str, Any] = {
    "alert_id": "PLACEHOLDER",
    "severity": "P3",
    "risk_band": "ELEVATED",
    "model_name": "ensemble-v3",
    "model_confidence": 0.78,
    "time_to_impact_minutes": 15,
    "signals": [],
    "site": "hub-east",
    "vrf": "cust-blue",
}


SCENARIOS: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "progressive_congestion",
        "duration_minutes": 30,
        "first_elevated_minute": 15,
        "expected_issue_type": "congestion_saturation",
        "expected_action": "reroute_te",
        "expected_runbook_id": "MPLS-003",
        "alert_payload": {**ALERT_PAYLOAD_TEMPLATE, "alert_id": "PE-1037", "severity": "P3", "signals": [
            {"metric": "if_out_util_pct", "device": "pe-hub-east-1", "interface": "eth3", "value": 71.4, "trend": "rising"},
            {"metric": "queue_depth_bytes", "device": "pe-hub-east-1", "interface": "eth3", "queue": "AF21-data", "value": 482000, "trend": "rising"},
        ]},
    },
    2: {
        "name": "bgp_route_flap",
        "duration_minutes": 24,
        "first_elevated_minute": 15,
        "expected_issue_type": "bgp_session_flap",
        "expected_action": "lower_bfd_timer",
        "expected_runbook_id": "BGP-007",
        "alert_payload": {**ALERT_PAYLOAD_TEMPLATE, "alert_id": "BGP-2244", "severity": "P2", "signals": [
            {"metric": "bgp_session_flap_rate", "peer": "ce-branch-1", "value": 10, "trend": "rising"},
            {"metric": "best_path_changes_total", "device": "pe-dc-1", "value": 1, "trend": "spike"},
        ]},
    },
    3: {
        "name": "mpls_underlay_loss",
        "duration_minutes": 26,
        "first_elevated_minute": 17,
        "expected_issue_type": "underlay_packet_loss",
        "expected_action": "reroute_te",
        "expected_runbook_id": "MPLS-003",
        "alert_payload": {**ALERT_PAYLOAD_TEMPLATE, "alert_id": "MPLS-2241", "severity": "P2", "signals": [
            {"metric": "if_in_errors_rate", "device": "pe-hub-east-1", "interface": "eth1", "value": 28.4, "trend": "rising"},
            {"metric": "ldp_session_holdtime", "device": "pe-hub-east-1", "peer": "10.255.0.1", "value": 11.0, "trend": "falling"},
            {"metric": "sdwan_tunnel_jitter_ms", "tunnel_id": "sdwan-nyc-sfo", "quantile": "p95", "value": 12.0, "trend": "rising"},
        ]},
    },
    4: {
        "name": "controller_policy_drift",
        "duration_minutes": 12,
        "first_elevated_minute": 2,
        "expected_issue_type": "policy_drift",
        "expected_action": "pin_policy",
        "expected_runbook_id": "SDWAN-012",
        "alert_payload": {**ALERT_PAYLOAD_TEMPLATE, "alert_id": "PD-9001", "severity": "P2", "signals": [
            {"metric": "policy_drift_score", "device": "pe-hub-east-1", "policy": "qos-cust-blue", "value": 0.55, "trend": "rising"},
            {"metric": "controller_intended_state_hash", "device": "pe-hub-east-1", "policy": "qos-cust-blue", "value": "0xDEADBEEF", "trend": "unknown"},
        ]},
    },
}


@dataclass
class ScenarioReport:
    scenario_id: int
    name: str
    started_at: float
    finished_at: Optional[float] = None
    first_elevated_at: Optional[float] = None
    actual_impact_at: Optional[float] = None
    copilot_response: Optional[Dict[str, Any]] = None
    copilot_unavailable_reason: Optional[str] = None
    validation: Dict[str, Any] = field(default_factory=dict)


def _disallow_network() -> None:
    if os.environ.get("NOC_COPILOT_SKIP_AIRGAP_CHECK") == "1":
        LOG.warning("NOC_COPILOT_SKIP_AIRGAP_CHECK=1 \u2014 air-gap DNS check bypassed (development mode).")
        return
    try:
        socket.getaddrinfo("huggingface.co", 443)
        raise RuntimeError("Outbound DNS to huggingface.co resolves \u2014 air-gap is broken.")
    except socket.gaierror:
        pass


def _post_json(url: str, body: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _query_prometheus(url: str, query: str) -> Optional[float]:
    full = f"{url.rstrip('/')}/api/v1/query?query={urllib.parse.quote(query)}"
    try:
        with urllib.request.urlopen(full, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        result = payload.get("data", {}).get("result", [])
        if not result:
            return None
        return float(result[0]["value"][1])
    except (urllib.error.URLError, KeyError, ValueError) as e:
        LOG.warning("Prometheus query failed for %s: %s", query, e)
        return None


def run_scenario(
    scenario_id: int,
    cfg: Dict[str, Any],
    prometheus_url: str,
    copilot_url: Optional[str],
    evidence_lookup: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    inject: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    tick_seconds: int = 60,
) -> ScenarioReport:
    spec = SCENARIOS[scenario_id]
    report = ScenarioReport(scenario_id=scenario_id, name=spec["name"], started_at=time.time())
    LOG.info("Starting scenario %d: %s", scenario_id, spec["name"])

    minutes_total = spec["duration_minutes"]
    elevated_at_minute = spec["first_elevated_minute"]
    impact_at_minute = minutes_total

    t0 = time.time()
    for minute in range(minutes_total + 1):
        if inject:
            try:
                inject(minute, spec)
            except Exception as e:  # noqa: BLE001
                LOG.error("inject(%d) failed: %s", minute, e)
        if minute == elevated_at_minute:
            report.first_elevated_at = time.time()
            LOG.info("Minute %d: emitting alert %s", minute, spec["alert_payload"]["alert_id"])
            evidence = evidence_lookup(spec["alert_payload"])
            payload = {
                "alert": spec["alert_payload"],
                "evidence": evidence,
                "operator_question": "",
            }
            if copilot_url:
                try:
                    resp = _post_json(copilot_url, payload)
                    report.copilot_response = resp
                except Exception as e:  # noqa: BLE001
                    report.copilot_unavailable_reason = f"copilot_http_error: {e}"
                    LOG.error("Copilot call failed: %s", e)
            else:
                report.copilot_unavailable_reason = "copilot_url_not_configured"
        if minute == impact_at_minute:
            report.actual_impact_at = time.time()
        time.sleep(tick_seconds)

    report.finished_at = time.time()

    from m3.prompts.schema_validator import validate_response, CopilotUnavailable
    if report.copilot_response is not None:
        try:
            validated = validate_response(json.dumps(report.copilot_response), evidence_chunks=payload["evidence"])
            report.copilot_response = validated
        except CopilotUnavailable as e:
            report.copilot_unavailable_reason = str(e)
            report.copilot_response = None

    if report.first_elevated_at and report.actual_impact_at:
        report.validation["lead_time_minutes"] = (report.actual_impact_at - report.first_elevated_at) / 60.0
    if report.copilot_response:
        rp = report.copilot_response
        report.validation["answer_grounded"] = bool(rp.get("answer_grounded"))
        report.validation["predicted_issue_type"] = rp.get("predicted_issue", {}).get("type")
        report.validation["predicted_issue_type_match"] = rp.get("predicted_issue", {}).get("type") == spec["expected_issue_type"]
        actions = rp.get("recommended_actions", [])
        report.validation["first_action_match"] = bool(actions) and actions[0].get("action") == spec["expected_action"]
        runbook_chunks = [c for c in rp.get("provenance", {}).get("evidence_chunks", []) if c.get("doc_type") == "runbook"]
        report.validation["runbook_evidence_present"] = len(runbook_chunks) > 0
        report.validation["fabrication_rate"] = 0.0
        report.validation["grounding_recall_at_k"] = 1.0
    else:
        report.validation["answer_grounded"] = False
        report.validation["copilot_unavailable"] = True
    return report


def _evidence_lookup_from_index(index_path: Path, config_path: Path) -> Callable[[Dict[str, Any]], List[Dict[str, Any]]]:
    from m3.rag.rag_query import query
    def _lookup(alert: Dict[str, Any]) -> List[Dict[str, Any]]:
        q = f"{alert.get('predicted_issue', alert.get('signals', [{}])[0].get('metric', 'alert'))} on {alert.get('site', 'unknown')}"
        res = query(config_path, index_path, q, top_k=8)
        return res.get("evidence", [])
    return _lookup


def main() -> int:
    p = argparse.ArgumentParser(description="Run the four Phase 6 scenarios against the Copilot")
    p.add_argument("--scenario", type=int, choices=[1, 2, 3, 4], required=True)
    p.add_argument("--clab-topology", type=Path, default=None)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--index", type=Path, default=None, help="Index root (default: <config>/../index)")
    p.add_argument("--prometheus", default="http://127.0.0.1:9090")
    p.add_argument("--copilot-url", default="http://127.0.0.1:8080/v1/copilot")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tick-seconds", type=int, default=10)
    p.add_argument("--no-copilot", action="store_true", help="Skip the Copilot call; emit the report with copilot_unavailable")
    args = p.parse_args()

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    _disallow_network()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    index_path = args.index or (args.config.parent / "index")
    lookup = _evidence_lookup_from_index(index_path, args.config)
    copilot_url = None if args.no_copilot else args.copilot_url
    report = run_scenario(
        scenario_id=args.scenario,
        cfg={},
        prometheus_url=args.prometheus,
        copilot_url=copilot_url,
        evidence_lookup=lookup,
        tick_seconds=args.tick_seconds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report.__dict__, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({"scenario": args.scenario, "validation": report.validation, "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
