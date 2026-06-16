"""Build the structured :class:`ALERT_PAYLOAD` from a model scoring result.

The alert payload is the **single contract** between the predictive
engine (Phase 3) and the offline LLM (Phase 5). The shape is pinned by
``controller/alert_schema.json`` and is a strict subset of the LLM
response schema (in fact, the LLM response schema's
``predicted_issue``, ``signals``, and ``provenance`` blocks are
designed to echo the alert payload's fields).

The alert builder is pure: it takes a :class:`ScoringResult` plus the
matching :class:`MetricFrame` and returns a JSON-serialisable dict.
The orchestrator stamps the runtime fields (``alert_id``,
``generated_at``, ``provenance``).
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema

from .metric_sampler import MetricFrame
from .model_scorer import ScoringResult

LOG = logging.getLogger("controller.alert")

SCHEMA_PATH = Path(__file__).parent / "alert_schema.json"
_SCHEMA: Optional[Dict[str, Any]] = None


def _load_schema() -> Dict[str, Any]:
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return _SCHEMA


# Mapping from the LSTM's "feature 0 = utilization" convention to the
# schema's `predicted_issue.type` enum. This is a *heuristic*: a real
# deployment would train a classifier head, but for the orchestrator
# integration a rules-based mapping is enough to drive the LLM end-to-end.
_RISK_BANDS = [
    (0.85, "CRITICAL"),
    (0.65, "ELEVATED"),
    (0.40, "GUARDED"),
    (0.0, "BASELINE"),
]

_SEVERITY_BY_RISK = {
    "CRITICAL": "P1",
    "ELEVATED": "P2",
    "GUARDED": "P3",
    "BASELINE": "P4",
    "UNKNOWN": "P4",
}

_VRF_HINT_BY_DEVICE_PREFIX = {
    "pe-hub": "cust-blue",
    "pe-dc": "cust-red",
    "p-core": "core",
    "ce-branch": "branch-inet",
}


def _classify(scoring: ScoringResult) -> str:
    """Map the (peak_forecast, peak_anomaly) pair to a closed enum."""
    peak = scoring.peak_forecast()
    anom = scoring.peak_anomaly()
    if anom < 0.30 and peak < 60.0:
        return "unknown"
    if peak >= 95.0 or anom >= 0.85:
        return "congestion_saturation"
    if peak >= 80.0:
        return "qos_starvation"
    if anom >= 0.70 and "bgp" in (scoring.interface or ""):
        return "bgp_session_flap"
    if anom >= 0.60 and "ospf" in (scoring.interface or ""):
        return "ospf_convergence_stress"
    if anom >= 0.50:
        return "tunnel_degradation"
    return "unknown"


def _risk_band(anom: float) -> str:
    for thresh, name in _RISK_BANDS:
        if anom >= thresh:
            return name
    return "UNKNOWN"


def _vrf_hint(host: str) -> str:
    for prefix, vrf in _VRF_HINT_BY_DEVICE_PREFIX.items():
        if host.startswith(prefix):
            return vrf
    return "default"


def _site_hint(host: str) -> str:
    # topology.clab.yml uses ``pe-hub-east-1`` / ``pe-dc-1`` etc.
    m = re.search(r"-(east|west|north|south|dc|hub|branch)-?(\d+)?", host)
    if m:
        region = m.group(1)
        n = m.group(2) or "1"
        return f"{region}-{n}"
    return "core"


def _cap_tti(minutes: float, lo: int, hi: int) -> int:
    if minutes <= 0:
        return lo
    return int(max(lo, min(hi, round(minutes))))


def build_alert_payload(
    scoring: ScoringResult,
    frame: MetricFrame,
    *,
    model_name: str,
    model_version: str,
    sampler_name: str,
    sequence_length: int,
    num_features: int,
    tti_minutes_min: int = 0,
    tti_minutes_max: int = 1440,
) -> Dict[str, Any]:
    """Assemble the strict ALERT_PAYLOAD for the LLM."""
    anom = scoring.peak_anomaly()
    forecast_peak = scoring.peak_forecast()
    issue_type = _classify(scoring)
    risk = _risk_band(anom)
    severity = _SEVERITY_BY_RISK.get(risk, "P4")

    tti_minutes = _cap_tti(
        scoring.median_tti(), tti_minutes_min, tti_minutes_max
    )

    alert_id = "ALR-" + str(int(time.time() * 1000))[-10:].rjust(10, "0")
    payload: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "alert_id": alert_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "severity": severity,
        "risk_band": risk,
        "predicted_issue": {
            "type": issue_type,
            "target": {
                "device": scoring.host,
                "interface_or_peer": scoring.interface,
                "vrf": _vrf_hint(scoring.host),
                "site": _site_hint(scoring.host),
            },
            "time_to_impact_minutes": tti_minutes,
            "confidence": round(min(1.0, max(0.0, anom)), 4),
        },
        "signals": frame.latest_signals[:16],
        "model": {
            "name": model_name,
            "version": model_version,
            "architecture": scoring.architecture,
            "device": "cpu",
            "horizon": len(scoring.forecast),
            "outputs": {
                "forecast": [round(x, 6) for x in scoring.forecast],
                "anomaly_prob": [round(x, 6) for x in scoring.anomaly_prob],
                "tti_minutes": [round(x, 4) for x in scoring.tti_minutes],
            },
        },
        "provenance": {
            "sampler": sampler_name,
            "sequence_length": int(sequence_length),
            "num_features": int(num_features),
            "checkpoint_sha256": scoring.checkpoint_sha256,
            "scaler_sha256": scoring.scaler_sha256,
        },
    }
    return payload


def validate_alert_payload(payload: Dict[str, Any]) -> None:
    """Strictly validate against ``alert_schema.json``. Raises
    :class:`jsonschema.ValidationError` on failure."""
    jsonschema.validate(instance=payload, schema=_load_schema())
