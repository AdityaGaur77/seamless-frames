"""Thin wrapper around :func:`m3.rag.rag_query.query`.

The RAG bridge converts an :class:`ALERT_PAYLOAD` into a structured
text query plus a ``where`` filter, and asks the local Chroma index
for the top-k evidence chunks. The m3 module already implements the
hybrid BM25 + dense retrieval; this layer just:

  * translates alert fields into a natural-language query,
  * propagates site / device / protocol / vrf into the Chroma ``where``
    clause,
  * caps the per-cycle cost at the configured token budget,
  * tags every chunk with the originating alert so the prompt
    assembler can render the citation trail.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("controller.rag")


def _filters_from_alert(alert: Dict[str, Any]) -> Dict[str, Any]:
    pred = alert.get("predicted_issue", {})
    target = pred.get("target", {}) if isinstance(pred, dict) else {}
    issue_type = pred.get("type", "unknown") if isinstance(pred, dict) else "unknown"

    # The chunker's metadata fields (see m3/rag/chunker.py):
    #   site, device, protocol, runbook_id, root_cause_class, doc_type
    out: Dict[str, Any] = {
        "site": target.get("site"),
        "device": target.get("device"),
        "protocol": _protocol_for_issue(issue_type),
    }
    return {k: v for k, v in out.items() if v}


def _build_where_clause(filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Chroma's ``where`` requires exactly one top-level operator, so
    when the caller passes multiple key/value filters we wrap them in
    ``$and``.
    """
    pairs = [(k, v) for k, v in filters.items() if v is not None]
    if not pairs:
        return None
    if len(pairs) == 1:
        k, v = pairs[0]
        return {k: v}
    return {"$and": [{k: v} for k, v in pairs]}


def _protocol_for_issue(issue_type: str) -> Optional[str]:
    return {
        "bgp_session_flap": "bgp",
        "ospf_convergence_stress": "ospf",
        "tunnel_degradation": "ipsec",
        "underlay_packet_loss": "mpls",
        "qos_starvation": "qos",
        "congestion_saturation": "qos",
        "policy_drift": "sdwan",
    }.get(issue_type)


def _text_query_from_alert(alert: Dict[str, Any]) -> str:
    pred = alert.get("predicted_issue", {}) or {}
    target = pred.get("target", {}) or {}
    issue_type = pred.get("type", "unknown")
    return (
        f"{issue_type} on {target.get('device', '?')} "
        f"interface {target.get('interface_or_peer', '?')} "
        f"(site {target.get('site', '?')}, vrf {target.get('vrf', '?')}); "
        f"signals: " + ", ".join(
            f"{s.get('metric')}={s.get('value')}({s.get('trend')})"
            for s in alert.get("signals", [])[:6]
        )
    )


class RagBridge:
    """Calls ``m3.rag.rag_query.query`` with alert-derived filters."""

    def __init__(
        self,
        *,
        config_path: Path,
        index_root: Path,
        top_k: int = 8,
        context_token_budget: int = 3500,
        default_filters: Optional[Dict[str, str]] = None,
    ):
        # Local import keeps the orchestrator import-clean even if m3
        # is unbuilt.
        from m3.rag import rag_query  # type: ignore

        self._rag_query = rag_query
        self._config_path = Path(config_path)
        self._index_root = Path(index_root)
        self._top_k = int(top_k)
        self._context_token_budget = int(context_token_budget)
        self._default_filters = dict(default_filters or {})

    def retrieve(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        cfg_filters = _filters_from_alert(alert)
        filters = {**self._default_filters, **cfg_filters}
        where = _build_where_clause(filters)
        query_text = _text_query_from_alert(alert)

        LOG.info(
            "RAG query: filters=%s text=%r", filters, query_text[:120]
        )
        result = self._rag_query.query(
            cfg_path=self._config_path,
            index_root=self._index_root,
            query_text=query_text,
            top_k=self._top_k,
            filters=where,
            budget_tokens=self._context_token_budget,
        )
        # Re-stamp each evidence chunk with the originating alert id
        # so the prompt assembler can render the trail.
        alert_id = alert.get("alert_id", "")
        for ev in result.get("evidence", []):
            ev.setdefault("metadata", {})["alert_id"] = alert_id
        return result
