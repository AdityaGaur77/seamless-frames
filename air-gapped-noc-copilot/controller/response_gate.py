"""Validate the LLM response and convert the validator envelope into a
NOC-UI-ready object.

Wraps :func:`m3.prompts.schema_validator.validate_response`. On
:class:`CopilotUnavailable`, the gate emits a banner envelope so the
operator UI can show "copilot unavailable" without crashing.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

LOG = logging.getLogger("controller.gate")


class ResponseGate:
    def __init__(self):
        from m3.prompts import schema_validator  # type: ignore

        self._validator = schema_validator

    def accept(self, llm_text: str, evidence_envelope: Dict[str, Any]) -> Dict[str, Any]:
        evidence_chunks = evidence_envelope.get("evidence", [])
        try:
            response = self._validator.validate_response(
                llm_text,
                evidence_chunks=evidence_chunks,
            )
        except self._validator.CopilotUnavailable as exc:
            LOG.warning("copilot_unavailable: %s", exc.reason)
            return exc.envelope

        response.setdefault("provenance", {})
        # Stamp retrieval score stats from the RAG envelope.
        manifest = evidence_envelope.get("manifest", {}) or {}
        response["provenance"]["retrieval_score_dense"] = float(
            sum(e.get("score", 0.0) for e in evidence_chunks)
            / max(1, len(evidence_chunks))
        )
        response["provenance"]["retrieval_score_lex"] = 0.0  # hybrid weighted
        return response
