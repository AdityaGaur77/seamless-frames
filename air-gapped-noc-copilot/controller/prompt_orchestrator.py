"""Thin wrapper around :func:`m3.prompts.prompt_assembler.assemble`.

The prompt orchestrator takes the validated :class:`ALERT_PAYLOAD`
plus the :class:`RETRIEVED_EVIDENCE` envelope and returns the chat
messages list to ship to the offline LLM.

The contract with the LLM is owned by ``m3/prompts/SYSTEM_PROMPT.md``
(``v1.0.0``) and ``m3/prompts/response_schema.json`` (schema
``1.0.0``). This module is intentionally a no-op wrapper — its job is
to (a) provide a stable interface for the orchestrator and (b) make
the operator question (if any) explicit and testable.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("controller.prompts")


class PromptOrchestrator:
    """Build chat-completion messages for the offline LLM."""

    def __init__(self, *, operator_question: str = ""):
        from m3.prompts import prompt_assembler  # type: ignore

        self._assembler = prompt_assembler
        self._operator_question = operator_question

    def set_operator_question(self, text: str) -> None:
        self._operator_question = text or ""

    def build(
        self, alert: Dict[str, Any], evidence_envelope: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        evidence = evidence_envelope.get("evidence", [])
        messages = self._assembler.assemble(
            alert=alert,
            evidence=evidence,
            operator_question=self._operator_question,
        )
        # Token estimate is a cheap heuristic; the real model server
        # re-tokens.
        approx_tokens = self._assembler.estimate_tokens(messages)
        LOG.info(
            "Built prompt: %d messages, ~%d tokens, %d evidence chunks",
            len(messages),
            approx_tokens,
            len(evidence),
        )
        return messages
