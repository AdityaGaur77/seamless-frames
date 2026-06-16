"""Strict validator for the offline LLM's response.

Validates the LLM's JSON output against `response_schema.json` and enforces
cross-field rules that the JSON Schema cannot express alone:

  1. Every chunk_id cited in `root_cause_hypothesis.evidence_chunks` must
     exist in `provenance.evidence_chunks`.
  2. Every `recommended_actions[*].linked_runbook_chunk_id` must exist in
     `provenance.evidence_chunks`.
  3. If `answer_grounded` is false, the validator still requires the JSON
     shape to be valid; the operator UI shows the `missing_context` and
     `warnings` instead of the recommendation.
  4. If the validator cannot parse the LLM output as JSON, it returns a
     `copilot_unavailable` envelope so the operator UI shows a clear
     failure mode rather than a silent miss.

Use:
    from m3.prompts.schema_validator import validate_response, CopilotUnavailable
    try:
        result = validate_response(llm_text, evidence_chunks=evs)
    except CopilotUnavailable as e:
        show_banner(e.envelope())

The validator never *modifies* the LLM output silently; if it truncates a
string to fit the schema, it returns the modified copy alongside a
`warnings += ["TRUNCATED"]` flag.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

LOG = logging.getLogger("noc_copilot.validator")

SCHEMA_PATH = Path(__file__).parent / "response_schema.json"
_SCHEMA: Optional[Dict[str, Any]] = None


def _load_schema() -> Dict[str, Any]:
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return _SCHEMA


def _extract_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    text = text.strip()
    decoder = json.JSONDecoder()

    if text.startswith("{"):
        try:
            obj, _ = decoder.raw_decode(text)
            if isinstance(obj, dict):
                return obj, None
        except json.JSONDecodeError as e:
            return None, f"json_parse_error: {e.msg} at line {e.lineno} col {e.colno}"

    fence = re.search(r"```(?:json)?\s*", text)
    if fence:
        start = fence.end()
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj, None
        except json.JSONDecodeError as e:
            return None, f"json_parse_error_in_fence: {e.msg} at pos {e.pos}"

    first_brace = text.find("{")
    if first_brace > 0:
        try:
            obj, _ = decoder.raw_decode(text[first_brace:])
            if isinstance(obj, dict):
                return obj, None
        except json.JSONDecodeError:
            pass

    return None, "no_json_object_found"


@dataclass
class CopilotUnavailable(Exception):
    reason: str
    envelope: Dict[str, Any]

    def __str__(self) -> str:
        return f"copilot_unavailable: {self.reason}"


def _truncate_to_limit(value: Any, limit: int) -> Tuple[Any, bool]:
    if isinstance(value, str) and len(value) > limit:
        return value[: limit - 1] + "\u2026", True
    return value, False


def _truncate_strings_in_object(obj: Dict[str, Any], limits: Dict[str, int]) -> Tuple[Dict[str, Any], bool]:
    out: Dict[str, Any] = {}
    truncated = False
    for k, v in obj.items():
        if k in limits:
            new_v, was_trunc = _truncate_to_limit(v, limits[k])
            truncated = truncated or was_trunc
            out[k] = new_v
        else:
            out[k] = v
    return out, truncated


# Field-level maxLengths used to pre-truncate the LLM output *before*
# JSON-Schema validation. The schema's maxLength constraints are
# authoritative; this map exists so the validator can trim a slightly-over
# field and pass the LLM response through with a "TRUNCATED" warning
# instead of rejecting it outright. If you add a new string field with a
# length cap to response_schema.json, add its maxLength here too.
_PRE_TRUNCATE_LIMITS = {
    ("root_cause_hypothesis", "summary"): 600,
    ("recommended_actions", "expected_effect"): 240,
    ("recommended_actions", "rollback"): 240,
    ("operator_questions", "q1_what_will_fail"): 400,
    ("operator_questions", "q2_why_elevated_risk"): 600,
    ("operator_questions", "q3_corrective_action"): 600,
}


def _pre_truncate(obj: Dict[str, Any], path: Tuple[str, ...] = ()) -> bool:
    truncated = False
    if isinstance(obj, dict):
        parent = path[-1] if path else ""
        for k, v in list(obj.items()):
            limit = _PRE_TRUNCATE_LIMITS.get((parent, k))
            if isinstance(v, str) and limit is not None and len(v) > limit:
                obj[k] = v[: limit - 1] + "\u2026"
                truncated = True
            elif _pre_truncate(v, path + (k,)):
                truncated = True
    elif isinstance(obj, list):
        for item in obj:
            if _pre_truncate(item, path):
                truncated = True
    return truncated


def validate_response(
    llm_text: str,
    evidence_chunks: Optional[List[Dict[str, Any]]] = None,
    evidence_provenance: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    schema = _load_schema()
    parsed, err = _extract_json(llm_text)
    if parsed is None:
        raise CopilotUnavailable(
            reason=err or "json_parse_failed",
            envelope={
                "copilot_unavailable": True,
                "reason": err or "json_parse_failed",
                "raw_excerpt": (llm_text or "")[:400],
            },
        )

    if evidence_provenance is None and evidence_chunks is not None:
        evidence_provenance = [
            {
                "chunk_id": e.get("chunk_id"),
                "doc_type": e.get("metadata", {}).get("doc_type", "unknown"),
                "source": e.get("metadata", {}).get("source", "unknown"),
                "score": float(e.get("score", 0.0)),
            }
            for e in evidence_chunks
        ]

    parsed.setdefault("provenance", {})
    if evidence_provenance is not None:
        parsed["provenance"]["evidence_chunks"] = evidence_provenance

    if _pre_truncate(parsed):
        parsed.setdefault("warnings", [])
        if "TRUNCATED" not in parsed["warnings"]:
            parsed["warnings"].append("TRUNCATED")

    try:
        jsonschema.validate(instance=parsed, schema=schema)
    except jsonschema.ValidationError as e:
        raise CopilotUnavailable(
            reason=f"schema_violation: {e.message}",
            envelope={"copilot_unavailable": True, "reason": "schema_violation", "path": list(e.absolute_path), "message": e.message},
        )

    evidence_ids = {c["chunk_id"] for c in parsed["provenance"]["evidence_chunks"]}

    bad_hyp = [c["chunk_id"] for c in parsed["root_cause_hypothesis"].get("evidence_chunks", []) if c["chunk_id"] not in evidence_ids]
    if bad_hyp:
        raise CopilotUnavailable(
            reason=f"ungrounded_evidence_chunks: {bad_hyp}",
            envelope={"copilot_unavailable": True, "reason": "ungrounded_evidence_chunks", "bad_ids": bad_hyp},
        )

    bad_actions = [a["linked_runbook_chunk_id"] for a in parsed["recommended_actions"] if a.get("linked_runbook_chunk_id") and a["linked_runbook_chunk_id"] not in evidence_ids]
    if bad_actions:
        raise CopilotUnavailable(
            reason=f"ungrounded_action_chunks: {bad_actions}",
            envelope={"copilot_unavailable": True, "reason": "ungrounded_action_chunks", "bad_ids": bad_actions},
        )

    if parsed.get("answer_grounded") is False:
        if not parsed.get("missing_context"):
            parsed["missing_context"] = ["insufficient evidence in local corpus"]
        if not any(w == "INSUFFICIENT_EVIDENCE" for w in parsed.get("warnings", [])):
            parsed.setdefault("warnings", []).append("INSUFFICIENT_EVIDENCE")

    return parsed


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Validate a Copilot response from a file")
    p.add_argument("--response", required=True, type=Path)
    p.add_argument("--evidence", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    raw = args.response.read_text(encoding="utf-8")
    evidence_chunks = None
    if args.evidence:
        ed = json.loads(args.evidence.read_text(encoding="utf-8"))
        evidence_chunks = ed.get("evidence", [])

    try:
        out = validate_response(raw, evidence_chunks=evidence_chunks)
        status = "OK"
    except CopilotUnavailable as e:
        out = e.envelope
        status = "UNAVAILABLE"

    payload = {"status": status, "result": out}
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(serialized, encoding="utf-8")
    else:
        print(serialized)
    return 0 if status == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
