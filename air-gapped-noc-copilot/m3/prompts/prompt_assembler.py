"""Runtime prompt assembly for the NOC Copilot.

Builds the full chat completion to send to the offline LLM by combining:

  - the static system prompt (m3/prompts/SYSTEM_PROMPT.md)
  - the structured ALERT_PAYLOAD block
  - the RETRIEVED_EVIDENCE block from the RAG query
  - the operator's natural-language follow-up question (optional)

The output is a list of {role, content} chat messages ready to hand to the
local inference server (llama.cpp, vLLM, Ollama, etc.).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


SYSTEM_PROMPT_PATH = Path(__file__).parent / "SYSTEM_PROMPT.md"

_SYSTEM_PROMPT_CACHE: Optional[str] = None


def load_system_prompt() -> str:
    global _SYSTEM_PROMPT_CACHE
    if _SYSTEM_PROMPT_CACHE is None:
        text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        start = text.find("```text")
        end = text.find("```", start + 7)
        if start < 0 or end < 0:
            raise RuntimeError("SYSTEM_PROMPT.md missing the ```text ... ``` block")
        _SYSTEM_PROMPT_CACHE = text[start + 7:end].strip()
    return _SYSTEM_PROMPT_CACHE


def _truncate_strings(obj: Any, max_str_len: int) -> Any:
    if isinstance(obj, dict):
        return {k: _truncate_strings(v, max_str_len) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_strings(v, max_str_len) for v in obj]
    if isinstance(obj, str) and len(obj) > max_str_len:
        return obj[: max_str_len - 1] + "\u2026"
    return obj


def format_evidence_block(evidence: List[Dict[str, Any]]) -> str:
    lines = ["RETRIEVED_EVIDENCE = {", "  chunks = ["]
    for i, e in enumerate(evidence):
        lines.append(f"    [{i}] chunk_id={e['chunk_id']}")
        lines.append(f"        doc_type={e['metadata'].get('doc_type', 'unknown')}")
        lines.append(f"        source={e['metadata'].get('source', 'unknown')}")
        lines.append(f"        score={round(e['score'], 4)}")
        body = e["text"].replace("\n", "\n            ")
        lines.append(f"        text=<<<\n            {body}\n        >>>")
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines)


def format_alert_block(alert: Dict[str, Any]) -> str:
    return "ALERT_PAYLOAD = " + json.dumps(alert, indent=2, sort_keys=True)


def format_question_block(question: str) -> str:
    if not question.strip():
        return "OPERATOR_QUESTION = (none \u2014 answer the structured alert only)"
    return f"OPERATOR_QUESTION = <<<\n{question.strip()}\n>>>"


def assemble(alert: Dict[str, Any], evidence: List[Dict[str, Any]], operator_question: str = "") -> List[Dict[str, str]]:
    sys_prompt = load_system_prompt()
    user_payload = "\n\n".join([
        format_alert_block(alert),
        format_evidence_block(evidence),
        format_question_block(operator_question),
        "Respond with a single JSON object matching response_schema.json (schema_version=1.0.0). No prose.",
    ])
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_payload},
    ]


def estimate_tokens(messages: List[Dict[str, str]]) -> int:
    n = 0
    for m in messages:
        n += len(m["content"].split())
    return n


def main() -> int:
    p = argparse.ArgumentParser(description="Assemble a Copilot prompt and print the messages JSON to stdout")
    p.add_argument("--alert", required=True, type=Path, help="Path to ALERT_PAYLOAD JSON file")
    p.add_argument("--evidence", required=True, type=Path, help="Path to evidence.json from rag_query.py")
    p.add_argument("--question", default="", help="Optional operator follow-up question")
    p.add_argument("--out", type=Path, default=None, help="If set, write messages JSON here")
    args = p.parse_args()

    alert = json.loads(args.alert.read_text(encoding="utf-8"))
    ev_doc = json.loads(args.evidence.read_text(encoding="utf-8"))
    evidence = ev_doc.get("evidence", [])
    messages = assemble(alert, evidence, args.question)

    payload = {
        "messages": messages,
        "token_estimate": estimate_tokens(messages),
        "evidence_chunk_count": len(evidence),
    }
    out = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(out, encoding="utf-8")
    else:
        sys.stdout.write(out + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
