# `m3/prompts/` — System Prompt, Response Schema, Validator (Phase 5)

The pieces that turn the offline LLM into a *bounded* Copilot. The LLM is
free to choose *which* runbook to cite and *which* order to recommend
actions in, but the shape of its output and the rules it must obey are
pinned by these files.

## Layout

```
m3/prompts/
  SYSTEM_PROMPT.md         the system-role contract (rules R1..R10, output shape)
  response_schema.json     strict JSON Schema (draft 2020-12) for the LLM's output
  schema_validator.py      validates LLM output + cross-field rules
  prompt_assembler.py      builds the full chat completion at runtime
  few_shot_examples.json   3 in-context exemplars (congestion, underlay, unknown)
  README.md                this file
```

## Files in this order of importance

1. **`SYSTEM_PROMPT.md`** — read the prompt first. Every rule (R1..R10) is
   derived from a constraint in the JSON schema; the prompt makes the
   constraints natural-language and the schema makes them machine-checkable.
2. **`response_schema.json`** — the strict output contract. The runtime
   refuses to accept anything that does not validate.
3. **`schema_validator.py`** — runs the JSON Schema check *and* the
   cross-field checks (evidence chunk IDs in `root_cause_hypothesis` and
   `recommended_actions` must exist in `provenance.evidence_chunks`).
4. **`prompt_assembler.py`** — composes the final chat completion from
   `SYSTEM_PROMPT` + `ALERT_PAYLOAD` + `RETRIEVED_EVIDENCE` + operator
   follow-up.
5. **`few_shot_examples.json`** — three exemplars: a congestion
   prediction (grounded), an underlay loss prediction (grounded with
   incident precedent), and an ungrounded case (answer_grounded: false
   with `INSUFFICIENT_EVIDENCE` warning). These are injected at the
   bottom of the system prompt or as user-role few-shots at deployment
   time.

## How the validator prevents hallucination

The validator runs after the LLM emits its raw text. It performs four
checks in order, and any failure raises `CopilotUnavailable`, which the
NOC UI renders as a clear "copilot unavailable" banner (never a silent
fallback):

| Check | Failure mode it catches |
|---|---|
| `_extract_json` | Model emitted prose, a code fence without JSON, or a JSON parse error. |
| `jsonschema.validate` | Model omitted a required field, used a value outside the enum, or invented a field. |
| evidence-chunk-id set equality | Model cited a `chunk_id` that was never in the RAG evidence. |
| action runbook-chunk-id set equality | Model recommended an action citing a runbook chunk that was not in the evidence. |

When the model emits a valid response but `answer_grounded` is `false`,
the validator does not reject; instead it surfaces the
`missing_context` and `warnings` to the operator, who can then decide
whether to re-run with more context.

## Trying the validator with the bundled few-shots

```bash
python -m m3.prompts.schema_validator \
  --response m3/prompts/few_shot_examples.json  # this would fail; the file is an array
```

Instead, drive it from Python:

```python
import json
from m3.prompts.schema_validator import validate_response, CopilotUnavailable

with open("m3/prompts/few_shot_examples.json") as f:
    examples = json.load(f)

for ex in examples:
    raw = json.dumps(ex["response"])
    try:
        validate_response(raw, evidence_chunks=ex.get("evidence", []))
        print(f"OK  {ex['name']}")
    except CopilotUnavailable as e:
        print(f"BAD {ex['name']}: {e}")
```

## Versioning contract

- `SYSTEM_PROMPT.md` is at v1.0.0. Bumping it requires bumping the
  `schema_version` in `response_schema.json` (to v1.1.0) and
  re-running the bundled self-tests.
- The model-side fingerprint in `provenance.embedding_model_fingerprint`
  and `index_manifest_sha256` are SHA-256 of the on-disk artefacts. If
  either changes, the run is auditable as a *new* Copilot build.
