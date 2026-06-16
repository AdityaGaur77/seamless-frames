# Air-Gapped NOC Copilot — End-to-End Architecture

> Authored by MiniMax M3 (Lead AI Architect role)
> Spans Phases 1–6 of the project specification.
> This document is the single-page system diagram and the component
> responsibility matrix. Read it before any single folder's README.

---

## 1. The three operator questions

The system is built to answer exactly these three questions, in real
time, in front of an NOC operator, with no internet access and no
hallucinated content:

| # | Question | Project-spec field | Where it lives in the response schema |
|---|----------|--------------------|---------------------------------------|
| Q1 | What is likely to fail next — and when? | `predicted_issue` | `predicted_issue.{type, target, time_to_impact_minutes, confidence}` |
| Q2 | Why is risk assessed as elevated — which signals contributed? | `root_cause_hypothesis` | `root_cause_hypothesis.{summary, signals[], evidence_chunks[], confidence}` |
| Q3 | What corrective action should be taken before SLA or security impact occurs? | `recommended_actions` | `recommended_actions[]` and the operator-facing `operator_questions.q3_corrective_action` |

## 2. End-to-end data flow

```
  ┌──────────────────────┐  SNMP  ┌────────────────────┐
  │ Containerlab         │───────▶│  Telegraf          │
  │ P/PE/CE routers      │        │  inputs.snmp       │──┐
  │ (FRR)                │ NetFlow│  inputs.netflow    │  │
  └──────────────────────┘───────▶│  outputs.prom      │  │
                                  └────────────────────┘  │
                                                          │ :9124
                                                          ▼
                                                  ┌────────────────┐
                                                  │  Prometheus    │
                                                  │  + recording   │
                                                  │  rules         │
                                                  └────────────────┘
                                                          │ PromQL
                                                          ▼
                                                  ┌────────────────┐
                                                  │ TimescaleDB    │
                                                  │ (long-term TSDB│
                                                  │  via remote    │
                                                  │  write)        │
                                                  └────────────────┘
                                                          │ time-series
                                                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │            Predictive Engine (Phase 3 \u2014 mimo v2.5)         │
  │  LSTM / Prophet / ensemble classifiers                       │
  │  Output: ALERT_PAYLOAD  {alert_id, severity, risk_band,      │
  │     time_to_impact_minutes, signals[], model_confidence}    │
  └─────────────────────────────────────────────────────────────┘
                                │  ALERT_PAYLOAD
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │              RAG Retriever  (m3/rag/)                        │
  │  1. Receive ALERT_PAYLOAD                                    │
  │  2. Build structured query (site/device/protocol filters)    │
  │  3. Hybrid BM25 + dense (bge-small-en-v1.5)                  │
  │  4. Return top-8 chunks from local Chroma / Qdrant           │
  │     corpus: topology + runbooks + incident history           │
  └─────────────────────────────────────────────────────────────┘
                                │  RETRIEVED_EVIDENCE
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │   Prompt Assembler  (m3/prompts/prompt_assembler.py)         │
  │   Builds: SYSTEM  +  ALERT_PAYLOAD                          │
  │          +  RETRIEVED_EVIDENCE  +  OPERATOR_QUESTION         │
  └─────────────────────────────────────────────────────────────┘
                                │  chat messages
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Offline LLM Inference  (quantized LLaMA 3 8B / Qwen 2.5 /   │
  │  Mistral 7B / Phi-3, served by llama.cpp / vLLM / Ollama)    │
  │  Output: a single JSON object                               │
  └─────────────────────────────────────────────────────────────┘
                                │  raw LLM text
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │   Schema Validator  (m3/prompts/schema_validator.py)         │
  │   1. _extract_json \u2014 reject prose / fenced garbage        │
  │   2. jsonschema.validate against response_schema.json        │
  │   3. evidence-chunk-id set equality                          │
  │   4. action runbook-chunk-id set equality                    │
  │   Output: validated CopilotResponse  OR  copilot_unavailable │
  └─────────────────────────────────────────────────────────────┘
                                │  validated JSON
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │   NOC UI / Alert Triage Pipeline                             │
  │   Renders q1/q2/q3 to the operator; shows recommended_actions│
  │   and citations; auto-fills the change-ticket draft.         │
  └─────────────────────────────────────────────────────────────┘
```

## 3. Component responsibility matrix (the AI Architect's "who does what")

| Component | Phase | Folder | Owned by | Air-gap | Notes |
|---|---|---|---|---|---|
| Containerlab topology (`topology.clab.yml`) | 1 | repo root | mimo v2.5 | n/a | Untouched by m3. |
| FRR `daemons` config | 1 | repo root | mimo v2.5 | n/a | Untouched by m3. |
| Telegraf config + SNMP targets YAML | 1/2 | `m3/telemetry/config/` | m3 | yes | Bind-mounted into the telegraf-collector node. |
| Prometheus scrape config | 1/2 | `m3/telemetry/config/` | m3 | yes | Bind-mounted into the prometheus node. |
| SNMP collector (pure Python) | 1/2 | `m3/telemetry/collectors/snmp_collector.py` | m3 | yes | Optional alternative to telegraf's snmp input. |
| NetFlow/IPFIX collector (pure Python) | 1/2 | `m3/telemetry/collectors/netflow_collector.py` | m3 | yes | Optional alternative to telegraf's netflow input. |
| OID \u2192 metric mapping | 1/2 | `m3/telemetry/collectors/mapper.py` | m3 | yes | Single source of truth for SNMP OIDs. |
| Predictive engine (LSTM/Prophet/ensemble) | 3 | (external) | mimo v2.5 | yes | Output is `ALERT_PAYLOAD`. |
| RAG chunker (3 chunker classes) | 4 | `m3/rag/chunker.py` | m3 | yes | Topology / Runbook / Incident chunkers. |
| Local embedder (bge-small-en-v1.5) | 4 | `m3/rag/embedder.py` | m3 | yes | Hard-fails if model missing. |
| Vector index (Chroma default / Qdrant opt) | 4 | `m3/rag/index_backend.py` | m3 | yes | `runtime.refuse_network_calls=true`. |
| RAG ingest / query | 4 | `m3/rag/rag_ingest.py`, `rag_query.py` | m3 | yes | Deterministic, idempotent. |
| Sample corpus | 4 | `m3/rag/corpus/` | m3 | yes | 1 topology + 3 runbooks + 1 incident. |
| System prompt + rules R1\u2013R10 | 5 | `m3/prompts/SYSTEM_PROMPT.md` | m3 | yes | Anti-hallucination contract. |
| JSON response schema (draft 2020-12) | 5 | `m3/prompts/response_schema.json` | m3 | yes | Pinned `schema_version: 1.0.0`. |
| Schema validator | 5 | `m3/prompts/schema_validator.py` | m3 | yes | 4-stage check. |
| Prompt assembler | 5 | `m3/rag/...` via `m3/prompts/prompt_assembler.py` | m3 | yes | SYSTEM + payload + evidence. |
| Few-shot exemplars | 5 | `m3/prompts/few_shot_examples.json` | m3 | yes | 3 examples; 1 ungrounded for the `INSUFFICIENT_EVIDENCE` warning. |
| Scenario playbooks | 6 | `m3/playbooks/scenario_*.md` | m3 | yes | Congestion, BGP flap, MPLS underlay, policy drift. |
| Fault-injection runner | 6 | `m3/playbooks/fault_injection_runner.py` | m3 | yes | Drives the 4 scenarios. |
| Validation metrics | 6 | `m3/playbooks/validation_metrics.py` | m3 | yes | lead time, grounding, fabrication, action applicability. |

## 4. Why this architecture is hard to bypass

The prompt + schema + validator form a **three-layer defence** against
hallucination, and the m3 deliverables add three more layers below:

| Layer | Defence | What it stops |
|---|---|---|
| 0. Embedder refuses to fall back | `LocalEmbedder` raises if model missing | Operator trusting wrong retrieval silently |
| 1. Hybrid BM25 + dense + filters | `rag_query.py` | Lexically missing chunk \u2192 ungrounded answer |
| 2. System-prompt rules R1\u2013R10 | `SYSTEM_PROMPT.md` | Model inventing device names / actions |
| 3. JSON schema enum + length limits | `response_schema.json` | Out-of-vocabulary types, runaway text |
| 4. Chunk-id set equality | `schema_validator.py` | Cited-but-unretrieved chunk_id |
| 5. Action runbook-chunk-id equality | `schema_validator.py` | Action that contradicts its cited runbook |

A bypass attempt has to defeat all five layers. Each layer is enforced
*in code*, not in the model; a deterministic check.

## 5. Air-gap integrity guarantees

These are *runtime* guarantees, not documentation:

- `_disallow_network()` in `rag_ingest.py` and `fault_injection_runner.py`
  raises if DNS to `huggingface.co` resolves. The script aborts.
- `LocalEmbedder` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
  before loading sentence-transformers; the library refuses to hit
  huggingface.co even by accident.
- `prompt_assembler.py` only reads local files; the inference server
  binds to `127.0.0.1`; the Prometheus / Telegraf / TimescaleDB
  services are all on the management network.
- The `provenance.embedding_model_fingerprint` and
  `provenance.index_manifest_sha256` fields in the response are
  SHA-256 of the on-disk artefacts. A security audit can verify the
  runtime model and index match the certified bundle.

## 6. The three operator-Q templates

The schema's `operator_questions` field is what the operator actually
reads. The prompt forces the model to fill it in as three strings:

| Field | Template | Why |
|---|---|---|
| `q1_what_will_fail` | `"The {device} {interface_or_peer} will {issue_type} in ~{time_to_impact_minutes} minutes, impacting {services} on {vrf}."` | 1 sentence, with the action verb from `predicted_issue.type` and the service impact from `affected_scope.services`. |
| `q2_why_elevated_risk` | `"Signals: {signals[]}. Evidence: {evidence_chunks[].quote}. Model confidence: {confidence}."` | 2\u20134 sentences, with the *citable* signal values, not paraphrased. |
| `q3_corrective_action` | `"{recommended_actions[0].action} on {recommended_actions[0].target} per {linked_runbook_chunk_id}. Rollback: {recommended_actions[0].rollback}."` | 2\u20134 sentences, with the action and the rollback. The operator can copy/paste this into a change ticket. |

The templates are not used to *generate* the strings; the LLM is free
to vary the wording. The templates are the **shape** of the answer the
NOC UI expects, which is why the schema enforces the length limits
(`q1 \u2264 400`, `q2 \u2264 600`, `q3 \u2264 600`).

## 7. Versioning & change management

| Artefact | Version field | How it is bumped |
|---|---|---|
| `SYSTEM_PROMPT.md` | semantic in the file (`v1.0.0`) | Major: new rule or relaxed constraint. Minor: new example. Patch: typo / wording. |
| `response_schema.json` | `schema_version: "1.0.0"` | Major: incompatible (removing enum, changing required). Minor: additive. Patch: cosmetic. |
| `few_shot_examples.json` | `name` field | Additive. New examples are added; old ones are not removed without a major schema bump. |
| `rag_config.yaml` | `schema_version: 1` | Config-only; does not require prompt / schema bumps. |
| `provenance.embedding_model_fingerprint` | SHA-256 of model dir | Auto, on each ingest. |
| `provenance.index_manifest_sha256` | SHA-256 of manifest | Auto, on each ingest. |

A schema or prompt change requires:
1. Bump `schema_version` in `response_schema.json`.
2. Bump the version stamp in `SYSTEM_PROMPT.md`.
3. Add a new few-shot to `few_shot_examples.json` covering the change.
4. Re-run `python -m m3.prompts.selftest_validator` and the four
   `m3/playbooks/fault_injection_runner` scenarios.

## 8. What the operator sees in the NOC UI

```
+--------------------------------------------------------------+
|  ALERT PE-1037 \u00b7 P3 \u00b7 ELEVATED \u00b7 15 min to impact         |
+--------------------------------------------------------------+
|  Q1. What will fail?                                          |
|  The pe-hub-east-1 uplink eth3 to p-core-1 will saturate in   |
|  ~12 minutes, dropping voice and video on cust-blue.          |
|                                                               |
|  Q2. Why is risk elevated?                                    |
|  if_out_util_pct on eth3 is 71.4% and rising;                 |
|  AF21-data queue depth is 482000 bytes and rising.            |
|  The ensemble model has 0.78 confidence based on the          |
|  recent 18-minute trend. (Citations: 2)                       |
|                                                               |
|  Q3. What should I do?                                        |
|  Open a change ticket, then raise the TE cost on              |
|  pe-hub-east-1 eth3 by 1000 per MPLS-003 step 6.              |
|  Monitor AF21 drops for 10 min; rollback with                 |
|  "no mpls traffic-eng interface eth3 metric 2000"             |
|  if traffic shifts are unexpected.                            |
|                                                               |
|  [View runbook MPLS-003]   [Open change ticket]  [Ack]       |
+--------------------------------------------------------------+
```

The raw LLM I/O is never shown; the operator only sees the
`operator_questions` block plus the structured alert. The validator
runs the moment the LLM emits; if it fails, the alert is rendered with
a `copilot_unavailable` banner and the operator falls back to the
underlying runbook chunk (which is what the prompt's R8
`INSUFFICIENT_EVIDENCE` warning triggers when it cannot ground).
