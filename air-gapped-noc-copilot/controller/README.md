# Air-Gapped NOC Copilot — Phase 3 Orchestrator (`controller/`)

> **What this is.** A single-process Python controller that wires the
> three existing subsystems of the air-gapped NOC Copilot into one
> closed loop:
>
> | Subsystem | Owner | Folder |
> |---|---|---|
> | Live telemetry (Phase 1 / 2) | mimo v2.5 | `topology.clab.yml`, `telemetry_normalizer.py`, `init-timescaledb.sql` |
> | Predictive model (Phase 3) | mimo v2.5 | `lstm_model.py`, `tcn_model.py`, `data_preprocessor.py`, `best_model.pth` |
> | Offline RAG + LLM (Phase 4 / 5) | m3 | `m3/rag/`, `m3/prompts/` |
>
> The controller is a *bridge*, not a replacement. It reads the live
> TSDB, calls the trained PyTorch model, queries the local Chroma
> index, calls the local LLM, and validates the response. One file per
> concern, all in one package.

---

## 1. Data-flow loop architecture

```
                          ┌──────────────────────────────────────────────────────┐
                          │  controller/orchestrator.py  (single thread)         │
                          │  sample_interval_seconds  ·  SIGINT-clean shutdown    │
                          └──────────────────────────────────────────────────────┘
                                          │
                                          ▼
   ┌────────────────────┐    ┌────────────────────────┐    ┌──────────────────────┐
   │  1. metric_sampler │    │  2. model_scorer       │    │  3. alert_builder    │
   │  ──────────────    │    │  ─────────────         │    │  ──────────────      │
   │  TimescaleDBSampler│──▶ │  LSTMMultiTask         │──▶ │  alert_schema.json   │
   │  PrometheusSampler │    │  (cpu, eval mode)      │    │  schema_version 1.0  │
   │  FixtureSampler    │    │  + saved scaler        │    │  alert_id, severity, │
   │                    │    │                        │    │  predicted_issue,    │
   │  → MetricFrame     │    │  → ScoringResult       │    │  signals[],          │
   │    (T, F) tensor   │    │    forecast/anomaly/tti│    │  model.provenance    │
   └────────────────────┘    └────────────────────────┘    └──────────┬───────────┘
                                                                     │
                                                                     ▼
   ┌────────────────────┐    ┌────────────────────────┐    ┌──────────────────────┐
   │  4. rag_bridge     │    │  5. prompt_orchestrator│    │  6. infer_client     │
   │  ──────────────    │    │  ──────────────        │    │  ──────────────      │
   │  m3.rag.rag_query  │──▶ │  m3.prompts.           │──▶ │  http://127.0.0.1    │
   │  + alert filters   │    │  prompt_assembler      │    │  :8080/v1/chat       │
   │  (site/device/     │    │  SYSTEM_PROMPT v1.0    │    │  (llama.cpp / Ollama │
   │   protocol)        │    │  + ALERT_PAYLOAD       │    │   OpenAI-compat)     │
   │                    │    │  + RETRIEVED_EVIDENCE  │    │                      │
   │  → top-k chunks    │    │  + OPERATOR_QUESTION   │    │  → raw LLM text      │
   │                    │    │                        │    │                      │
   │  → evidence envelope│   │  → chat messages []    │    │                      │
   └────────────────────┘    └────────────────────────┘    └──────────┬───────────┘
                                                                     │
                                                                     ▼
                                                   ┌──────────────────────────┐
                                                   │  7. response_gate        │
                                                   │  ──────────────          │
                                                   │  m3.prompts.             │
                                                   │  schema_validator        │
                                                   │                          │
                                                   │  → NOC-ready JSON        │
                                                   │    OR                    │
                                                   │  → copilot_unavailable   │
                                                   │    envelope              │
                                                   └──────────┬───────────────┘
                                                              │
                                                              ▼
                                                   ┌──────────────────────────┐
                                                   │  sink (NDJSON + latest)   │
                                                   │  ──────────────          │
                                                   │  sink.ndjson (append)    │
                                                   │  latest.json (overwrite) │
                                                   │  stdout  (if disabled)   │
                                                   └──────────────────────────┘
```

### Per-stage contracts

| # | Module | Input | Output | Hard failures |
|---|---|---|---|---|
| 1 | `metric_sampler` | `(host, interface)` | `MetricFrame` (tensor + signals) | DB down, no targets |
| 2 | `model_scorer` | `MetricFrame.tensor` | `ScoringResult` (forecast / anomaly / tti) | Checkpoint missing, feature mismatch |
| 3 | `alert_builder` | `ScoringResult` + `MetricFrame` | ALERT_PAYLOAD dict | Schema violation (fail-loud) |
| 4 | `rag_bridge` | ALERT_PAYLOAD | evidence envelope | Index missing → RAG failure envelope |
| 5 | `prompt_orchestrator` | ALERT + evidence | chat messages | Prompt failure envelope |
| 6 | `infer_client` | chat messages | raw LLM text | `OfflineLLMUnavailable` envelope |
| 7 | `response_gate` | LLM text + evidence | validated `CopilotResponse` | `copilot_unavailable` envelope |

Every step is synchronous, side-effect free except the final sink,
and fail-loud (no silent fallbacks). When a step fails, the cycle
publishes a `copilot_unavailable` envelope with the failure reason;
the loop does *not* stop.

---

## 2. ALERT_PAYLOAD ↔ response schema

`ALERT_PAYLOAD` (Phase 3 → Phase 5) is a strict subset of the
LLM's response schema (Phase 5 → NOC UI). The validator
(`m3.prompts.schema_validator`) re-uses the LLM response schema to
validate the LLM's output and re-stamps `provenance.evidence_chunks`
with the chunks the RAG actually returned.

```
ALERT_PAYLOAD  ──feeds──▶  prompt_assembler.assemble()
                                  │
                                  ▼
                            chat messages
                                  │
                                  ▼
                         offline LLM (quantized)
                                  │
                                  ▼
                          response_schema.json
                          (schema_version 1.0.0)
                                  │
                                  ▼
                            CopilotResponse
                            (or copilot_unavailable)
```

The two schemas are designed to be **shape-compatible**:

| Field | ALERT_PAYLOAD | CopilotResponse |
|---|---|---|
| `schema_version` | `"1.0.0"` (pinned) | `"1.0.0"` (pinned) |
| `alert_id` | `ALR-XXXXXXXXXX` | echoes back |
| `generated_at` | stamped by orchestrator | echoes back |
| `predicted_issue` | LSTM/TCN output | LLM-narrated |
| `signals[]` | 1–16 raw signals | LLM re-summarises (or omits) |
| `provenance` | sampler / model fingerprints | sampler / model + RAG fingerprints |

---

## 3. RAG integration — anomaly ↔ runbook pairing

The RAG bridge (step 4) translates the alert into a structured
query. The mapping is deterministic and auditable:

| Alert field | RAG field | Source |
|---|---|---|
| `predicted_issue.target.device` | `where: device = ...` | alert |
| `predicted_issue.target.site` | `where: site = ...` | alert |
| `predicted_issue.type` | `where: protocol = ...` (mapped) | `_protocol_for_issue()` |
| `predicted_issue.type` + `signals[]` | query text | `_text_query_from_alert()` |

`m3.rag.rag_query.query()` runs the **hybrid BM25 + dense** retrieval
the m3 pipeline already implements, with the alert-derived filters
applied to the Chroma `where` clause (wrapped in `$and` to satisfy
Chroma's "exactly one operator" rule).

The returned envelope is passed verbatim to
`m3.prompts.prompt_assembler.assemble()`, which renders it into the
`RETRIEVED_EVIDENCE = {...}` block the LLM sees.

The validator then enforces the closed-world contract:

  * every `chunk_id` cited in `root_cause_hypothesis.evidence_chunks`
    must appear in `provenance.evidence_chunks`,
  * every `recommended_actions[*].linked_runbook_chunk_id` must
    appear in `provenance.evidence_chunks`.

That is the anti-hallucination guarantee: the LLM cannot cite a
runbook it was not given, and the operator UI cannot accidentally
surface a fabricated action.

---

## 4. Files

```
controller/
├── __init__.py                    # public surface
├── config.py                      # YAML → typed dataclass
├── alert_schema.json              # ALERT_PAYLOAD schema
├── alert_builder.py               # build + validate ALERT_PAYLOAD
├── metric_sampler.py              # Timescale / Prom / Fixture samplers
├── model_scorer.py                # PyTorch inference (LSTM/TCN/hybrid)
├── rag_bridge.py                  # wrap m3.rag.rag_query
├── prompt_orchestrator.py         # wrap m3.prompts.prompt_assembler
├── infer_client.py                # offline LLM HTTP (llama.cpp/Ollama)
├── response_gate.py               # wrap m3.prompts.schema_validator
├── orchestrator.py                # the main loop
├── controller_config.yaml         # default config
├── tests/
│   ├── __init__.py
│   └── test_controller_e2e.py     # 3 self-contained e2e tests
└── README.md                      # this file
```

---

## 5. Running

### 5.1 One-shot smoke test (no live deps)

```bash
cd air-gapped-noc-copilot
NOC_COPILOT_FAKE_EMBEDDER=1 \
NOC_COPILOT_SKIP_AIRGAP_CHECK=1 \
python -m controller.tests.test_controller_e2e
```

Expected: 3 tests pass; one NDJSON line in the temp sink, one
`latest.json` sidecar.

### 5.2 One cycle against the live stack

```bash
# Start the Phase 1/2 stack (Containerlab + Telegraf + Prometheus + TimescaleDB)
./deploy_topology.sh
cd telemetry-stack && docker-compose up -d && cd ..

# Start the inference server on 127.0.0.1:8080 (llama.cpp / Ollama / vLLM)
# e.g. ./llama-server -m models/airgap-noc.Q4_K_M.gguf --port 8080

cd air-gapped-noc-copilot
python -m controller.orchestrator --once
```

`--once` runs a single cycle and exits. The result is dumped to
stdout (or to `sink.ndjson` if `sink.ndjson_path` is set in
`controller_config.yaml`).

### 5.3 Continuous operation

```bash
python -m controller.orchestrator
# Ctrl-C to stop (responds to SIGINT, writes a final NDJSON line)
```

The orchestrator sleeps `sample_interval_seconds` between cycles and
is fully SIGINT/SIGTERM-clean.

### 5.4 Custom config

```bash
python -m controller.orchestrator --config path/to/config.yaml
```

---

## 6. Configuration

`controller/controller_config.yaml` is the single editable surface.
Hot fields:

| Field | Default | Notes |
|---|---|---|
| `sample_interval_seconds` | 30 | wall-clock period between cycles |
| `fail_loud` | true | publish a banner on cycle errors instead of failing silent |
| `sampler.backend` | `timescaledb` | `timescaledb` / `prometheus` / `fixture` |
| `sampler.sequence_length` | 60 | lookback window (must match `data_preprocessor`) |
| `sampler.num_features` | 25 | must equal `model.input_size` |
| `model.model_path` | `best_model.pth` | relative to project root |
| `model.architecture` | `lstm_multitask` | `lstm_multitask` / `lstm` / `tcn` / `hybrid` |
| `model.anomaly_threshold` | 0.5 | below this, skip the RAG/LLM call |
| `rag.config_path` | `m3/rag/rag_config.yaml` | m3's RAG config |
| `rag.index_root` | `m3/rag/index` | Chroma persistent dir |
| `llm.base_url` | `http://127.0.0.1:8080` | local inference server |
| `sink.ndjson_path` | (empty) | append-mode NDJSON sink |
| `sink.latest_path` | (empty) | sidecar "latest" file for the NOC UI |

---

## 7. Air-gap guarantees (inherited from m3)

* `_load_index` / `_disallow_network` are untouched in m3.
* `LocalEmbedder` honours `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`.
* The orchestrator binds the LLM HTTP client to `127.0.0.1` only
  (`session.trust_env = False`, no proxies, no redirects).
* The `provenance.checkpoint_sha256` and `provenance.scaler_sha256`
  fields let a security audit verify the bundled model + scaler
  haven't been swapped on disk.

---

## 8. What's tested

`controller/tests/test_controller_e2e.py` runs three self-contained
end-to-end tests, all using the deterministic `FixtureSampler` and a
stub LLM (no live database, no live inference server, no internet):

| Test | What it covers |
|---|---|
| `test_full_pipeline` | sampler → scorer → alert → RAG → prompt → stub LLM → validator → NDJSON sink |
| `test_alert_builder_schema` | ALERT_PAYLOAD strictly validates against `alert_schema.json` |
| `test_below_threshold_no_llm_call` | scoring below `anomaly_threshold` does **not** invoke the LLM |

Run with:

```bash
NOC_COPILOT_FAKE_EMBEDDER=1 \
NOC_COPILOT_SKIP_AIRGAP_CHECK=1 \
python -m controller.tests.test_controller_e2e
```
