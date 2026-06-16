# Air-Gapped NOC Copilot — `m3/` Contribution

This subtree contains the deliverables authored by **MiniMax M3** (the Lead AI / RAG architect
role) for the Air-Gapped Predictive Copilot for Secure MPLS Operations project. It
complements the existing `topology.clab.yml` and `daemons` artefacts at the repo root,
which are owned by the mimo v2.5 network-engineering track (Phase 1 — simulation).

Everything under `m3/` is **self-contained, has zero outbound network dependencies at
runtime, and is designed to be packaged into the air-gapped bundle together with the
quantized LLM weights.**

## Scope covered

| Task | Project Phase | Folder | What it delivers |
|------|---------------|--------|------------------|
| 1 | Phase 4 — Offline LLM Deployment | `m3/rag/` | Offline RAG pipeline (Chroma/Qdrant), chunking strategy, ingestion + query scripts, sample corpus |
| 2 | Phase 5 — Copilot Integration | `m3/prompts/` | Anti-hallucination system prompt, strict JSON response schema, schema validator, few-shot exemplars, operator-Q templates |
| 3 | Phase 6 — Scenario Validation | `m3/playbooks/` | Four fault-injection playbooks (congestion, BGP flap, MPLS underlay, policy drift), runner + metrics script |
| (support) | Phase 1/2 — Simulation + Telemetry | `m3/telemetry/` | SNMP collector, NetFlow collector, Telegraf + Prometheus configs that map the simulated devices into the local Prometheus instance |
| (overarching) | Phases 1–6 | `m3/ARCHITECTURE.md` | End-to-end architecture tying every component together, with the exact data-flow from packet → alert → operator |

## How to read this subtree

1. Start at `m3/ARCHITECTURE.md` — the single-page system diagram and component
   responsibilities.
2. Read `m3/rag/CHUNKING_STRATEGY.md` to understand *why* the RAG chunker is shaped
   the way it is.
3. Read `m3/prompts/SYSTEM_PROMPT.md` to understand the anti-hallucination contract
   placed on the offline LLM.
4. Read `m3/playbooks/README.md` for the four validation scenarios and how they are
   executed end-to-end.
5. The `m3/telemetry/` folder is the bridge that the mimo v2.5 simulation track must
   bind into the existing Containerlab topology to feed this Copilot.

## Design principles followed in every file

- **Air-gap first.** No HTTP clients to external hosts, no `requests.get` to model
  registries, no cloud SDKs. All inference, embedding, and storage are local.
- **Determinism.** Embedding model + chunk boundaries are pinned by content hash; the
  same input corpus always produces the same index.
- **No silent fallbacks.** If the LLM cannot answer from retrieval, it must return
  `answer_grounded: false` with a `missing_context` array — never invent.
- **Schema-bound output.** The LLM is constrained to emit a single JSON object
  matching `prompts/response_schema.json`. Free-form text outside the JSON block is
  rejected by the validator.
- **Operator-first language.** The three operator questions (Q1 What will fail, Q2
  Why, Q3 What action) are first-class fields in the schema, not buried prose.

## How to run the tests

The permanent test suite lives at `m3/tests/test_all.py`. It runs 48 tests
covering every module end-to-end and exits non-zero on any failure.

```bash
cd /path/to/air-gapped-noc-copilot
python -m m3.tests.test_all
```

The tests use two environment variables to allow running on a development
machine without breaking the air-gap guarantees:

- `NOC_COPILOT_FAKE_EMBEDDER=1` — uses a deterministic hash-based embedder
  instead of the real `bge-small-en-v1.5` model. The pipeline logic is
  identical; only the vectors differ. Do NOT set this in production.
- `NOC_COPILOT_SKIP_AIRGAP_CHECK=1` — bypasses the DNS check to
  `huggingface.co` in the startup guard. Only set during development.

The RAG tests use a real Chroma vector store (the `chromadb` package); the
SNMP and NetFlow tests use a hash-based fake embedder where applicable.

## What was reviewed and fixed

The review-and-fix pass (see conversation history) caught and fixed:

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `m3/rag/chunker.py` | `TopologyChunker` did not strip YAML frontmatter, causing `yaml.safe_load` to fail on multi-doc YAML | Added `_strip_frontmatter_yaml` helper and use it in `chunk_file` |
| 2 | `m3/rag/chunker.py` | `rstrip("s")` produced wrong singulars (`"policies"` -> `"policie"`) | Added explicit `_SINGULAR_ENTITY` mapping |
| 3 | `m3/rag/chunker.py` | Header templates used key names that did not match the chunker metadata | Updated templates to use actual keys (`runbook_id`, `incident_id`, `affected_sites`, `root_cause_class`, `last_reviewed`, `topology_version`) |
| 4 | `m3/rag/index_backend.py` | ChromaDB rejects empty-list metadata values; the runbook chunker produces `steps=[]` | Added `_sanitise_metadata` that flattens lists to CSV and drops None/empty |
| 5 | `m3/rag/embedder.py` | Hard-fails if the model is missing — fine for production, but no way to test without bundling 80 MB of weights | Added `NOC_COPILOT_FAKE_EMBEDDER=1` mode with a deterministic hash embedder |
| 6 | `m3/rag/rag_query.py` | `BM25Okapi` crashes with ZeroDivisionError on empty docs | Guarded the function to return an empty array when no docs |
| 7 | `m3/rag/rag_query.py` | `_load_index` ignored `cfg["retriever"]["chroma_path"]` and hardcoded `index_root / "chroma"` | Use the configured path, fall back to `index_root / "chroma"` if not set |
| 8 | `m3/rag/rag_ingest.py` and `m3/playbooks/fault_injection_runner.py` | Air-gap DNS check blocks dev machines with internet | Honour `NOC_COPILOT_SKIP_AIRGAP_CHECK=1` |
| 9 | `m3/prompts/schema_validator.py` | `_extract_json` fence regex `\{.*?\}` matched the innermost JSON, not the outer one for nested objects | Replaced with `json.JSONDecoder().raw_decode` and added a prose-prefix fallback |
| 10 | `m3/prompts/schema_validator.py` | Truncation was dead code (schema rejected over-max strings before the truncation ran) | Moved pre-truncation to before schema validation; added `TRUNCATED` to the schema warning enum |
| 11 | `m3/playbooks/fault_injection_runner.py` | `urllib.parse` was used but not imported (NameError) | Added the import |
| 12 | `m3/playbooks/fault_injection_runner.py` | `index_path` was hard-coded to `args.config.parent / "index"`; no way to override | Added `--index` CLI arg |
| 13 | `m3/telemetry/collectors/snmp_collector.py` | `_coerce` returned the bit width ("32") instead of the Counter value | Use `m.group(m.lastindex)` to get the LAST capturing group |
| 14 | `m3/telemetry/collectors/snmp_collector.py` | Gauge regex `(\d+)` matched the first digits ("32" in "Gauge32") | Updated to match digits after the type prefix |
| 15 | `m3/telemetry/collectors/netflow_collector.py` | NetFlow v5 record format had 11 chars but tried to unpack 15 names | Used the correct 20-field format per RFC 3954 |
| 16 | `m3/telemetry/collectors/netflow_collector.py` | NetFlow v5 header format had 12 chars but 9 names; v9 header had `H` for source_id instead of `I` | Used correct 24-byte header format `>HHIIIIBBH`; fixed v9 to `>HHIIII` |
| 17 | `m3/telemetry/collectors/netflow_collector.py` | `_ip4` was called with an int (4-byte packed IPv4) instead of 4 raw bytes | Added `_ip4_from_int` helper that uses `socket.inet_ntoa` |
| 18 | `m3/prompts/response_schema.json` | `warnings` enum did not include `"TRUNCATED"` | Added to the enum |
