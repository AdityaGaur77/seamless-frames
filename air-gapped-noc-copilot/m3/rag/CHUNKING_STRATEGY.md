# RAG Chunking & Indexing Strategy

> Phase 4 — Offline LLM Deployment
> Authored by MiniMax M3 (Lead AI Architect role)

This document is the single source of truth for **how** the offline RAG pipeline
chunks, enriches, embeds, and indexes the three corpora used by the NOC Copilot:

1. **Topology maps** — graph + YAML describing devices, links, VPNs, tunnels.
2. **Runbooks** — operator procedures (BGP, MPLS, SD-WAN, QoS, etc.).
3. **Incident history** — past postmortems, RCAs, alert→action mappings.

The strategy is tuned for an air-gapped deployment that must:

- Run on a single on-prem box with limited RAM (no cloud vector DB).
- Provide **high-precision retrieval** (operators cannot tolerate hallucinated
  remediation steps).
- Stay **deterministic** — the same source files must always produce the same
  index, so the LLM's grounding evidence is auditable.
- Expose **structured filters** so the retriever can be constrained by site, role,
  time window, or signal class.

---

## 1. Storage backend — Chroma (default) vs. Qdrant (optional)

| Aspect | Chroma (`chroma` 0.5.x) | Qdrant (`qdrant-client` 1.7.x) |
|---|---|---|
| Mode | Embedded (DuckDB+Parquet) — no server | Local server (`qdrant` binary) or in-process `QdrantClient(":memory:")` |
| Air-gap install | `pip install chromadb` — wheels only, no post-install network | `pip install qdrant-client` + local `qdrant` binary tarball |
| Metadata filtering | Yes (where-clauses) | Yes (richer payload indexes) |
| Hybrid (BM25 + dense) | Add `rank-bm25` separately | Built-in sparse vectors |
| Best for | Single-host, < 1 M chunks, low-ops | Multi-reader, larger corpora, hybrid search |

**Default choice: Chroma in persistent mode** at `m3/rag/index/chroma/`. The
`rag_ingest.py` script uses Chroma by default and exposes `--backend qdrant` for
sites that already run Qdrant internally. Both paths share the same chunker and
embedder.

## 2. Embedding model — fully local, quantized

We pin **`BAAI/bge-small-en-v1.5`** (33 M params, 384-dim, MIT license) as the
default dense embedder. The model is:

- Bundled inside the air-gapped bundle at `m3/rag/models/bge-small-en-v1.5/`
  (exported via `sentence-transformers` to ONNX, then optionally INT8 quantised
  with `optimum`).
- Loaded once at ingest time and again at query time — no HTTP call to
  HuggingFace, no telemetry, no update check.
- Replaced with `BAAI/bge-m3` only if the operator needs multilingual runbooks;
  the swap is a single config flag (`embed.model_name` in `rag_config.yaml`).

For hybrid retrieval, `rank-bm25` provides a lexical scorer on the same chunks;
we do not run a separate sparse embedding model.

## 3. Chunking strategy — three corpora, three chunkers

A single chunk size is wrong: topology tables are dense and tabular, runbooks
have clear procedure boundaries, and incident postmortems have a narrative
spanning many sections. We use **document-class-aware chunkers** selected by the
file path / metadata.

### 3.1 Topology corpus (`corpus/topology/`)

| Field | Value |
|---|---|
| Chunker | `TopologyChunker` — splits YAML/JSON into **one chunk per logical entity** (device, link, VPN, tunnel, prefix-list, route-map) and emits a natural-language summary line. |
| Target size | 80–200 tokens per chunk |
| Overlap | 0 — entities are atomic |
| Metadata keys | `device`, `role`, `site`, `asn`, `vrf`, `tunnel_id`, `link_id`, `topology_version` |
| Why | The retriever must answer "is `ce-branch-3` in VPN `cust-blue`?" with a single hit, not three adjacent chunks of YAML noise. |

### 3.2 Runbook corpus (`corpus/runbooks/`)

| Field | Value |
|---|---|
| Chunker | `RunbookChunker` — markdown-aware structural splitter that treats `#`, `##`, `###` headings as chunk boundaries; preserves numbered procedure steps inside a single chunk. |
| Target size | 350 tokens, **15 % overlap** with the previous chunk (only on the trailing paragraph) |
| Metadata keys | `runbook_id`, `title`, `protocol`, `severity`, `last_reviewed`, `approved_by`, `steps[]` |
| Why | Operators search for *symptom → procedure* mappings. Keeping all steps of "BGP neighbor stuck in Active" inside one chunk is critical; the 15 % overlap covers the case where a step spans a heading. |

### 3.3 Incident corpus (`corpus/incidents/`)

| Field | Value |
|---|---|
| Chunker | `IncidentChunker` — sections by `## Symptom`, `## Root cause`, `## Detection signal`, `## Remediation`, `## Prevention`; each becomes one chunk, **plus a final summary chunk** that re-states the RCA in 1–2 sentences for fast retrieval. |
| Target size | 300 tokens for detail chunks, 80 tokens for the summary chunk |
| Overlap | 0 (sections are already non-overlapping) |
| Metadata keys | `incident_id`, `date`, `affected_sites`, `signals[]`, `root_cause_class`, `mttr_minutes`, `linked_runbook_id` |
| Why | We want the retriever to be able to find "an incident where latency rose 40 % on hub-east before BGP converged" without forcing the LLM to read 3 000 tokens of narrative. |

## 4. Universal enrichment — every chunk gets a header

To drastically improve retrieval recall (and to give the offline LLM a
self-describing citation), every chunk is **prefixed with a natural-language
summary header** before embedding. The header is deterministic — it is built
from metadata, not generated by an LLM, so it adds zero inference cost and is
fully auditable.

Example for a runbook chunk:

```
[Runbook BGP-007 — Neighbor stuck in Active | protocol=BGP | severity=P2 |
 last_reviewed=2025-11-04 | approved_by=neteng-lead]
# Symptom
BGP session with neighbor X remains in Active state for > 3 minutes.
# Step 1
Verify TCP/179 reachability with `show tcp` ...
```

Example for a topology chunk:

```
[Topology entity | device=pe-hub-east-1 | role=PE | site=hub-east |
 vpn=cust-blue | topology_version=v3.2]
pe-hub-east-1 connects 4 branches and 1 DC over MPLS. VRF cust-blue
carries 12 prefixes; iBGP peer 10.255.0.1 (route-reflector).
```

The retriever embeds the **header + body** as one string; the **metadata
payload** is stored alongside and used for filtering and citation.

## 5. Determinism & re-indexing

- Every chunk gets a stable `chunk_id = sha256(corpus + relative_path + byte_offset + content_hash)`.
- Re-running ingest produces an **idempotent** index: existing chunk_ids are
  updated in place, removed files are pruned, new ones appended.
- The entire index state is reproducible from `corpus/` + `rag_config.yaml` —
  there is no hidden state. This is essential for audit in regulated
  environments.
- The ingest script emits `index_manifest.json` containing model hash, chunk
  count, corpus hash, and timestamp. The validator in `prompts/` refuses to
  answer if the index is older than `max_index_age_days` (default 30) unless
  the operator explicitly acknowledges staleness.

## 6. Retrieval-time strategy

- **Hybrid score** = `0.7 * cosine_dense + 0.3 * bm25_lexical`, both
  normalised to `[0, 1]`.
- **Metadata pre-filtering** before scoring when the LLM's structured query
  specifies `site`, `device`, `protocol`, or `time_window` — this is how we
  answer "what is the runbook for **this** device" without leaking
  cross-site runbooks.
- **Top-k = 8** for first-pass retrieval, then **re-rank with a cross-encoder**
  only if a cross-encoder model is bundled (off by default — adds latency).
- **Context budget** for the LLM prompt is **3 500 tokens** of retrieved
  evidence, leaving the rest of the 8 K context for the system prompt,
  operator question, structured alert payload, and the JSON schema
  instructions.

## 7. Corpus preparation pipeline (offline)

```
corpus/  ──►  scan + classify  ──►  per-class chunker  ──►  enrich with header
                                                                  │
                                                                  ▼
                                                          embed (bge-small)
                                                                  │
                                                                  ▼
                                              persist to Chroma / Qdrant
                                                                  │
                                                                  ▼
                                                    write index_manifest.json
```

Everything runs from `m3/rag/rag_ingest.py --corpus m3/rag/corpus
--index m3/rag/index --config m3/rag/rag_config.yaml`. The script is fully
air-gapped: it does not call out to download anything, even for first-time
setup. Operators place the embedding model directory on disk before running.

## 8. Failure modes & guardrails

| Failure | Detection | Behaviour |
|---|---|---|
| Embedding model missing | `rag_ingest.py` pre-flight check | Abort with actionable error; do **not** silently fall back to a non-semantic bag-of-words retriever. |
| Index corrupt / partial | Manifest SHA mismatch at query time | Refuse to serve answers; surface `index_unhealthy: true` in the structured response. |
| No chunk above similarity threshold | Empty result set | Return `answer_grounded: false`; the prompt forces the LLM to say "I do not have a grounded runbook for this alert in the current corpus" rather than guess. |
| Stale index | `index_age > max_index_age_days` | Inject a `WARN_STALE_INDEX` block in the prompt; LLM must add a `warnings` field to its response. |
| Query references a device not in topology | Topology filter returns 0 | LLM is told the device is unknown and must not invent properties. |
