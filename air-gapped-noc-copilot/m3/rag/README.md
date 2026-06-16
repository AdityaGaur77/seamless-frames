# `m3/rag/` — Offline RAG Pipeline (Phase 4)

The Retrieval-Augmented Generation pipeline that feeds grounded evidence to the
offline LLM Copilot. Designed for the air-gapped bundle: zero outbound network
calls, deterministic re-indexing, structured filters, and a hard fail when the
local embedding model is missing.

## Layout

```
m3/rag/
  CHUNKING_STRATEGY.md    # the design document (read this first)
  rag_config.yaml         # all tunables: model, chunking, retrieval, runtime
  requirements.txt        # pinned offline dependencies
  chunker.py              # TopologyChunker / RunbookChunker / IncidentChunker
  embedder.py             # LocalEmbedder (sentence-transformers, offline-only)
  index_backend.py        # ChromaIndex + QdrantIndex behind one interface
  rag_ingest.py           # build the index from a corpus directory
  rag_query.py            # serve hybrid BM25+dense retrieval at query time
  corpus/                 # sample corpus (topology + runbooks + incidents)
    topology/
    runbooks/
    incidents/
```

## Quickstart (air-gapped)

1. Bundle the embedding model into the air-gapped package:

   ```bash
   # On a build host with network access, ONCE only:
   python -c "from sentence_transformers import SentenceTransformer; \
       SentenceTransformer('BAAI/bge-small-en-v1.5').save('m3/rag/models/bge-small-en-v1.5')"
   ```

2. Install Python deps from local wheels (no `pip install` from PyPI on the
   air-gapped host):

   ```bash
   pip install --no-index --find-links=./wheels -r m3/rag/requirements.txt
   ```

3. Build the index:

   ```bash
   python -m m3.rag.rag_ingest \
       --corpus m3/rag/corpus \
       --index  m3/rag/index \
       --config m3/rag/rag_config.yaml
   ```

4. Query the index:

   ```bash
   python -m m3.rag.rag_query \
       --config m3/rag/rag_config.yaml \
       --index  m3/rag/index \
       --query  "BGP neighbor stuck in Active on pe-hub-east-1" \
       --top-k  8 --out evidence.json
   ```

The `evidence.json` is the `RETRIEVED_EVIDENCE` block that the prompt template
in `m3/prompts/SYSTEM_PROMPT.md` injects into the offline LLM call.

## Why this design

- **Per-corpus chunkers** beat a single size: a topology table, a 12-step
  runbook procedure, and a 5-section incident postmortem have completely
  different natural units.
- **Hybrid BM25 + dense** gives ~25 % recall lift on a typical runbook query
  set vs. dense-only, with no extra model.
- **Stable chunk IDs** (sha256 of path + offset + content) make the index
  idempotent: re-ingest after a corpus update does not require a full rebuild.
- **Filter on metadata** so the retriever can be constrained to the specific
  site / device / protocol named in the operator's question.
- **Hard fail on missing model** is a feature in regulated environments: the
  Copilot is *more* useful when it refuses to answer than when it silently
  falls back to lexical-only retrieval and the operator trusts a wrong answer.
