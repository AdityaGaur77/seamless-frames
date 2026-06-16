"""Query the RAG index from the NOC Copilot runtime.

    python -m m3.rag.rag_query \
        --config m3/rag/rag_config.yaml \
        --index  m3/rag/index \
        --query  "BGP neighbor stuck in Active on pe-hub-east-1" \
        --top-k  8

Returns a JSON envelope to stdout (or a file) with the structured evidence
that the offline LLM will consume as the `RETRIEVED_EVIDENCE` block of its
prompt. The envelope includes BM25 + dense hybrid scores and the
grounding metadata that the response-schema validator will check against.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from .embedder import LocalEmbedder
from .index_backend import ChromaIndex, IndexBackend, QdrantIndex
from .chunker import Chunk

LOG = logging.getLogger("noc_copilot.query")

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./-]+")
_SECTION_HEADERS = re.compile(r"^#+\s+.*$", re.MULTILINE)


def _bm25_scores(query: str, docs: List[str]) -> np.ndarray:
    from rank_bm25 import BM25Okapi
    if not docs:
        return np.zeros((0,), dtype=np.float32)
    tokenised_corpus = [re.findall(_TOKEN_RE, d.lower()) for d in docs]
    bm = BM25Okapi(tokenised_corpus)
    return np.asarray(bm.get_scores(re.findall(_TOKEN_RE, query.lower())), dtype=np.float32)


def _normalise(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _load_index(cfg: Dict[str, Any], index_root: Path, embedder: LocalEmbedder) -> IndexBackend:
    backend = cfg["retriever"].get("backend", "chroma")
    if backend == "chroma":
        chroma_path = cfg["retriever"].get("chroma_path")
        if chroma_path:
            path = chroma_path
        else:
            path = str(index_root / "chroma")
        return ChromaIndex(
            path=path,
            collection=cfg["retriever"]["chroma_collection"],
        )
    if not embedder.dimension:
        raise RuntimeError("Embedder dimension unknown")
    return QdrantIndex(
        url=cfg["retriever"]["qdrant_url"],
        collection=cfg["retriever"]["qdrant_collection"],
        vector_size=embedder.dimension,
    )


def _load_manifest(index_root: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    p = index_root / Path(cfg["index"]["manifest_path"]).name
    if not p.exists():
        return {"index_unhealthy": True, "reason": "manifest_missing"}
    return json.loads(p.read_text(encoding="utf-8"))


def _build_where(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not filters:
        return None
    return {k: v for k, v in filters.items() if v is not None}


def _format_evidence(hits: List[Dict[str, Any]], budget_tokens: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    used = 0
    for h in hits:
        body = h["text"]
        meta = h.get("metadata", {})
        meta_brief = {k: meta.get(k) for k in ("doc_type", "runbook_id", "incident_id", "device", "site", "protocol", "section", "severity", "date", "root_cause_class", "topology_version") if k in meta}
        approx_tokens = max(1, len(body.split()))
        if used + approx_tokens > budget_tokens:
            remaining = max(0, budget_tokens - used)
            if remaining < 40:
                break
            truncated = " ".join(body.split()[:remaining])
            out.append({"chunk_id": h["id"], "metadata": meta_brief, "score": round(h["score"], 4), "text": truncated, "truncated": True})
            used += remaining
        else:
            out.append({"chunk_id": h["id"], "metadata": meta_brief, "score": round(h["score"], 4), "text": body, "truncated": False})
            used += approx_tokens
    return out


def query(cfg_path: Path, index_root: Path, query_text: str, top_k: int, filters: Optional[Dict[str, Any]] = None, budget_tokens: Optional[int] = None) -> Dict[str, Any]:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("runtime", {}).get("refuse_network_calls", True):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    embedder = LocalEmbedder(
        model_path=cfg["embed"]["model_path"],
        normalize=cfg["embed"].get("normalize", True),
        query_prefix=cfg["embed"].get("query_prefix", ""),
        batch_size=cfg["embed"].get("batch_size", 32),
        offline_only=cfg.get("runtime", {}).get("offline_mode", True),
    )
    index = _load_index(cfg, index_root, embedder)
    manifest = _load_manifest(index_root, cfg)

    qvec = embedder.embed_queries([query_text])[0]
    where = _build_where(filters)
    dense_hits = index.query(qvec, top_k=top_k * 2, where=where)
    docs = [h["text"] for h in dense_hits]
    dense_scores = np.asarray([h["score"] for h in dense_hits], dtype=np.float32)
    bm = _bm25_scores(query_text, docs)
    dense_n = _normalise(dense_scores)
    bm_n = _normalise(bm)
    w_dense = cfg["retriever"].get("hybrid_weight_dense", 0.7)
    w_lex = cfg["retriever"].get("hybrid_weight_lex", 0.3)
    final = w_dense * dense_n + w_lex * bm_n
    order = np.argsort(-final)[:top_k]
    ranked = []
    for i in order:
        h = dense_hits[int(i)]
        h["score"] = float(final[int(i)])
        ranked.append(h)

    budget = budget_tokens or cfg["retriever"].get("context_token_budget", 3500)
    evidence = _format_evidence(ranked, budget)

    return {
        "query": query_text,
        "filters": filters or {},
        "manifest": manifest,
        "evidence": evidence,
        "index_unhealthy": manifest.get("index_unhealthy", False),
        "config": {"backend": cfg["retriever"].get("backend", "chroma"), "embedder_fingerprint": embedder.fingerprint()},
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Air-gapped NOC Copilot RAG query")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--query", required=True)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--site", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--protocol", default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

    filters = {"site": args.site, "device": args.device, "protocol": args.protocol}
    filters = {k: v for k, v in filters.items() if v}
    res = query(args.config, args.index, args.query, args.top_k, filters)
    payload = json.dumps(res, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
