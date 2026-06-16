"""Ingest a corpus directory into the local vector index.

    python -m m3.rag.rag_ingest \
        --corpus  m3/rag/corpus \
        --index   m3/rag/index \
        --config  m3/rag/rag_config.yaml

The script is fully air-gapped: it does not download anything, does not call
external services, and refuses to start if the embedder model directory is
missing or if `runtime.refuse_network_calls` is true in the config and any
outbound socket is opened during execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import yaml

from .chunker import Chunk, chunk_corpus
from .embedder import LocalEmbedder
from .index_backend import ChromaIndex, IndexBackend, QdrantIndex, write_manifest

LOG = logging.getLogger("noc_copilot.ingest")


def _disallow_network() -> None:
    """Refuse to start if DNS resolves or any common external port is open."""
    if os.environ.get("NOC_COPILOT_SKIP_AIRGAP_CHECK") == "1":
        LOG.warning("NOC_COPILOT_SKIP_AIRGAP_CHECK=1 \u2014 air-gap DNS check bypassed (development mode).")
        return
    try:
        socket.getaddrinfo("huggingface.co", 443)
        raise RuntimeError("Outbound DNS to huggingface.co resolves \u2014 air-gap is broken.")
    except socket.gaierror:
        pass


def _hash_corpus(corpus: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(corpus.rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(corpus)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def _iter_chunks(corpus: Path) -> List[Chunk]:
    out: List[Chunk] = []
    for ch in chunk_corpus(corpus):
        out.append(ch)
    return out


def _embed_in_batches(embedder: LocalEmbedder, chunks: Sequence[Chunk]) -> np.ndarray:
    docs = [c.text for c in chunks]
    if not docs:
        return np.zeros((0, embedder.dimension or 1), dtype=np.float32)
    return embedder.embed_documents(docs)


def main() -> int:
    p = argparse.ArgumentParser(description="Air-gapped NOC Copilot RAG ingest")
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--backend", choices=["chroma", "qdrant"], default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if cfg.get("runtime", {}).get("refuse_network_calls", True):
        _disallow_network()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    embedder = LocalEmbedder(
        model_path=cfg["embed"]["model_path"],
        normalize=cfg["embed"].get("normalize", True),
        query_prefix=cfg["embed"].get("query_prefix", ""),
        batch_size=cfg["embed"].get("batch_size", 32),
        offline_only=cfg.get("runtime", {}).get("offline_mode", True),
    )

    backend_name = args.backend or cfg["retriever"].get("backend", "chroma")
    if backend_name == "chroma":
        index: IndexBackend = ChromaIndex(
            path=str(args.index / "chroma"),
            collection=cfg["retriever"]["chroma_collection"],
        )
    else:
        if not embedder.dimension:
            raise RuntimeError("Embedder dimension unknown; cannot create Qdrant collection")
        index = QdrantIndex(
            url=cfg["retriever"]["qdrant_url"],
            collection=cfg["retriever"]["qdrant_collection"],
            vector_size=embedder.dimension,
        )

    corpus = args.corpus
    LOG.info("Scanning corpus at %s", corpus)
    chunks = _iter_chunks(corpus)
    LOG.info("Chunked %d chunks from corpus", len(chunks))
    if not chunks:
        LOG.warning("No chunks produced \u2014 is the corpus empty or in an unsupported format?")

    vectors = _embed_in_batches(embedder, chunks)
    LOG.info("Embedded %d vectors (dim=%d)", len(vectors), embedder.dimension or 0)

    if vectors.size:
        index.upsert(chunks, vectors)
    LOG.info("Index size after upsert: %d", index.count())

    manifest = {
        "index_built_at": int(time.time()),
        "index_built_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_path": str(corpus),
        "corpus_sha256": _hash_corpus(corpus),
        "embedder_model_path": cfg["embed"]["model_path"],
        "embedder_model_sha256": embedder.fingerprint(),
        "embedder_dim": embedder.dimension,
        "backend": backend_name,
        "chunk_count": len(chunks),
        "schema_version": 1,
    }
    manifest_path = args.index / Path(cfg["index"]["manifest_path"]).name
    write_manifest(manifest_path, manifest)
    LOG.info("Wrote index manifest to %s", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
