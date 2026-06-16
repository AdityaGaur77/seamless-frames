"""End-to-end RAG pipeline test using a deterministic fake embedder.

We don't need a real model for testing the pipeline mechanics. The fake
embedder produces a 64-dim vector where each dimension is a hash bucket
of the text. Semantically related texts will share hash buckets and
end up with similar vectors; unrelated texts will be orthogonal.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np

ROOT = Path(r'C:\Users\adity\Downloads\real-port\test\air-gapped-noc-copilot')
sys.path.insert(0, str(ROOT))

from m3.rag.chunker import chunk_corpus, Chunk
from m3.rag.embedder import LocalEmbedder
from m3.rag.index_backend import ChromaIndex, write_manifest
from m3.rag.rag_query import _bm25_scores, _normalise


DIM = 64


class FakeEmbedder:
    def __init__(self, dim: int = DIM):
        self.dimension = dim

    def fingerprint(self) -> str:
        return "0" * 64

    def embed(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, t in enumerate(texts):
            words = t.lower().split()
            for w in words:
                h = hashlib.md5(w.encode("utf-8")).digest()
                for j in range(0, len(h), 2):
                    bucket = (h[j] * 256 + h[j + 1]) % self.dimension
                    out[i, bucket] += 1.0
            norm = float(np.linalg.norm(out[i]))
            if norm > 0:
                out[i] /= norm
        return out

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        return self.embed(queries, is_query=True)

    def embed_documents(self, docs: Sequence[str]) -> np.ndarray:
        return self.embed(docs, is_query=False)


def monkeypatch_embedder():
    import m3.rag.embedder as emb
    orig = emb.LocalEmbedder
    class Stub(LocalEmbedder):
        def __init__(self, model_path, **kwargs):
            self.model_path = Path(model_path)
            self.normalize = True
            self.query_prefix = ""
            self.batch_size = 32
            self.offline_only = True
            self._model = object()
            self.dimension = DIM
            self._fake = FakeEmbedder(DIM)
        def fingerprint(self):
            return "0" * 64
        def embed(self, texts, is_query=False):
            return self._fake.embed(texts, is_query=is_query)
        def embed_queries(self, qs):
            return self._fake.embed_queries(qs)
        def embed_documents(self, ds):
            return self._fake.embed_documents(ds)
    emb.LocalEmbedder = Stub
    return orig


def test_chunk_corpus_returns_expected_types():
    corpus = ROOT / "m3" / "rag" / "corpus"
    chunks = list(chunk_corpus(corpus))
    assert len(chunks) > 0, "no chunks produced"
    by_kind = {}
    for c in chunks:
        k = c.metadata.get("doc_type")
        by_kind[k] = by_kind.get(k, 0) + 1
    assert by_kind.get("topology", 0) > 0
    assert by_kind.get("runbook", 0) > 0
    assert by_kind.get("incident_summary", 0) > 0
    assert by_kind.get("incident_detail", 0) > 0
    print(f"PASS  chunker produced {len(chunks)} chunks: {by_kind}")


def test_chunker_stable_ids():
    corpus = ROOT / "m3" / "rag" / "corpus"
    c1 = list(chunk_corpus(corpus))
    c2 = list(chunk_corpus(corpus))
    ids1 = sorted(c.chunk_id for c in c1)
    ids2 = sorted(c.chunk_id for c in c2)
    assert ids1 == ids2, "chunk ids not deterministic"
    assert len(ids1) == len(set(ids1)), "duplicate chunk ids"
    print(f"PASS  chunker produces {len(ids1)} unique stable ids")


def test_chunker_headers_present():
    corpus = ROOT / "m3" / "rag" / "corpus"
    chunks = list(chunk_corpus(corpus))
    for c in chunks:
        first_line = c.text.split("\n", 1)[0]
        assert first_line.startswith("["), f"missing header in chunk: {c.text[:80]}"
    print(f"PASS  all {len(chunks)} chunks have deterministic headers")


def test_chunker_topology_singular_keys():
    corpus = ROOT / "m3" / "rag" / "corpus"
    chunks = list(chunk_corpus(corpus))
    for c in chunks:
        if c.metadata.get("entity_key") == "policies":
            text = c.text.lower()
            assert "policy:" in text or "policies:" in text
            assert "policie:" not in text
            print(f"PASS  policies entity singular: {c.chunk_id}")
            return
    assert False, "no policies chunk found"


def test_index_upsert_and_query_with_fake_embedder():
    orig = monkeypatch_embedder()
    import gc
    tmp = tempfile.mkdtemp(prefix="rag_test_")
    try:
        index = ChromaIndex(path=tmp, collection="test_v1")
        corpus = ROOT / "m3" / "rag" / "corpus"
        chunks = list(chunk_corpus(corpus))
        embedder = FakeEmbedder(DIM)
        vectors = embedder.embed_documents([c.text for c in chunks])
        assert vectors.shape == (len(chunks), DIM)
        index.upsert(chunks, vectors)
        assert index.count() == len(chunks)
        qvec = embedder.embed_queries(["BGP neighbor stuck in Active"])[0]
        results = index.query(qvec, top_k=5)
        assert len(results) == 5
        runbook_hits = [r for r in results if r["metadata"].get("doc_type") == "runbook"]
        assert len(runbook_hits) > 0, "expected at least one runbook hit for BGP query"
        print(f"PASS  ChromaIndex upsert+query: {index.count()} chunks, {len(runbook_hits)} runbook hits for BGP query")
        del index
        gc.collect()
    finally:
        import m3.rag.embedder as emb
        emb.LocalEmbedder = orig
        time.sleep(0.2)
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass


def test_hybrid_scoring_basic():
    docs = [
        "BGP neighbor stuck in Active state on pe-hub-east-1",
        "MPLS underlay loss with CRC errors on fibre",
        "SD-WAN tunnel jitter rising on sdwan-nyc-sfo",
        "Random unrelated text about cooking recipes and gardening tips",
    ]
    query = "BGP neighbor stuck in Active"
    dense = np.asarray([0.9, 0.1, 0.05, 0.0], dtype=np.float32)
    bm = _bm25_scores(query, docs)
    bm_n = _normalise(bm)
    dense_n = _normalise(dense)
    final = 0.7 * dense_n + 0.3 * bm_n
    order = np.argsort(-final)
    assert order[0] == 0, f"expected doc 0 (BGP) to rank first, got {order}"
    print(f"PASS  hybrid scoring: doc 0 (BGP) ranked first; final scores = {final.tolist()}")


def test_bm25_keyword_match():
    docs = [
        "BGP neighbor stuck in Active on pe-hub-east-1",
        "policy drift on controller for qos-cust-blue",
        "OSPF area mismatch on pe-dc-1 requires LSDB resync",
        "MPLS underlay loss with CRC errors on fibre",
        "NetFlow export stopped from ce-branch-3",
        "LDP session holdtime decreasing on hub-east uplinks",
        "interface ifInErrors rising on pe-hub-east-1 eth1",
        "voice packet drops on the sdwan-nyc-sfo IPSec tunnel",
    ]
    query = "BGP neighbor stuck Active"
    bm = _bm25_scores(query, docs)
    top = int(np.argmax(bm))
    assert top == 0, f"expected doc 0 (BGP neighbor) to rank first; got doc {top} with scores {bm.tolist()}"
    print(f"PASS  BM25 keyword match: doc 0 ranked first (BGP); scores={bm.tolist()}")


def test_manifest_write_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = Path(tmp) / "index_manifest.json"
        write_manifest(manifest_path, {
            "schema_version": 1,
            "chunk_count": 25,
            "embedder_dim": 384,
        })
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert loaded["schema_version"] == 1
        assert loaded["chunk_count"] == 25
        print("PASS  manifest write+load roundtrip")


if __name__ == "__main__":
    test_chunk_corpus_returns_expected_types()
    test_chunker_stable_ids()
    test_chunker_headers_present()
    test_chunker_topology_singular_keys()
    test_index_upsert_and_query_with_fake_embedder()
    test_hybrid_scoring_basic()
    test_bm25_keyword_match()
    test_manifest_write_and_load()
    print("\nAll RAG integration tests passed.")
