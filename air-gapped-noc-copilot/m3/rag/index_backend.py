"""RAG index backends.

Two implementations behind a small interface:

  - ChromaIndex  persistent embedded mode (default, single-host)
  - QdrantIndex  local Qdrant server or in-process

Both backends accept the same `Chunk` objects and store the chunk text in
`document` plus the metadata in `metadata`. The vector is stored under `embedding`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .chunker import Chunk

LOG = logging.getLogger("noc_copilot.index")


def _sanitise_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """ChromaDB and Qdrant both reject empty-list metadata values. Filter
    them out and coerce every value to a primitive (str / int / float / bool)."""
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            if not v:
                continue
            out[k] = ",".join(str(x) for x in v)
            continue
        if isinstance(v, dict):
            out[k] = json.dumps(v, sort_keys=True)
            continue
        if isinstance(v, bool):
            out[k] = v
            continue
        if isinstance(v, (int, float, str)):
            out[k] = v
            continue
        out[k] = str(v)
    return out


class IndexBackend:
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...
    def query(self, vector: np.ndarray, top_k: int, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]: ...
    def count(self) -> int: ...
    def delete_where(self, where: Dict[str, Any]) -> int: ...


class ChromaIndex(IndexBackend):
    def __init__(self, path: str, collection: str):
        import chromadb
        from chromadb.config import Settings
        self._client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False, allow_reset=False))
        self._coll = self._client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [_sanitise_metadata(c.metadata) for c in chunks]
        embs = vectors.tolist()
        self._coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

    def query(self, vector: np.ndarray, top_k: int, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        kw: Dict[str, Any] = {"query_embeddings": vector.tolist(), "n_results": top_k}
        if where:
            kw["where"] = where
        res = self._coll.query(**kw)
        out: List[Dict[str, Any]] = []
        for i, doc in enumerate(res.get("documents", [[]])[0]):
            out.append({
                "id": res["ids"][0][i],
                "text": doc,
                "metadata": res["metadatas"][0][i],
                "score": 1.0 - float(res["distances"][0][i]),
            })
        return out

    def count(self) -> int:
        return self._coll.count()

    def delete_where(self, where: Dict[str, Any]) -> int:
        before = self._coll.count()
        self._coll.delete(where=where)
        return before - self._coll.count()


class QdrantIndex(IndexBackend):
    def __init__(self, url: str, collection: str, vector_size: int):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels
        self._client = QdrantClient(url=url, timeout=30.0, prefer_grpc=False)
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
            )
        self._coll = collection

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        from qdrant_client.http import models as qmodels
        points = [
            qmodels.PointStruct(
                id=c.chunk_id,
                vector=vectors[i].tolist(),
                payload={"text": c.text, **_sanitise_metadata(c.metadata)},
            )
            for i, c in enumerate(chunks)
        ]
        self._client.upsert(collection_name=self._coll, points=points, wait=True)

    def query(self, vector: np.ndarray, top_k: int, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        from qdrant_client.http import models as qmodels
        flt = None
        if where:
            must = [qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v)) for k, v in where.items()]
            flt = qmodels.Filter(must=must)
        res = self._client.search(collection_name=self._coll, query_vector=vector.tolist(), limit=top_k, query_filter=flt)
        return [{"id": str(r.id), "text": r.payload.get("text", ""), "metadata": {k: v for k, v in r.payload.items() if k != "text"}, "score": float(r.score)} for r in res]

    def count(self) -> int:
        return int(self._client.count(self._coll).count)

    def delete_where(self, where: Dict[str, Any]) -> int:
        from qdrant_client.http import models as qmodels
        before = self.count()
        flt = qmodels.Filter(must=[qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v)) for k, v in where.items()])
        self._client.delete(collection_name=self._coll, points_selector=qmodels.FilterSelector(filter=flt))
        return before - self.count()


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
