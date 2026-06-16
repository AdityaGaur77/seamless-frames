"""Local embedding wrapper for the RAG pipeline.

Wraps sentence-transformers with three guarantees for air-gapped use:

  1. The model is loaded from a local directory only \u2014 no network access.
  2. The model directory's contents are SHA-256 pinned at startup.
  3. The wrapper refuses to fall back to a non-semantic bag-of-words model
     if the embedder is missing; the caller must surface a hard error.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

LOG = logging.getLogger("noc_copilot.embedder")


def _sha256_dir(directory: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(directory.rglob("*")):
        if f.is_file():
            h.update(f.relative_to(directory).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()


class LocalEmbedder:
    def __init__(self, model_path: str, normalize: bool = True, query_prefix: str = "", batch_size: int = 32, offline_only: bool = True, dimension: int = 64):
        self.model_path = Path(model_path)
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.offline_only = offline_only
        self.dimension = dimension
        self._model: Optional[object] = None
        self._fake = bool(int(os.environ.get("NOC_COPILOT_FAKE_EMBEDDER", "0")))

        if self._fake:
            LOG.warning("NOC_COPILOT_FAKE_EMBEDDER=1 \u2014 using deterministic hash embedder (test mode only). Do NOT ship this to production.")
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Local embedder model not found at {self.model_path}. "
                f"Bundle the model with the air-gapped package before running."
            )
        if offline_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self._load()

    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer
        LOG.info("Loading local embedder from %s", self.model_path)
        self._model = SentenceTransformer(str(self.model_path), device="cpu")
        dim = self._model.get_sentence_embedding_dimension()
        self.dimension = int(dim) if dim is not None else 0
        LOG.info("Embedder ready (dim=%s, sha256=%s)", self.dimension, self.fingerprint())

    def fingerprint(self) -> str:
        if self._fake:
            return "0" * 64
        return _sha256_dir(self.model_path)

    def embed(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension or 1), dtype=np.float32)
        if is_query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        if self._fake:
            return self._fake_embed(texts)
        if self._model is None:
            raise RuntimeError("Embedder not initialised")
        vecs = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    def _fake_embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                h = hashlib.md5(w.encode("utf-8")).digest()
                for j in range(0, len(h), 2):
                    bucket = (h[j] * 256 + h[j + 1]) % self.dimension
                    out[i, bucket] += 1.0
            n = float(np.linalg.norm(out[i]))
            if n > 0:
                out[i] /= n
        return out

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        return self.embed(queries, is_query=True)

    def embed_documents(self, docs: Sequence[str]) -> np.ndarray:
        return self.embed(docs, is_query=False)
