"""m3.rag: offline Retrieval-Augmented Generation pipeline for the NOC Copilot."""
from .chunker import Chunk, TopologyChunker, RunbookChunker, IncidentChunker, chunk_corpus
from .embedder import LocalEmbedder
from .index_backend import ChromaIndex, QdrantIndex, IndexBackend
