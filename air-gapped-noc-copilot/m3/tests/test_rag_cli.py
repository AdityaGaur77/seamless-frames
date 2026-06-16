"""Test the rag_ingest.py and rag_query.py CLI entry points end-to-end."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(r'C:\Users\adity\Downloads\real-port\test\air-gapped-noc-copilot')
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "m3" / "rag" / "corpus"
FAKE_ENV = {
    "PYTHONPATH": str(ROOT),
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "NOC_COPILOT_FAKE_EMBEDDER": "1",
}


def write_fake_config(index_dir: Path) -> Path:
    cfg = {
        "embed": {
            "model_name": "fake-embedder",
            "model_path": str(index_dir / "fake_model"),
            "dimension": 64,
            "batch_size": 16,
            "normalize": True,
            "query_prefix": "",
        },
        "retriever": {
            "backend": "chroma",
            "chroma_path": str(index_dir / "chroma"),
            "chroma_collection": "noc_copilot_e2e",
            "qdrant_url": "http://127.0.0.1:6333",
            "qdrant_collection": "noc_copilot_e2e",
            "hybrid_weight_dense": 0.7,
            "hybrid_weight_lex": 0.3,
            "top_k": 5,
            "context_token_budget": 1500,
        },
        "index": {
            "max_index_age_days": 30,
            "manifest_path": "index_manifest.json",
        },
        "corpus": {
            "roots": [str(CORPUS)],
            "file_globs": ["*.md", "*.txt", "*.yaml", "*.yml", "*.json"],
        },
        "chunker": {
            "topology": {"target_tokens": 140, "overlap_tokens": 0},
            "runbook": {"target_tokens": 350, "overlap_tokens": 50},
            "incident": {"detail_target_tokens": 300, "summary_target_tokens": 80, "overlap_tokens": 0},
        },
        "runtime": {"log_level": "INFO", "offline_mode": True, "refuse_network_calls": False},
    }
    cfg_path = index_dir / "rag_config.yaml"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    fake_model_dir = index_dir / "fake_model"
    fake_model_dir.mkdir(parents=True, exist_ok=True)
    (fake_model_dir / "config.json").write_text('{"fake": true}', encoding="utf-8")
    return cfg_path


def test_rag_ingest_and_query():
    index_dir = Path(tempfile.mkdtemp(prefix="rag_e2e_"))
    try:
        cfg_path = write_fake_config(index_dir)
        cmd = [sys.executable, "-m", "m3.rag.rag_ingest",
               "--corpus", str(CORPUS),
               "--index", str(index_dir),
               "--config", str(cfg_path),
               "--backend", "chroma"]
        res = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **FAKE_ENV}, timeout=60)
        if res.returncode != 0:
            print("STDOUT:", res.stdout)
            print("STDERR:", res.stderr)
            assert False, f"rag_ingest returned {res.returncode}"
        manifest = index_dir / "index_manifest.json"
        assert manifest.exists(), "manifest not written"
        m = json.loads(manifest.read_text(encoding="utf-8"))
        assert m["chunk_count"] > 0
        assert m["backend"] == "chroma"
        assert m["embedder_dim"] == 64
        print(f"PASS  rag_ingest.py CLI: ingested {m['chunk_count']} chunks; manifest written")

        out_path = index_dir / "evidence.json"
        cmd2 = [sys.executable, "-m", "m3.rag.rag_query",
                "--config", str(cfg_path),
                "--index", str(index_dir),
                "--query", "BGP neighbor stuck in Active on pe-hub-east-1",
                "--top-k", "5",
                "--out", str(out_path)]
        res2 = subprocess.run(cmd2, capture_output=True, text=True, env={**os.environ, **FAKE_ENV}, timeout=30)
        if res2.returncode != 0:
            print("STDOUT:", res2.stdout)
            print("STDERR:", res2.stderr)
            assert False, f"rag_query returned {res2.returncode}"
        evidence = json.loads(out_path.read_text(encoding="utf-8"))
        assert "evidence" in evidence
        assert len(evidence["evidence"]) > 0, f"expected hits, got 0"
        runbook_hits = [e for e in evidence["evidence"] if e["metadata"].get("doc_type") == "runbook"]
        print(f"PASS  rag_query.py CLI: {len(evidence['evidence'])} hits ({len(runbook_hits)} runbook) for BGP query")
    finally:
        shutil.rmtree(index_dir, ignore_errors=True)


if __name__ == "__main__":
    test_rag_ingest_and_query()
    print("\nAll RAG CLI tests passed.")
