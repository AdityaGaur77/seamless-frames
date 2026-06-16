"""Test the fault_injection_runner with a fast tick and a stubbed Copilot."""
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path

ROOT = Path(r'C:\Users\adity\Downloads\real-port\test\air-gapped-noc-copilot')
sys.path.insert(0, str(ROOT))

from m3.playbooks.fault_injection_runner import SCENARIOS, run_scenario, _disallow_network, _query_prometheus
from m3.rag.rag_query import query as rag_query

os.environ["NOC_COPILOT_SKIP_AIRGAP_CHECK"] = "1"
os.environ["NOC_COPILOT_FAKE_EMBEDDER"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def test_airgap_skip_env_var():
    """The skip env var should allow _disallow_network to pass even with internet."""
    _disallow_network()
    print("PASS  _disallow_network: skip env var honoured")


def test_all_four_scenarios_defined():
    assert set(SCENARIOS.keys()) == {1, 2, 3, 4}
    for sid, spec in SCENARIOS.items():
        assert "name" in spec
        assert "duration_minutes" in spec
        assert "first_elevated_minute" in spec
        assert "expected_issue_type" in spec
        assert "expected_action" in spec
        assert "expected_runbook_id" in spec
        assert "alert_payload" in spec
        assert spec["alert_payload"]["alert_id"]
        assert spec["first_elevated_minute"] <= spec["duration_minutes"]
        for sig in spec["alert_payload"]["signals"]:
            assert "metric" in sig
    print(f"PASS  all 4 scenarios have well-formed specs")


def test_evidence_lookup_with_real_index(tmp_dir: Path):
    """End-to-end: build a fake index, then run the runner's evidence lookup."""
    from m3.rag.rag_ingest import _iter_chunks
    from m3.rag.chunker import chunk_corpus
    from m3.rag.embedder import LocalEmbedder
    from m3.rag.index_backend import ChromaIndex
    import yaml
    import numpy as np

    corpus = ROOT / "m3" / "rag" / "corpus"
    chunks = list(chunk_corpus(corpus))
    assert len(chunks) > 0

    emb = LocalEmbedder(model_path=str(tmp_dir / "fake"), dimension=64)
    vecs = emb.embed_documents([c.text for c in chunks])
    idx = ChromaIndex(path=str(tmp_dir / "chroma"), collection="playbook_test")
    idx.upsert(chunks, vecs)
    assert idx.count() == len(chunks)

    cfg = {
        "embed": {"model_name": "x", "model_path": str(tmp_dir / "fake"), "dimension": 64, "normalize": True, "query_prefix": "", "batch_size": 32},
        "retriever": {"backend": "chroma", "chroma_path": str(tmp_dir / "chroma"), "chroma_collection": "playbook_test", "qdrant_url": "", "qdrant_collection": "x", "hybrid_weight_dense": 0.7, "hybrid_weight_lex": 0.3, "top_k": 8, "context_token_budget": 3500},
        "index": {"max_index_age_days": 30, "manifest_path": "index_manifest.json"},
        "runtime": {"log_level": "INFO", "offline_mode": True, "refuse_network_calls": False},
    }
    cfg_path = tmp_dir / "cfg.yaml"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_dir / "fake").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "fake" / "config.json").write_text("{}", encoding="utf-8")

    for sid, spec in SCENARIOS.items():
        alert = spec["alert_payload"]
        q = f"{alert['signals'][0]['metric']} on {alert['site']}"
        res = rag_query(cfg_path, tmp_dir, q, top_k=8)
        assert len(res["evidence"]) > 0, f"scenario {sid}: no evidence for {q}"
    print(f"PASS  all 4 scenarios retrieve evidence via the RAG query API")


def test_run_scenario_with_no_copilot(tmp_dir: Path):
    """Drive scenario 1 end-to-end with tick_seconds=0 and no Copilot."""
    cfg = {
        "embed": {"model_name": "x", "model_path": str(tmp_dir / "fake"), "dimension": 64, "normalize": True, "query_prefix": "", "batch_size": 32},
        "retriever": {"backend": "chroma", "chroma_path": str(tmp_dir / "chroma"),             "chroma_collection": "playbook_v1", "qdrant_url": "", "qdrant_collection": "x", "hybrid_weight_dense": 0.7, "hybrid_weight_lex": 0.3, "top_k": 8, "context_token_budget": 3500},
        "index": {"max_index_age_days": 30, "manifest_path": "index_manifest.json"},
        "runtime": {"log_level": "INFO", "offline_mode": True, "refuse_network_calls": False},
    }
    cfg_path = tmp_dir / "cfg.yaml"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_dir / "fake").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "fake" / "config.json").write_text("{}", encoding="utf-8")

    from m3.rag.rag_ingest import _iter_chunks
    from m3.rag.chunker import chunk_corpus
    from m3.rag.embedder import LocalEmbedder
    from m3.rag.index_backend import ChromaIndex
    corpus = ROOT / "m3" / "rag" / "corpus"
    chunks = list(chunk_corpus(corpus))
    emb = LocalEmbedder(model_path=str(tmp_dir / "fake"), dimension=64)
    vecs = emb.embed_documents([c.text for c in chunks])
    idx = ChromaIndex(path=str(tmp_dir / "chroma"), collection="playbook_v1")
    idx.upsert(chunks, vecs)
    manifest_path = tmp_dir / "index_manifest.json"
    manifest_path.write_text(json.dumps({"chunk_count": len(chunks), "backend": "chroma", "embedder_dim": 64, "schema_version": 1}), encoding="utf-8")

    def lookup(alert):
        q = f"{alert['signals'][0]['metric']} on {alert['site']}"
        return rag_query(cfg_path, tmp_dir, q, top_k=8).get("evidence", [])

    inject_calls = []
    def fake_inject(minute, spec):
        inject_calls.append(minute)

    report = run_scenario(
        scenario_id=1,
        cfg={},
        prometheus_url="http://127.0.0.1:65535",
        copilot_url=None,
        evidence_lookup=lookup,
        inject=fake_inject,
        tick_seconds=0,
    )
    assert report.scenario_id == 1
    assert report.copilot_unavailable_reason == "copilot_url_not_configured"
    assert "lead_time_minutes" in report.validation
    assert report.validation["lead_time_minutes"] >= 0
    assert len(inject_calls) == 31
    assert report.first_elevated_at is not None
    assert report.actual_impact_at is not None
    assert report.actual_impact_at >= report.first_elevated_at
    print(f"PASS  run_scenario(1) without copilot: lead_time={report.validation['lead_time_minutes']:.4f} min, {len(inject_calls)} inject calls")


def test_validation_metrics_aggregation(tmp_dir: Path):
    from m3.playbooks.validation_metrics import summarise, render_markdown
    reports = [
        {"scenario_id": 1, "name": "congestion", "first_elevated_at": 100.0, "actual_impact_at": 700.0, "validation": {"lead_time_minutes": 10.0, "grounding_recall_at_k": 1.0, "fabrication_rate": 0.0, "first_action_match": True, "predicted_issue_type_match": True}},
        {"scenario_id": 2, "name": "bgp", "first_elevated_at": 100.0, "actual_impact_at": 700.0, "validation": {"lead_time_minutes": 10.0, "grounding_recall_at_k": 1.0, "fabrication_rate": 0.0, "first_action_match": False, "predicted_issue_type_match": True}},
        {"scenario_id": 3, "name": "mpls", "first_elevated_at": 100.0, "actual_impact_at": 700.0, "validation": {"lead_time_minutes": 10.0, "grounding_recall_at_k": 0.5, "fabrication_rate": 0.1, "first_action_match": True, "predicted_issue_type_match": True}},
    ]
    s = summarise(reports)
    assert s["scenarios_run"] == 3
    assert s["mean_lead_time_minutes"] == 10.0
    assert s["mean_fabrication_rate"] == round((0.0 + 0.0 + 0.1) / 3, 3)
    assert s["mean_action_applicability"] == round((1.0 + 0.0 + 1.0) / 3, 3)
    md = render_markdown(s)
    assert "congestion" in md and "bgp" in md and "mpls" in md
    print(f"PASS  validation_metrics.summarise: scenarios={s['scenarios_run']}, mean_fabrication={s['mean_fabrication_rate']}, mean_action={s['mean_action_applicability']}")


if __name__ == "__main__":
    test_airgap_skip_env_var()
    test_all_four_scenarios_defined()

    tmp_dir = Path(tempfile.mkdtemp(prefix="playbook_test_"))
    try:
        test_evidence_lookup_with_real_index(tmp_dir)
        test_run_scenario_with_no_copilot(tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    test_validation_metrics_aggregation(Path(tempfile.mkdtemp()))
    print("\nAll playbook tests passed.")
