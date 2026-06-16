"""Comprehensive tests for the schema validator."""
import json
import sys
from pathlib import Path

ROOT = Path(r'C:\Users\adity\Downloads\real-port\test\air-gapped-noc-copilot')
sys.path.insert(0, str(ROOT))

from m3.prompts.schema_validator import validate_response, CopilotUnavailable, _extract_json

examples = json.loads((ROOT / "m3" / "prompts" / "few_shot_examples.json").read_text(encoding="utf-8"))


def test_three_few_shots_still_validate():
    for ex in examples:
        raw = json.dumps(ex["response"])
        try:
            validate_response(raw, evidence_chunks=ex.get("evidence", []))
            print(f"PASS  few_shot[{ex['name']}] validates")
        except CopilotUnavailable as e:
            print(f"FAIL  few_shot[{ex['name']}]: {e}")
            assert False


def test_extract_json_bare_object():
    obj, err = _extract_json('{"a": 1, "b": [1, 2]}')
    assert obj == {"a": 1, "b": [1, 2]}
    assert err is None
    print("PASS  extract bare JSON")


def test_extract_json_fenced():
    text = 'Some prose\n```json\n{"a": {"b": 1}, "c": [1, 2, 3]}\n```\nMore prose'
    obj, err = _extract_json(text)
    assert obj == {"a": {"b": 1}, "c": [1, 2, 3]}, f"expected full obj, got {obj}"
    assert err is None
    print("PASS  extract fenced JSON with nested objects")


def test_extract_json_nested_no_fence():
    text = 'Reasoning prose here\n{"predicted": {"severity": "P1", "factors": ["a", "b"]}}'
    obj, err = _extract_json(text)
    assert obj == {"predicted": {"severity": "P1", "factors": ["a", "b"]}}
    assert err is None
    print("PASS  extract nested JSON without fence (uses raw_decode)")


def test_extract_json_invalid():
    obj, err = _extract_json("this is not json at all")
    assert obj is None
    assert err is not None
    print(f"PASS  extract invalid: {err}")


def test_extract_json_truncated():
    obj, err = _extract_json('{"a": 1, "b": [1, 2')
    assert obj is None
    assert err is not None
    print(f"PASS  extract truncated: {err}")


def test_validator_rejects_garbage():
    try:
        validate_response("not even json")
        assert False, "should have raised"
    except CopilotUnavailable as e:
        assert e.reason.startswith("json_parse") or e.reason == "no_json_object_found"
        assert e.envelope.get("copilot_unavailable") is True
        print(f"PASS  validator rejects garbage: {e.reason}")


def test_validator_rejects_ungrounded_evidence():
    response = json.dumps({
        "schema_version": "1.0.0",
        "alert_id": "PE-1037",
        "generated_at": "2026-05-15T09:21:00Z",
        "answer_grounded": True,
        "missing_context": [],
        "predicted_issue": {
            "type": "congestion_saturation",
            "target": {"device": "pe-hub-east-1", "interface_or_peer": "eth3", "vrf": "cust-blue", "site": "hub-east"},
            "time_to_impact_minutes": 12,
            "confidence": 0.74,
        },
        "root_cause_hypothesis": {
            "summary": "Rising utilization on eth3.",
            "signals": [{"metric": "if_out_util_pct", "value": 71.4, "trend": "rising"}],
            "evidence_chunks": [{"chunk_id": "0" * 32, "quote": "made up quote", "relevance": "x"}],
            "confidence": 0.5,
        },
        "affected_scope": {"sites": ["hub-east"], "vrfs": ["cust-blue"], "services": ["voice"], "estimated_users_affected": 1},
        "recommended_actions": [],
        "operator_questions": {
            "q1_what_will_fail": "x" * 20,
            "q2_why_elevated_risk": "x" * 20,
            "q3_corrective_action": "x" * 20,
        },
        "warnings": [],
        "provenance": {
            "model_name": "x",
            "model_revision": "y",
            "embedding_model_fingerprint": "0" * 64,
            "index_manifest_sha256": "0" * 64,
            "evidence_chunks": [
                {"chunk_id": "1" * 32, "doc_type": "runbook", "source": "x", "score": 0.5}
            ],
        },
    })
    try:
        validate_response(response, evidence_chunks=[{"chunk_id": "1" * 32, "metadata": {"doc_type": "runbook", "source": "x"}, "score": 0.5, "text": "..."}])
        assert False, "should have raised"
    except CopilotUnavailable as e:
        assert "ungrounded_evidence_chunks" in e.reason
        print(f"PASS  validator rejects ungrounded evidence: {e.reason}")


def test_validator_accepts_fenced_response():
    fingerprint = "0" * 64
    payload = {
        "schema_version": "1.0.0",
        "alert_id": "PE-1037",
        "generated_at": "2026-05-15T09:21:00Z",
        "answer_grounded": False,
        "missing_context": ["insufficient evidence in local corpus"],
        "predicted_issue": {
            "type": "unknown",
            "target": {"device": "x", "interface_or_peer": "y", "vrf": "z", "site": "w"},
            "time_to_impact_minutes": None,
            "confidence": 0.0,
        },
        "root_cause_hypothesis": {
            "summary": "Insufficient evidence.",
            "signals": [],
            "evidence_chunks": [],
            "confidence": 0.0,
        },
        "affected_scope": {"sites": [], "vrfs": [], "services": [], "estimated_users_affected": 0},
        "recommended_actions": [],
        "operator_questions": {
            "q1_what_will_fail": "Unknown - insufficient evidence here for review",
            "q2_why_elevated_risk": "No signals grounded here for review",
            "q3_corrective_action": "Escalate to human reviewer immediately",
        },
        "warnings": ["INSUFFICIENT_EVIDENCE"],
        "provenance": {
            "model_name": "x",
            "model_revision": "y",
            "embedding_model_fingerprint": fingerprint,
            "index_manifest_sha256": fingerprint,
            "evidence_chunks": [],
        },
    }
    text = "```json\n" + json.dumps(payload) + "\n```"
    result = validate_response(text)
    assert result["answer_grounded"] is False
    assert "INSUFFICIENT_EVIDENCE" in result["warnings"]
    print("PASS  validator accepts fenced JSON response")


def test_validator_truncates_long_strings():
    long_q1 = "x" * 1000  # noqa: F841  (kept for documentation; test below uses 700)
    response = {
        "schema_version": "1.0.0",
        "alert_id": "PE-1037",
        "generated_at": "2026-05-15T09:21:00Z",
        "answer_grounded": True,
        "missing_context": [],
        "predicted_issue": {
            "type": "congestion_saturation",
            "target": {"device": "x", "interface_or_peer": "y", "vrf": "z", "site": "w"},
            "time_to_impact_minutes": 1,
            "confidence": 0.5,
        },
        "root_cause_hypothesis": {
            "summary": "x" * 700,
            "signals": [],
            "evidence_chunks": [],
            "confidence": 0.5,
        },
        "affected_scope": {"sites": [], "vrfs": [], "services": [], "estimated_users_affected": 0},
        "recommended_actions": [],
        "operator_questions": {
            "q1_what_will_fail": "x" * 700,
            "q2_why_elevated_risk": "x" * 700,
            "q3_corrective_action": "x" * 700,
        },
        "warnings": [],
        "provenance": {
            "model_name": "x", "model_revision": "y",
            "embedding_model_fingerprint": "0" * 64, "index_manifest_sha256": "0" * 64,
            "evidence_chunks": [],
        },
    }
    result = validate_response(json.dumps(response))
    assert len(result["operator_questions"]["q1_what_will_fail"]) <= 400
    assert "TRUNCATED" in result["warnings"]
    print("PASS  validator truncates long strings")


def test_validator_cli(tmp_path: Path):
    res_file = tmp_path / "response.json"
    res_file.write_text(json.dumps(examples[0]["response"]), encoding="utf-8")
    ev_file = tmp_path / "evidence.json"
    ev_file.write_text(json.dumps({"evidence": examples[0]["evidence"]}), encoding="utf-8")
    import subprocess
    proc = subprocess.run(
        [sys.executable, "-m", "m3.prompts.schema_validator",
         "--response", str(res_file), "--evidence", str(ev_file)],
        capture_output=True, text=True, env={"PYTHONPATH": str(ROOT), **__import__("os").environ},
    )
    assert proc.returncode == 0, f"rc={proc.returncode}, stderr={proc.stderr}"
    out = json.loads(proc.stdout)
    assert out["status"] == "OK"
    print("PASS  schema_validator.py CLI works")


if __name__ == "__main__":
    import tempfile
    test_three_few_shots_still_validate()
    test_extract_json_bare_object()
    test_extract_json_fenced()
    test_extract_json_nested_no_fence()
    test_extract_json_invalid()
    test_extract_json_truncated()
    test_validator_rejects_garbage()
    test_validator_rejects_ungrounded_evidence()
    test_validator_accepts_fenced_response()
    test_validator_truncates_long_strings()
    with tempfile.TemporaryDirectory() as tmp:
        test_validator_cli(Path(tmp))
    print("\nAll schema_validator tests passed.")
