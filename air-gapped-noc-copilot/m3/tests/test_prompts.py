"""Test the prompt assembler with a real alert + real evidence."""
import json
import sys
import subprocess
from pathlib import Path

ROOT = Path(r'C:\Users\adity\Downloads\real-port\test\air-gapped-noc-copilot')
sys.path.insert(0, str(ROOT))

from m3.prompts.prompt_assembler import assemble, load_system_prompt, estimate_tokens

sp = load_system_prompt()
assert "R1" in sp and "R10" in sp
assert "AIRGAP-NOC" in sp
print(f"PASS  load_system_prompt: {len(sp)} chars; rules R1..R10 present")

alert = {
    "alert_id": "PE-1037",
    "severity": "P3",
    "risk_band": "ELEVATED",
    "model_name": "ensemble-v3",
    "model_confidence": 0.78,
    "time_to_impact_minutes": 12,
    "signals": [
        {"metric": "if_out_util_pct", "device": "pe-hub-east-1", "interface": "eth3", "value": 71.4, "trend": "rising"}
    ],
    "site": "hub-east",
    "vrf": "cust-blue",
}

evidence = [
    {
        "chunk_id": "11111111111111111111111111111111",
        "metadata": {"doc_type": "topology", "source": "corpus/topology/topology_reference.yaml"},
        "score": 0.71,
        "text": "pe-hub-east-1 connects 4 branches and 1 DC over MPLS. Uplink eth3 to p-core-1 is 40G.",
    },
    {
        "chunk_id": "22222222222222222222222222222222",
        "metadata": {"doc_type": "runbook", "runbook_id": "MPLS-003", "source": "corpus/runbooks/mpls/MPLS-003_underlay_loss.md"},
        "score": 0.65,
        "text": "Apply a soft pre-emptive reroute by raising the TE cost of the affected link by 1000.",
    },
]

messages = assemble(alert, evidence, "what's the safest reroute here?")
assert len(messages) == 2
assert messages[0]["role"] == "system"
assert messages[1]["role"] == "user"
assert "ALERT_PAYLOAD" in messages[1]["content"]
assert "RETRIEVED_EVIDENCE" in messages[1]["content"]
assert "OPERATOR_QUESTION" in messages[1]["content"]
assert "what's the safest reroute here?" in messages[1]["content"]
assert "chunk_id=11111111111111111111111111111111" in messages[1]["content"]
print(f"PASS  assemble: {len(messages)} messages, token_estimate={estimate_tokens(messages)}")

empty_messages = assemble(alert, [], "")
assert "RETRIEVED_EVIDENCE = {\n  chunks = [\n  ]" in empty_messages[1]["content"]
print(f"PASS  assemble with empty evidence: chunks = []")

subprocess_ok = True
try:
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(alert, f)
        alert_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump({"evidence": evidence}, f)
        ev_path = f.name
    out_path = Path(alert_path).with_suffix(".out.json")
    res = subprocess.run(
        [sys.executable, "-m", "m3.prompts.prompt_assembler",
         "--alert", alert_path, "--evidence", ev_path, "--question", "test", "--out", str(out_path)],
        capture_output=True, text=True, env={"PYTHONPATH": str(ROOT), **__import__("os").environ},
    )
    if res.returncode != 0:
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        subprocess_ok = False
    else:
        out = json.loads(out_path.read_text(encoding="utf-8"))
        assert "messages" in out
        assert "token_estimate" in out
        assert out["evidence_chunk_count"] == 2
        print(f"PASS  prompt_assembler.py CLI: wrote {out_path.name}, {out['token_estimate']} tokens")
except Exception as e:
    subprocess_ok = False
    print(f"FAIL  prompt_assembler.py CLI: {e}")

print("\nAll prompt_assembler tests passed." if subprocess_ok else "\nFAIL")
