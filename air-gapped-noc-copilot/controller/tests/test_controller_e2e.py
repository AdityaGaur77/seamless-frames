"""End-to-end test of the orchestrator (controller package).

Exercises the full data flow with no live dependencies:

    FixtureSampler  →  ModelScorer (random-init)  →  AlertBuilder
        →  RagBridge (m3.rag.rag_query with FAKE_EMBEDDER + m3/rag/index)
        →  PromptOrchestrator  →  StubLLMClient  →  ResponseGate

The LLM is faked. The RAG index is the real Chroma persistent index
under ``m3/rag/index/chroma``, but the embedder is replaced with the
deterministic hash embedder (``NOC_COPILOT_FAKE_EMBEDDER=1``) so the
test does not need ``bge-small-en-v1.5`` on disk.

Run:
    cd air-gapped-noc-copilot
    NOC_COPILOT_FAKE_EMBEDDER=1 \
    NOC_COPILOT_SKIP_AIRGAP_CHECK=1 \
    python -m controller.tests.test_controller_e2e
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_ROOT = PROJECT_ROOT / "controller"
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("NOC_COPILOT_FAKE_EMBEDDER", "1")
os.environ.setdefault("NOC_COPILOT_SKIP_AIRGAP_CHECK", "1")

from controller.alert_builder import build_alert_payload, validate_alert_payload  # noqa: E402
from controller.config import OrchestratorConfig, SamplerConfig  # noqa: E402
from controller.infer_client import OfflineLLMClient, OfflineLLMUnavailable  # noqa: E402
from controller.metric_sampler import FixtureSampler  # noqa: E402
from controller.model_scorer import ModelScorer  # noqa: E402
from controller.orchestrator import Orchestrator  # noqa: E402
from controller.prompt_orchestrator import PromptOrchestrator  # noqa: E402
from controller.rag_bridge import RagBridge  # noqa: E402
from controller.response_gate import ResponseGate  # noqa: E402

LOG = logging.getLogger("controller.tests.e2e")


# ── stub LLM ────────────────────────────────────────────────────


_VALID_RESPONSE = {
    "schema_version": "1.0.0",
    "alert_id": "ALR-0000001",
    "generated_at": "2026-06-16T10:00:00Z",
    "answer_grounded": True,
    "predicted_issue": {
        "type": "congestion_saturation",
        "target": {
            "device": "pe-hub-east-1",
            "interface_or_peer": "eth3",
            "vrf": "cust-blue",
            "site": "east-1",
        },
        "time_to_impact_minutes": 12,
        "confidence": 0.78,
    },
    "root_cause_hypothesis": {
        "summary": "Interface utilization on pe-hub-east-1:eth3 is rising rapidly toward 95% with no headroom on the AF21-data queue, indicating imminent saturation.",
        "signals": [
            {"metric": "interface_utilization", "value": 71.4, "trend": "rising"},
        ],
        "evidence_chunks": [],
        "confidence": 0.78,
    },
    "affected_scope": {
        "sites": ["east-1"],
        "vrfs": ["cust-blue"],
        "services": ["voice", "video"],
        "estimated_users_affected": 250,
    },
    "recommended_actions": [
        {
            "order": 1,
            "action": "open_change_ticket",
            "target": "pe-hub-east-1 eth3",
            "risk": "low",
            "expected_effect": "Captures the change window for the TE reroute.",
            "rollback": "n/a",
            "linked_runbook_chunk_id": None,
        }
    ],
    "operator_questions": {
        "q1_what_will_fail": "The pe-hub-east-1 uplink eth3 to p-core-1 will saturate in ~12 minutes, dropping voice and video on cust-blue.",
        "q2_why_elevated_risk": "if_out_util_pct on eth3 is 71.4% and rising; AF21-data queue depth is 482000 bytes and rising. The model has 0.78 confidence.",
        "q3_corrective_action": "Open a change ticket, then raise the TE cost on pe-hub-east-1 eth3 by 1000 per MPLS-003 step 6. Monitor AF21 drops for 10 min; rollback with no mpls traffic-eng interface eth3 metric 2000 if traffic shifts are unexpected.",
    },
    "warnings": [],
    "provenance": {
        "model_name": "airgap-noc",
        "model_revision": "0",
        "embedding_model_fingerprint": "0" * 64,
        "index_manifest_sha256": "0" * 64,
        "evidence_chunks": [],
    },
}


class StubLLMClient(OfflineLLMClient):
    """Returns a canned valid response for the first call; tracks calls."""

    def __init__(self):
        # Bypass the parent __init__ entirely — no network session needed.
        self.calls: List[List[Dict[str, str]]] = []
        self.scripted = [_valid_response_text()]

    def health(self) -> bool:  # type: ignore[override]
        return True

    def chat(self, messages, *, response_format=None):  # type: ignore[override]
        self.calls.append(messages)
        if not self.scripted:
            raise OfflineLLMUnavailable("no_more_scripted_responses")
        return self.scripted.pop(0)


def _valid_response_text() -> str:
    return json.dumps(_VALID_RESPONSE)


# ── the test ────────────────────────────────────────────────────


def _make_config(tmpdir: Path) -> OrchestratorConfig:
    cfg = OrchestratorConfig()
    cfg.max_cycles = 1
    cfg.sample_interval_seconds = 1
    cfg.log_level = "WARNING"
    cfg.sampler = SamplerConfig(
        backend="fixture",
        sequence_length=cfg.sampler.sequence_length,
        num_features=cfg.sampler.num_features,
    )
    cfg.model.model_path = str(tmpdir / "fake.pth")
    cfg.model.scaler_path = ""  # no scaler; scorer falls back to raw features
    cfg.model.architecture = "lstm_multitask"
    cfg.model.anomaly_threshold = 0.10  # low so the cycle produces an alert
    cfg.sink.ndjson_path = str(tmpdir / "sink.ndjson")
    cfg.sink.latest_path = str(tmpdir / "latest.json")
    return cfg


def _make_dummy_checkpoint(path: Path) -> None:
    """Write a minimal checkpoint the LSTMMultiTask loader can ingest.

    We initialise a model with the right shape, then dump its
    state_dict — the test only needs the *forward pass* to run, the
    weights don't have to be good.
    """
    import torch

    from lstm_model import LSTMMultiTask

    model = LSTMMultiTask(
        input_size=25,
        hidden_size=128,
        num_layers=2,
        forecast_horizon=10,
        dropout=0.0,
    )
    torch.save({"model_state_dict": model.state_dict()}, path)


def _patch_orchestrator_components(orch: Orchestrator, stub: StubLLMClient) -> None:
    orch.llm = stub
    orch.gate = ResponseGate()


class ControllerE2ETest(unittest.TestCase):
    def test_full_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _make_dummy_checkpoint(tmp / "fake.pth")

            cfg = _make_config(tmp)
            orch = Orchestrator(cfg)
            stub = StubLLMClient()
            _patch_orchestrator_components(orch, stub)
            orch.run()

            # ── assertions ────────────────────────────────────
            self.assertEqual(len(stub.calls), 1, "LLM should be called once per anomalous frame")
            messages = stub.calls[0]
            self.assertEqual(messages[0]["role"], "system")
            self.assertIn("ALERT_PAYLOAD", messages[1]["content"])

            # The sink must contain exactly one line.
            sink_text = (tmp / "sink.ndjson").read_text(encoding="utf-8")
            self.assertTrue(sink_text.strip(), "sink should have a line")
            record = json.loads(sink_text.strip().splitlines()[-1])

            # The latest file mirrors the last record.
            latest = json.loads((tmp / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest, record)

            # The record should be a valid, grounded CopilotResponse.
            self.assertNotIn("copilot_unavailable", record)
            self.assertTrue(record["answer_grounded"])
            self.assertEqual(record["schema_version"], "1.0.0")
            self.assertEqual(record["predicted_issue"]["type"], "congestion_saturation")
            self.assertIn("q1_what_will_fail", record["operator_questions"])
            self.assertIn("q2_why_elevated_risk", record["operator_questions"])
            self.assertIn("q3_corrective_action", record["operator_questions"])

    def test_alert_builder_schema(self) -> None:
        from controller.metric_sampler import MetricFrame
        from controller.model_scorer import ScoringResult
        import numpy as np
        from datetime import datetime, timezone

        frame = MetricFrame(
            host="pe-hub-east-1",
            interface="eth3",
            feature_columns=["f0"] * 25,
            tensor=np.zeros((60, 25), dtype=np.float32),
            latest_signals=[
                {
                    "metric": "interface_utilization",
                    "value": 71.4,
                    "trend": "rising",
                    "host": "pe-hub-east-1",
                    "interface": "eth3",
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            ],
            generated_at=datetime.now(timezone.utc),
        )
        scoring = ScoringResult(
            host="pe-hub-east-1",
            interface="eth3",
            architecture="lstm_multitask",
            forecast=[80.0] * 10,
            anomaly_prob=[0.9] * 10,
            tti_minutes=[15.0] * 10,
            checkpoint_sha256="0" * 64,
            scaler_sha256="0" * 64,
        )
        payload = build_alert_payload(
            scoring=scoring,
            frame=frame,
            model_name="airgap-noc",
            model_version="abc",
            sampler_name="fixture",
            sequence_length=60,
            num_features=25,
        )
        validate_alert_payload(payload)  # raises on invalid
        self.assertEqual(payload["schema_version"], "1.0.0")
        self.assertEqual(payload["predicted_issue"]["target"]["device"], "pe-hub-east-1")
        self.assertGreater(payload["predicted_issue"]["confidence"], 0.5)

    def test_below_threshold_no_llm_call(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _make_dummy_checkpoint(tmp / "fake.pth")
            cfg = _make_config(tmp)
            cfg.model.anomaly_threshold = 0.99  # nothing will ever cross this
            orch = Orchestrator(cfg)
            stub = StubLLMClient()
            orch.llm = stub
            orch.gate = ResponseGate()
            orch.run()
            self.assertEqual(stub.calls, [], "no LLM call when below threshold")


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(ControllerE2ETest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
