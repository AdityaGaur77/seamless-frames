"""Phase 3 → 5 orchestrator: live telemetry → ML scoring → offline RAG copilot.

Public surface:

    from controller import Orchestrator, OrchestratorConfig
    from controller.orchestrator import main

The controller is the *only* process that needs to run for the air-gapped
NOC Copilot to be live. It is a single asyncio-free, single-threaded loop
that:

    1. samples the latest N-step window from TimescaleDB / Prometheus,
    2. runs the trained LSTM/TCN model to produce an anomaly score and
       time-to-impact,
    3. builds a structured ALERT_PAYLOAD JSON,
    4. queries the local Chroma index (m3.rag.rag_query) with site /
       device / protocol filters derived from the alert,
    5. assembles a chat prompt (m3.prompts.prompt_assembler) bound to
       the SYSTEM_PROMPT and the strict response schema,
    6. calls the offline LLM (llama.cpp / Ollama / vLLM, bound to
       127.0.0.1) over a synchronous HTTP chat-completion,
    7. validates the response (m3.prompts.schema_validator) and either
       publishes a NOC-ready CopilotResponse or a `copilot_unavailable`
       banner to the NOC UI.

Every step is synchronous and side-effecting only at the end (writes one
JSON line per cycle to the configured sink). The loop sleeps for
`sample_interval_seconds` between iterations.

Design constraints (inherited from m3/ARCHITECTURE.md):

    * No outbound network calls at runtime.
    * Deterministic, idempotent, fail-loud (no silent fallbacks).
    * The response schema is the contract with the LLM; the validator is
      the contract with the operator UI.
"""
from __future__ import annotations

from .alert_builder import build_alert_payload, validate_alert_payload
from .config import OrchestratorConfig, load_config
from .infer_client import OfflineLLMClient, OfflineLLMUnavailable
from .metric_sampler import (
    FixtureSampler,
    MetricFrame,
    MetricSampler,
    PrometheusSampler,
    TimescaleDBSampler,
)
from .model_scorer import ModelScorer, ScoringResult
from .orchestrator import Orchestrator, main
from .prompt_orchestrator import PromptOrchestrator
from .rag_bridge import RagBridge
from .response_gate import ResponseGate

__all__ = [
    "Orchestrator",
    "OrchestratorConfig",
    "OfflineLLMClient",
    "OfflineLLMUnavailable",
    "MetricSampler",
    "PrometheusSampler",
    "TimescaleDBSampler",
    "FixtureSampler",
    "MetricFrame",
    "ModelScorer",
    "ScoringResult",
    "PromptOrchestrator",
    "RagBridge",
    "ResponseGate",
    "build_alert_payload",
    "validate_alert_payload",
    "load_config",
    "main",
]
