"""The main loop.

Pipeline (per cycle):

    1. sampler.fetch_all()        → list[MetricFrame]
    2. for each frame:
         scorer.score(...)        → ScoringResult
         alert = build_alert_payload(...)
         validate_alert_payload(alert)        (fail-loud)
         evidence = rag_bridge.retrieve(alert)
         messages = prompt_orchestrator.build(alert, evidence)
         llm_text = infer_client.chat(messages)
         response = response_gate.accept(llm_text, evidence)
    3. publish one JSON per cycle to the configured sink.

The orchestrator never falls back silently. If any step fails and
``fail_loud`` is true, the loop logs the error and continues with the
next cycle (so a transient RAG/LLM outage does not stop the scoring
loop), but the cycle's published object is a
``copilot_unavailable`` envelope rather than a partial answer.
"""
from __future__ import annotations

import json
import logging
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .alert_builder import build_alert_payload, validate_alert_payload
from .config import OrchestratorConfig, load_config
from .infer_client import OfflineLLMClient, OfflineLLMUnavailable, OllamaClient
from .metric_sampler import (
    FixtureSampler,
    MetricFrame,
    MetricSampler,
    PrometheusSampler,
    TimescaleDBSampler,
)
from .model_scorer import ModelScorer, ScoringResult
from .prompt_orchestrator import PromptOrchestrator
from .rag_bridge import RagBridge
from .response_gate import ResponseGate

LOG = logging.getLogger("controller.orchestrator")


# Default feature columns — must match
# ``data_preprocessor.PreprocessingConfig.feature_columns``. The
# defaults the LSTM was trained on are 25 numeric features; the first
# ten are the ones the alert builder echoes into ``signals[]``.
DEFAULT_FEATURE_COLUMNS = [
    "interface_utilization",
    "interface_in_errors",
    "interface_out_errors",
    "interface_in_discards",
    "interface_out_discards",
    "interface_in_packets",
    "interface_out_packets",
    "latency_avg_ms",
    "latency_jitter_ms",
    "packet_loss_percent",
    "ospf_neighbor_state",
    "ospf_dead_timer",
    "bgp_fsm_transitions",
    "bgp_update_rate",
    "mpls_ldp_state_changes",
    "ipsec_throughput_bps",
    "ipsec_error_rate",
    "congestion_trend",
    "queue_depth",
    "tti_estimate",
    "hour_of_day",
    "day_of_week",
    "is_peak_hour",
    "cpu_utilization",
    "memory_utilization",
]


class Orchestrator:
    """The end-to-end controller.

    Constructed once at process start. The :meth:`run` method blocks
    until :meth:`stop` is called (or :attr:`max_cycles` cycles have
    elapsed). Each cycle writes one NDJSON line to the configured
    sink, plus a sidecar "latest" file for the NOC UI to poll.
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._stop = False
        self._cycle = 0
        self._install_signal_handlers()

        # ── build sub-components ───────────────────────────────
        self.feature_columns: List[str] = list(DEFAULT_FEATURE_COLUMNS)[: self.config.sampler.num_features]
        self.sampler = self._build_sampler()
        self.scorer = self._build_scorer()
        self.rag = self._build_rag()
        self.prompts = PromptOrchestrator()
        self.llm = self._build_llm()
        self.gate = ResponseGate()

        # ── sinks ─────────────────────────────────────────────
        self._sink_path = (
            Path(self.config.sink.ndjson_path).resolve()
            if self.config.sink.ndjson_path
            else None
        )
        self._latest_path = (
            Path(self.config.sink.latest_path).resolve()
            if self.config.sink.latest_path
            else None
        )

    # ── factory helpers ─────────────────────────────────────────

    def _build_sampler(self) -> MetricSampler:
        s = self.config.sampler
        if s.backend == "prometheus":
            return PrometheusSampler(
                feature_columns=self.feature_columns,
                sequence_length=s.sequence_length,
                base_url=s.prometheus_url,
                target_hosts=s.target_hosts or None,
                target_interfaces=s.target_interfaces or None,
                timeout_seconds=s.poll_timeout_seconds,
            )
        if s.backend == "timescaledb":
            return TimescaleDBSampler(
                feature_columns=self.feature_columns,
                sequence_length=s.sequence_length,
                host=s.db_host,
                port=s.db_port,
                dbname=s.db_name,
                user=s.db_user,
                password=s.db_password,
                connect_timeout=s.db_connect_timeout,
                target_hosts=s.target_hosts or None,
                target_interfaces=s.target_interfaces or None,
                active_window_seconds=s.active_window_seconds,
            )
        if s.backend == "fixture":
            return FixtureSampler(
                feature_columns=self.feature_columns,
                sequence_length=s.sequence_length,
            )
        raise ValueError(f"Unknown sampler backend: {s.backend}")

    def _build_scorer(self) -> ModelScorer:
        m = self.config.model
        scaler_path = self.config.resolve(m.scaler_path) if m.scaler_path else None
        return ModelScorer(
            checkpoint_path=self.config.resolve(m.model_path),
            scaler_path=scaler_path,
            architecture=m.architecture,
            device=m.device,
            num_features=self.config.sampler.num_features,
            forecast_horizon=10,
        )

    def _build_rag(self) -> RagBridge:
        r = self.config.rag
        return RagBridge(
            config_path=self.config.resolve(r.config_path),
            index_root=self.config.resolve(r.index_root),
            top_k=r.top_k,
            context_token_budget=r.context_token_budget,
            default_filters=r.default_filters,
        )

    def _build_llm(self) -> OfflineLLMClient | OllamaClient:
        c = self.config.llm
        if c.backend == "ollama":
            return OllamaClient(
                base_url=c.ollama_url,
                model_name=c.ollama_model,
                request_timeout_seconds=c.request_timeout_seconds,
                max_retries=c.max_retries,
                temperature=c.temperature,
                top_p=c.top_p,
                max_tokens=c.max_tokens,
                num_ctx=c.ollama_num_ctx,
                num_gpu=c.ollama_num_gpu,
            )
        return OfflineLLMClient(
            base_url=c.base_url,
            api_path=c.api_path,
            model_name=c.model_name,
            request_timeout_seconds=c.request_timeout_seconds,
            max_retries=c.max_retries,
            temperature=c.temperature,
            top_p=c.top_p,
            max_tokens=c.max_tokens,
        )

    # ── signal handling ────────────────────────────────────────

    def _install_signal_handlers(self) -> None:
        def _handler(signum, frame):  # noqa: ARG001
            LOG.info("Signal %d received; requesting shutdown", signum)
            self._stop = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except ValueError:
                # Not in main thread (e.g. embedded use) — skip.
                pass

    def stop(self) -> None:
        self._stop = True

    # ── main loop ──────────────────────────────────────────────

    def run(self) -> None:
        LOG.info(
            "Orchestrator starting: interval=%ds max_cycles=%d sampler=%s model=%s",
            self.config.sample_interval_seconds,
            self.config.max_cycles,
            self.config.sampler.backend,
            self.config.model.architecture,
        )
        while not self._stop:
            self._cycle += 1
            try:
                self._run_one_cycle()
            except Exception as exc:  # noqa: BLE001
                LOG.error("Cycle %d failed: %s", self._cycle, exc)
                LOG.debug(traceback.format_exc())
                if self.config.fail_loud:
                    # Fail loud = write a banner so the NOC UI does not
                    # think the loop is alive but silent.
                    self._publish(
                        {
                            "copilot_unavailable": True,
                            "reason": "orchestrator_exception",
                            "cycle": self._cycle,
                            "error": repr(exc),
                            "ts": datetime.now(timezone.utc).isoformat(),
                        }
                    )
            if self.config.max_cycles and self._cycle >= self.config.max_cycles:
                LOG.info("Max cycles reached (%d); exiting", self.config.max_cycles)
                break
            # Sleep in small slices so SIGINT is responsive.
            self._sleep(self.config.sample_interval_seconds)
        LOG.info("Orchestrator stopped after %d cycle(s)", self._cycle)

    def _run_one_cycle(self) -> None:
        LOG.info("── cycle %d ──", self._cycle)
        frames = self.sampler.fetch_all()
        if not frames:
            LOG.warning("No metric frames this cycle; sleeping")
            return

        for frame in frames:
            response = self._handle_one_frame(frame)
            self._publish(response)

    def _handle_one_frame(self, frame: MetricFrame) -> Dict[str, Any]:
        scoring: ScoringResult = self.scorer.score(
            host=frame.host,
            interface=frame.interface,
            tensor=frame.tensor,
        )

        # Threshold gate: skip the expensive RAG/LLM call when the
        # model says everything is fine. The NOC UI only sees
        # baselines as a no-op summary.
        if scoring.peak_anomaly() < self.config.model.anomaly_threshold:
            LOG.info(
                "Below threshold (%.2f < %.2f) for %s/%s; skipping RAG",
                scoring.peak_anomaly(),
                self.config.model.anomaly_threshold,
                frame.host,
                frame.interface,
            )
            return {
                "copilot_unavailable": False,
                "no_action": True,
                "alert_id": None,
                "host": frame.host,
                "interface": frame.interface,
                "anomaly_prob": scoring.peak_anomaly(),
                "forecast_peak": scoring.peak_forecast(),
                "ts": datetime.now(timezone.utc).isoformat(),
            }

        alert = build_alert_payload(
            scoring=scoring,
            frame=frame,
            model_name=self.config.llm.model_name,
            model_version=self._model_version(),
            sampler_name=self.config.sampler.backend,
            sequence_length=self.config.sampler.sequence_length,
            num_features=self.config.sampler.num_features,
            tti_minutes_min=self.config.model.tti_minutes_min,
            tti_minutes_max=self.config.model.tti_minutes_max,
        )
        validate_alert_payload(alert)

        try:
            evidence_envelope = self.rag.retrieve(alert)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("RAG retrieve failed: %s", exc)
            return self._unavailable("rag_failure", alert, exc)

        try:
            messages = self.prompts.build(alert, evidence_envelope)
        except Exception as exc:  # noqa: BLE001
            return self._unavailable("prompt_assembly_failure", alert, exc)

        try:
            llm_text = self.llm.chat(messages)
        except OfflineLLMUnavailable as exc:
            return self._unavailable("llm_unavailable", alert, exc)

        try:
            return self.gate.accept(llm_text, evidence_envelope)
        except Exception as exc:  # noqa: BLE001
            return self._unavailable("validator_failure", alert, exc)

    # ── helpers ────────────────────────────────────────────────

    def _unavailable(
        self, reason: str, alert: Dict[str, Any], exc: BaseException
    ) -> Dict[str, Any]:
        return {
            "copilot_unavailable": True,
            "reason": reason,
            "alert_id": alert.get("alert_id"),
            "error": repr(exc),
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    def _model_version(self) -> str:
        return self.scorer.checkpoint_sha256()[:12]

    def _publish(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, sort_keys=True)
        if self._sink_path:
            try:
                self._sink_path.parent.mkdir(parents=True, exist_ok=True)
                with self._sink_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError as exc:
                LOG.error("Sink write failed: %s", exc)
        if self._latest_path:
            try:
                self._latest_path.parent.mkdir(parents=True, exist_ok=True)
                self._latest_path.write_text(line, encoding="utf-8")
            except OSError as exc:
                LOG.error("Latest write failed: %s", exc)
        if not self._sink_path and not self._latest_path:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while not self._stop and time.monotonic() < end:
            time.sleep(min(0.5, max(0.0, end - time.monotonic())))


# ── CLI entry point ───────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Air-Gapped NOC Copilot - Phase 3 orchestrator. sample -> score -> RAG -> LLM -> validate."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "controller_config.yaml",
        help="Path to the controller YAML config",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Override config.max_cycles for one-shot runs",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one cycle and exit (alias for --max-cycles 1)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.once:
        cfg.max_cycles = 1
    if args.max_cycles is not None:
        cfg.max_cycles = args.max_cycles

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    LOG.info("Loaded config from %s", args.config)

    orch = Orchestrator(cfg)
    orch.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
