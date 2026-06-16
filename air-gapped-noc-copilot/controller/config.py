"""YAML-driven configuration for the orchestrator controller.

The orchestrator is configured by a single YAML file (default:
``controller/controller_config.yaml``). All sub-components receive a
*typed* dataclass view of the same file so each layer can validate its
own slice without re-parsing.

A copy-pasteable default lives at ``controller/controller_config.yaml``.

Environment variable overrides
-------------------------------
When running inside Docker the YAML file may not know the container
network addresses.  The loader checks for ``ORCHESTRATOR_*`` env vars
and applies them after the YAML merge.  The mapping is::

    ORCHESTRATOR_DB_HOST           -> sampler.db_host
    ORCHESTRATOR_DB_PORT           -> sampler.db_port  (int)
    ORCHESTRATOR_DB_NAME           -> sampler.db_name
    ORCHESTRATOR_DB_USER           -> sampler.db_user
    ORCHESTRATOR_DB_PASSWORD       -> sampler.db_password
    ORCHESTRATOR_PROMETHEUS_URL    -> sampler.prometheus_url
    ORCHESTRATOR_OLLAMA_URL        -> llm.ollama_url
    ORCHESTRATOR_OLLAMA_MODEL      -> llm.ollama_model
    ORCHESTRATOR_LLM_BACKEND       -> llm.backend
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class SamplerConfig:
    backend: str = "timescaledb"  # "timescaledb" | "prometheus"
    # The model expects a (sequence_length, num_features) tensor for each
    # (host, interface) pair. ``tick_seconds`` is the wall-clock
    # granularity of the source telemetry; ``sequence_length`` is the
    # lookback window.
    sequence_length: int = 60
    num_features: int = 25
    tick_seconds: int = 10
    poll_timeout_seconds: int = 5

    # Targets to monitor. If empty, the sampler enumerates every active
    # (host, interface) pair present in the last ``active_window_seconds``
    # of telemetry.
    target_hosts: List[str] = field(default_factory=list)
    target_interfaces: List[str] = field(default_factory=list)
    active_window_seconds: int = 300

    # TimescaleDB connection (used when backend == "timescaledb").
    db_host: str = "127.0.0.1"
    db_port: int = 5432
    db_name: str = "noc_telemetry"
    db_user: str = "postgres"
    db_password: str = "noc_copilot_db"
    db_connect_timeout: int = 5

    # Prometheus connection (used when backend == "prometheus").
    prometheus_url: str = "http://127.0.0.1:9090"
    prometheus_query_chunk_seconds: int = 600


@dataclass
class ModelConfig:
    # Which trained model artefact to load. The path is resolved
    # relative to the project root.
    model_path: str = "best_model.pth"
    architecture: str = "lstm_multitask"  # "lstm_multitask" | "tcn" | "hybrid"
    device: str = "cpu"

    # The scaler that was fit during training *must* be reused at
    # inference. The data_preprocessor saves it next to the checkpoint
    # as ``<model_path>.scalers.pkl``.
    scaler_path: str = "best_model.pth.scalers.pkl"

    # Decision thresholds.
    anomaly_threshold: float = 0.5      # 0..1  — flag an alert above this
    severity_high_threshold: float = 0.85
    severity_medium_threshold: float = 0.65
    tti_minutes_min: int = 0
    tti_minutes_max: int = 1440

    # Warm-up: a (host, interface) pair must have at least
    # ``sequence_length`` ticks before the first score is published.
    warmup_cycles: int = 1


@dataclass
class RagConfig:
    config_path: str = "m3/rag/rag_config.yaml"
    index_root: str = "m3/rag/index"
    top_k: int = 8
    context_token_budget: int = 3500

    # Filters applied to the Chroma `where` clause, in addition to the
    # alert-derived ones. Site / device / protocol from the alert win
    # when both are set.
    default_filters: Dict[str, str] = field(default_factory=dict)


@dataclass
class LLMConfig:
    # Backend: "openai" (llama.cpp/vLLM/Ollama-compat) or "ollama" (native)
    backend: str = "openai"
    base_url: str = "http://127.0.0.1:8080"
    api_path: str = "/v1/chat/completions"
    model_name: str = "airgap-noc"
    request_timeout_seconds: int = 60
    max_retries: int = 1
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 1024
    # Ollama-specific (used when backend == "ollama")
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3:latest"
    ollama_num_ctx: int = 4096
    ollama_num_gpu: int = -1


@dataclass
class SinkConfig:
    # Where to write the validated NOC-ready response (one JSON per
    # line). If empty, the response is printed to stdout.
    ndjson_path: str = ""
    # A sidecar file that always reflects the most recent cycle, so an
    # external NOC UI can poll a single file. Empty disables.
    latest_path: str = ""


@dataclass
class OrchestratorConfig:
    sample_interval_seconds: int = 30
    max_cycles: int = 0  # 0 = run forever; >0 = stop after N cycles
    log_level: str = "INFO"
    fail_loud: bool = True
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    sink: SinkConfig = field(default_factory=SinkConfig)

    @property
    def project_root(self) -> Path:
        # The default config file is always co-located with the
        # controller package; ``config_path`` is resolved relative to
        # the *project root* (one level up from the controller dir).
        return Path(__file__).resolve().parent.parent

    def resolve(self, raw: str) -> Path:
        if not raw:
            return Path("")
        p = Path(raw)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()


def load_config(path: Optional[Path] = None) -> OrchestratorConfig:
    """Load the YAML config and return a typed :class:`OrchestratorConfig`.

    Parameters
    ----------
    path:
        Path to the YAML file. ``None`` loads the default
        ``controller/controller_config.yaml`` next to this package.
    """
    if path is None:
        path = Path(__file__).parent / "controller_config.yaml"
    cfg = OrchestratorConfig()
    if not path.exists():
        _apply_env_overrides(cfg)
        return cfg
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _merge(cfg, raw)
    _apply_env_overrides(cfg)
    return cfg


def _merge(cfg: OrchestratorConfig, raw: Dict[str, Any]) -> None:
    for section in ("sampler", "model", "rag", "llm", "sink"):
        if section not in raw:
            continue
        sub = getattr(cfg, section)
        for k, v in raw[section].items():
            if hasattr(sub, k):
                setattr(sub, k, v)
    for k, v in raw.items():
        if k in ("sampler", "model", "rag", "llm", "sink"):
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)


_ENV_MAP = {
    "ORCHESTRATOR_DB_HOST": ("sampler", "db_host"),
    "ORCHESTRATOR_DB_PORT": ("sampler", "db_port"),
    "ORCHESTRATOR_DB_NAME": ("sampler", "db_name"),
    "ORCHESTRATOR_DB_USER": ("sampler", "db_user"),
    "ORCHESTRATOR_DB_PASSWORD": ("sampler", "db_password"),
    "ORCHESTRATOR_PROMETHEUS_URL": ("sampler", "prometheus_url"),
    "ORCHESTRATOR_OLLAMA_URL": ("llm", "ollama_url"),
    "ORCHESTRATOR_OLLAMA_MODEL": ("llm", "ollama_model"),
    "ORCHESTRATOR_LLM_BACKEND": ("llm", "backend"),
    "ORCHESTRATOR_LLM_BASE_URL": ("llm", "base_url"),
    "ORCHESTRATOR_LLM_MODEL": ("llm", "model_name"),
    "ORCHESTRATOR_SAMPLE_INTERVAL": (None, "sample_interval_seconds"),
    "ORCHESTRATOR_MAX_CYCLES": (None, "max_cycles"),
    "ORCHESTRATOR_LOG_LEVEL": (None, "log_level"),
}


def _apply_env_overrides(cfg: OrchestratorConfig) -> None:
    """Override config values from environment variables.

    Only sets a value when the env var is non-empty, so the YAML
    defaults win when the container omits the variable.
    """
    _INT_FIELDS = {"db_port", "max_retries", "request_timeout_seconds",
                   "max_tokens", "ollama_num_ctx", "ollama_num_gpu",
                   "sequence_length", "num_features", "tick_seconds",
                   "poll_timeout_seconds", "active_window_seconds",
                   "db_connect_timeout", "prometheus_query_chunk_seconds",
                   "top_k", "context_token_budget", "sample_interval_seconds",
                   "max_cycles"}
    _FLOAT_FIELDS = {"temperature", "top_p", "anomaly_threshold",
                     "severity_high_threshold", "severity_medium_threshold"}
    _BOOL_FIELDS = {"fail_loud"}

    for env_key, (section, field_name) in _ENV_MAP.items():
        val = os.environ.get(env_key, "").strip()
        if not val:
            continue
        if section is None:
            target = cfg
        else:
            target = getattr(cfg, section)
        if field_name in _INT_FIELDS:
            val = int(val)
        elif field_name in _FLOAT_FIELDS:
            val = float(val)
        elif field_name in _BOOL_FIELDS:
            val = val.lower() in ("true", "1", "yes")
        setattr(target, field_name, val)
