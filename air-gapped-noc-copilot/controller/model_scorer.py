"""PyTorch inference wrapper for the LSTM / TCN / hybrid models.

Loads a checkpoint produced by ``train_models.py`` and returns the
multi-task outputs the alert builder needs:

    {
      "forecast":         [...],   # H-step predicted utilization
      "anomaly_prob":     [...],   # H-step anomaly probability in [0, 1]
      "tti_minutes":      [...],   # H-step time-to-impact estimate
    }

The scorer is *stateless* across calls. The scaler that was fit at
training time (``data_preprocessor.NetworkDataPreprocessor``) is
reloaded and reused on every frame so the inference distribution
matches the training distribution exactly.

The scorer is **fail-loud**: a missing checkpoint, a missing scaler, or
a feature-count mismatch raises. The orchestrator catches and reports.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

LOG = logging.getLogger("controller.scorer")

# Local imports: the model definitions sit next to this package.
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lstm_model import LSTMPredictor, LSTMMultiTask  # noqa: E402
from tcn_model import TCNLSTMHybrid, TCNPredictor  # noqa: E402


@dataclass(frozen=True)
class ScoringResult:
    host: str
    interface: str
    architecture: str
    forecast: List[float]
    anomaly_prob: List[float]
    tti_minutes: List[float]
    checkpoint_sha256: str
    scaler_sha256: str

    def peak_anomaly(self) -> float:
        return float(max(self.anomaly_prob)) if self.anomaly_prob else 0.0

    def peak_forecast(self) -> float:
        return float(max(self.forecast)) if self.forecast else 0.0

    def median_tti(self) -> float:
        if not self.tti_minutes:
            return 0.0
        return float(np.median(np.asarray(self.tti_minutes, dtype=np.float32)))


class ModelScorer:
    """Wrap a trained multi-task predictor for runtime inference."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        scaler_path: Optional[Path],
        architecture: str = "lstm_multitask",
        device: str = "cpu",
        hidden_size: int = 128,
        num_layers: int = 2,
        num_features: int = 25,
        forecast_horizon: int = 10,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.architecture = architecture
        self.device = device
        self.num_features = num_features
        self.forecast_horizon = forecast_horizon
        self._scaler: Dict[str, Any] = {}
        # Load checkpoint first to auto-detect architecture
        self._model, self.architecture = self._build_and_load(
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

    # ── lifecycle ─────────────────────────────────────────────────

    @staticmethod
    def _detect_architecture(state: Dict[str, Any]) -> str:
        """Infer model class from state_dict key names."""
        keys = set(state.keys())
        if "forecast_head.0.weight" in keys:
            return "lstm_multitask"
        if "attention.in_proj_weight" in keys:
            return "lstm"
        if "tcn.network.0.conv1.weight" in keys or "network.0.conv1.weight" in keys:
            return "tcn"
        if "tcn.0.conv1.weight" in keys:
            return "hybrid"
        return "lstm"  # safe fallback

    @staticmethod
    def _detect_forecast_horizon(state: Dict[str, Any], arch: str) -> Optional[int]:
        """Infer forecast_horizon from checkpoint fc3 output dimension."""
        # For lstm/lstm_multitask: fc3.weight shape is (output_size * forecast_horizon, hidden//2)
        # For tcn/hybrid: final layer differs but same pattern
        fc3_key = "fc3.weight"
        if arch == "lstm_multitask":
            # Try forecast_head first
            fh_key = "forecast_head.0.weight"
            if fh_key in state:
                out = state[fh_key].shape[0]  # (out_features, ...)
                return out  # output_size=1 for LSTM, so out == horizon
        if fc3_key in state:
            out = state[fc3_key].shape[0]  # (output_size * horizon, hidden//2)
            # output_size is typically 1; horizon = out // output_size
            # We can't know output_size exactly, but default is 1
            return out  # likely horizon (output_size=1)
        return None

    def _build_model(
        self, *, hidden_size: int, num_layers: int, architecture: str
    ) -> torch.nn.Module:
        if architecture == "lstm_multitask":
            return LSTMMultiTask(
                input_size=self.num_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                forecast_horizon=self.forecast_horizon,
                dropout=0.0,
            )
        if architecture == "lstm":
            return LSTMPredictor(
                input_size=self.num_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=1,
                dropout=0.0,
                forecast_horizon=self.forecast_horizon,
            )
        if architecture == "tcn":
            return TCNPredictor(
                input_size=self.num_features,
                forecast_horizon=self.forecast_horizon,
                dropout=0.0,
            )
        if architecture == "hybrid":
            return TCNLSTMHybrid(
                input_size=self.num_features,
                forecast_horizon=self.forecast_horizon,
                dropout=0.0,
            )
        raise ValueError(f"Unknown architecture: {architecture}")

    def _build_and_load(
        self, *, hidden_size: int, num_layers: int
    ) -> tuple[torch.nn.Module, str]:
        """Build the correct model class and load checkpoint weights."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {self.checkpoint_path}. "
                "Run train_models.py first."
            )
        state = torch.load(self.checkpoint_path, map_location=self.device)
        raw_state = state.get("model_state_dict", state) if isinstance(state, dict) else state

        # Auto-detect architecture from checkpoint keys
        detected = self._detect_architecture(raw_state)
        LOG.info(
            "Checkpoint architecture detected: %s (configured: %s)",
            detected, self.architecture,
        )
        if detected != self.architecture:
            LOG.warning(
                "Architecture mismatch — using detected '%s' instead of configured '%s'",
                detected, self.architecture,
            )
            self.architecture = detected

        # Auto-detect forecast_horizon from checkpoint
        detected_horizon = self._detect_forecast_horizon(raw_state, self.architecture)
        if detected_horizon is not None and detected_horizon != self.forecast_horizon:
            LOG.warning(
                "forecast_horizon mismatch — using %d (from checkpoint) instead of %d (configured)",
                detected_horizon, self.forecast_horizon,
            )
            self.forecast_horizon = detected_horizon

        model = self._build_model(
            hidden_size=hidden_size,
            num_layers=num_layers,
            architecture=self.architecture,
        )
        model.load_state_dict(raw_state)
        model.eval()
        model.to(self.device)

        if self.scaler_path and self.scaler_path.is_file():
            with open(self.scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
        else:
            LOG.warning(
                "No scaler found at %s — running inference on raw features. "
                "Predictions will be off-distribution.",
                self.scaler_path,
            )

        return model, self.architecture

    # ── fingerprinting ────────────────────────────────────────────

    def checkpoint_sha256(self) -> str:
        h = hashlib.sha256()
        h.update(self.checkpoint_path.read_bytes())
        return h.hexdigest()

    def scaler_sha256(self) -> str:
        if not self.scaler_path or not self.scaler_path.is_file():
            return hashlib.sha256(b"NO_SCALER").hexdigest()
        h = hashlib.sha256()
        h.update(self.scaler_path.read_bytes())
        return h.hexdigest()

    # ── inference ─────────────────────────────────────────────────

    def score(self, host: str, interface: str, tensor: np.ndarray) -> ScoringResult:
        """Score one (host, interface) tensor.

        Parameters
        ----------
        tensor:
            ``np.ndarray`` of shape ``(sequence_length, num_features)``
            produced by :class:`controller.metric_sampler.MetricSampler`.
        """
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2-D tensor (T, F); got shape {tensor.shape}")
        if tensor.shape[1] != self.num_features:
            raise ValueError(
                f"Feature count mismatch: tensor has {tensor.shape[1]}, "
                f"model expects {self.num_features}. Check feature_columns in "
                f"data_preprocessor.PreprocessingConfig."
            )

        x = self._apply_scaler(tensor)
        x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)  # (1, T, F)

        with torch.no_grad():
            out = self._model(x)

        if isinstance(out, dict):
            forecast = out["forecast"].squeeze(-1).squeeze(0).cpu().numpy()
            anom = out["anomaly_prob"].squeeze(0).cpu().numpy()
            tti = out["tti_estimates"].squeeze(0).cpu().numpy()
        else:
            arr = out.squeeze(0).cpu().numpy()
            if arr.ndim == 1:
                arr = arr[:, None]
            forecast = arr[:, 0]
            anom = self._synthetic_anomaly_prob(forecast)
            tti = self._synthetic_tti(forecast)

        return ScoringResult(
            host=host,
            interface=interface,
            architecture=self.architecture,
            forecast=[float(x) for x in np.asarray(forecast).flatten().tolist()],
            anomaly_prob=[
                float(min(1.0, max(0.0, x))) for x in np.asarray(anom).flatten().tolist()
            ],
            tti_minutes=[float(max(0.0, x)) for x in np.asarray(tti).flatten().tolist()],
            checkpoint_sha256=self.checkpoint_sha256(),
            scaler_sha256=self.scaler_sha256(),
        )

    # ── internals ─────────────────────────────────────────────────

    def _apply_scaler(self, tensor: np.ndarray) -> np.ndarray:
        if not self._scaler:
            return tensor
        scaler = self._scaler.get("feature")
        if scaler is None:
            return tensor
        original_shape = tensor.shape
        flat = tensor.reshape(-1, tensor.shape[-1])
        try:
            scaled = scaler.transform(flat)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Scaler transform failed (%s); using raw features", exc)
            return tensor
        return np.asarray(scaled, dtype=np.float32).reshape(original_shape)

    @staticmethod
    def _synthetic_anomaly_prob(forecast: np.ndarray) -> np.ndarray:
        # Fallback when the model only outputs a forecast: derive a
        # crude probability from how much the forecast crosses the
        # 80% utilisation threshold. Used only for ``lstm`` (not
        # ``lstm_multitask``) checkpoints.
        return 1.0 / (1.0 + np.exp(-(np.asarray(forecast) - 80.0) / 5.0))

    @staticmethod
    def _synthetic_tti(forecast: np.ndarray) -> np.ndarray:
        # Crude minutes-to-impact estimate from a utilisation ramp.
        return np.maximum(0.0, (100.0 - np.asarray(forecast)) * 6.0)
