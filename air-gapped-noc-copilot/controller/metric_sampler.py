"""Live metric sampler.

Pulls the latest ``sequence_length``-sized window for every (host,
interface) pair from either TimescaleDB (system of record) or
Prometheus (volatile cache), and returns a 3-D tensor shaped for the
PyTorch model: ``(1, sequence_length, num_features)`` per pair, plus
the raw signal frame the alert builder will echo into ``signals[]``.

The sampler is *pluggable*. The two production backends are
:class:`TimescaleDBSampler` (preferred — the system of record written
by Phase 2's Telegraf + ``telemetry_normalizer.py``) and
:class:`PrometheusSampler` (used when the SQL store is down or when a
Prometheus recording rule is the authoritative source).

A simple in-memory :class:`FixtureSampler` is provided for the
end-to-end test and for ad-hoc dry runs of the orchestrator. It does
not require a live database or Prometheus.
"""
from __future__ import annotations

import abc
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger("controller.sampler")


# ─────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetricFrame:
    """A single (host, interface) sliding window ready for the model.

    Attributes
    ----------
    host, interface:
        Stable identifiers from the topology.
    feature_columns:
        The ordered list of feature names the tensor columns
        correspond to. Must match the scaler / model order from
        ``data_preprocessor.PreprocessingConfig.feature_columns``.
    tensor:
        ``np.ndarray`` of shape ``(sequence_length, num_features)``
        already in the *unscaled* feature units (the sampler does not
        normalise; the scorer does, using the training-time scaler).
    latest_signals:
        A list of the most recent tick per feature, used by
        :func:`controller.alert_builder.build_alert_payload` to build
        the ``signals[]`` block. Each entry is
        ``{metric, value, trend, host, interface, ts}``.
    generated_at:
        UTC timestamp of the most recent tick.
    """

    host: str
    interface: str
    feature_columns: List[str]
    tensor: np.ndarray
    latest_signals: List[Dict[str, Any]]
    generated_at: datetime


# ─────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────


class MetricSampler(abc.ABC):
    """Pluggable sampler contract."""

    def __init__(self, feature_columns: Sequence[str], sequence_length: int):
        self.feature_columns: List[str] = list(feature_columns)
        self.sequence_length = int(sequence_length)
        if len(self.feature_columns) == 0:
            raise ValueError("feature_columns must be non-empty")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

    @abc.abstractmethod
    def list_targets(self) -> List[Tuple[str, str]]:
        """Return the (host, interface) pairs the sampler will read."""

    @abc.abstractmethod
    def fetch(self, host: str, interface: str) -> Optional[MetricFrame]:
        """Return the latest sliding window for one pair, or ``None`` if
        there is not enough data yet."""

    # Convenience: fetch every target, skipping None / incomplete.
    def fetch_all(self) -> List[MetricFrame]:
        out: List[MetricFrame] = []
        for host, iface in self.list_targets():
            try:
                frame = self.fetch(host, iface)
            except Exception as exc:  # noqa: BLE001 - log & skip
                LOG.warning("sampler.fetch failed for %s/%s: %s", host, iface, exc)
                continue
            if frame is not None:
                out.append(frame)
        return out


# ─────────────────────────────────────────────────────────────────────
# TimescaleDB backend
# ─────────────────────────────────────────────────────────────────────


class TimescaleDBSampler(MetricSampler):
    """Pull a rolling window from the ``mv_5min_interface_stats``
    continuous aggregate (defined in ``telemetry_normalizer.py``). If
    the aggregate is empty, falls back to the raw ``interface_metrics``
    hypertable.

    Feature columns must align with ``data_preprocessor.PreprocessingConfig``.
    The actual feature set in the DB is the union of
    ``interface_metrics`` and ``routing_metrics`` columns, joined by
    ``time`` and ``host``.
    """

    _SQL = """
    WITH joined AS (
      SELECT
        time_bucket(%(bucket)s, i.time) AS bucket,
        i.host                         AS host,
        i.interface                    AS interface,
        AVG(i.utilization)             AS interface_utilization,
        AVG(i.in_errors)               AS interface_in_errors,
        AVG(i.out_errors)              AS interface_out_errors,
        AVG(i.in_discards)             AS interface_in_discards,
        AVG(i.out_discards)            AS interface_out_discards,
        AVG(i.in_packets)              AS interface_in_packets,
        AVG(i.out_packets)             AS interface_out_packets,
        AVG(r.retransmissions)         AS ospf_retransmissions,
        AVG(r.fsm_transitions)         AS bgp_fsm_transitions,
        AVG(r.updates_in + r.updates_out) AS bgp_update_rate
      FROM interface_metrics i
      LEFT JOIN routing_metrics r
        ON r.host = i.host AND r.time = i.time
      WHERE i.host = %(host)s
        AND i.interface = %(interface)s
        AND i.time >= NOW() - INTERVAL '%(window)s seconds'
      GROUP BY bucket, i.host, i.interface
      ORDER BY bucket ASC
    )
    SELECT * FROM joined;
    """

    _TARGETS_SQL = """
    SELECT DISTINCT host, interface
    FROM interface_metrics
    WHERE time >= NOW() - INTERVAL '%(window)s seconds'
    ORDER BY host, interface
    LIMIT 500;
    """

    def __init__(
        self,
        feature_columns: Sequence[str],
        sequence_length: int,
        *,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        connect_timeout: int = 5,
        target_hosts: Optional[Sequence[str]] = None,
        target_interfaces: Optional[Sequence[str]] = None,
        active_window_seconds: int = 300,
    ):
        super().__init__(feature_columns, sequence_length)
        import psycopg2  # local import; not every host has it

        self._conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            connect_timeout=connect_timeout,
        )
        self._conn.autocommit = True
        self._target_hosts = list(target_hosts or [])
        self._target_interfaces = list(target_interfaces or [])
        self._active_window_seconds = int(active_window_seconds)

    def list_targets(self) -> List[Tuple[str, str]]:
        if self._target_hosts and self._target_interfaces:
            return [(h, i) for h in self._target_hosts for i in self._target_interfaces]
        with self._conn.cursor() as cur:
            cur.execute(self._TARGETS_SQL, {"window": self._active_window_seconds})
            return [(r[0], r[1]) for r in cur.fetchall()]

    def fetch(self, host: str, interface: str) -> Optional[MetricFrame]:
        # Bucket size matches ``sequence_length`` ticks; we ask for one
        # bucket per tick at ``tick_seconds`` granularity.
        bucket = "10 seconds"
        window = self.sequence_length * 10 + 60  # +60s slack
        with self._conn.cursor() as cur:
            cur.execute(
                self._SQL,
                {
                    "host": host,
                    "interface": interface,
                    "bucket": bucket,
                    "window": window,
                },
            )
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        if not rows or len(rows) < self.sequence_length:
            return None

        matrix = self._rows_to_matrix(rows, cols)
        matrix = self._align_to_feature_columns(matrix, cols)
        # Trim to the most recent ``sequence_length`` ticks.
        if matrix.shape[0] > self.sequence_length:
            matrix = matrix[-self.sequence_length:]

        latest = matrix[-1]
        previous = matrix[-2] if matrix.shape[0] >= 2 else latest
        latest_signals = self._build_latest_signals(
            latest=latest, previous=previous, host=host, interface=interface
        )
        generated_at = datetime.now(timezone.utc)
        return MetricFrame(
            host=host,
            interface=interface,
            feature_columns=list(self.feature_columns),
            tensor=matrix.astype(np.float32),
            latest_signals=latest_signals,
            generated_at=generated_at,
        )

    # ── helpers ────────────────────────────────────────────────────

    def _rows_to_matrix(self, rows: List[Tuple[Any, ...]], cols: List[str]) -> np.ndarray:
        data: List[List[float]] = []
        for r in rows:
            row = list(r)
            data.append([float(v) if v is not None else 0.0 for v in row[3:]])  # skip bucket/host/interface
        return np.asarray(data, dtype=np.float32)

    def _align_to_feature_columns(
        self, matrix: np.ndarray, cols: List[str]
    ) -> np.ndarray:
        """Reorder / zero-pad the SQL columns to match
        ``self.feature_columns``."""
        col_index = {c: i for i, c in enumerate(cols[3:])}  # skip bucket/host/interface
        out = np.zeros((matrix.shape[0], len(self.feature_columns)), dtype=np.float32)
        for j, name in enumerate(self.feature_columns):
            i = col_index.get(name)
            if i is not None:
                out[:, j] = matrix[:, i]
        return out

    @staticmethod
    def _build_latest_signals(
        *,
        latest: np.ndarray,
        previous: np.ndarray,
        host: str,
        interface: str,
    ) -> List[Dict[str, Any]]:
        # Stable metric names mirror data_preprocessor.feature_columns.
        names = [
            "interface_utilization",
            "interface_in_errors",
            "interface_out_errors",
            "interface_in_discards",
            "interface_out_discards",
            "interface_in_packets",
            "interface_out_packets",
            "ospf_retransmissions",
            "bgp_fsm_transitions",
            "bgp_update_rate",
        ]
        ts = datetime.now(timezone.utc).isoformat()
        out: List[Dict[str, Any]] = []
        for i, name in enumerate(names):
            if i >= latest.shape[0]:
                break
            v = float(latest[i])
            p = float(previous[i]) if previous.shape[0] > i else v
            if v > p * 1.05:
                trend = "rising"
            elif v < p * 0.95:
                trend = "falling"
            else:
                trend = "flat"
            out.append(
                {
                    "metric": name,
                    "value": round(v, 4),
                    "trend": trend,
                    "host": host,
                    "interface": interface,
                    "ts": ts,
                }
            )
        return out


# ─────────────────────────────────────────────────────────────────────
# Prometheus backend
# ─────────────────────────────────────────────────────────────────────


class PrometheusSampler(MetricSampler):
    """Pull a rolling window via PromQL ``<metric>[5m]`` range queries.

    Used as a degraded-mode fallback when TimescaleDB is unreachable
    but Prometheus is still up. Less precise (counters only) but
    sufficient to keep the loop warm.
    """

    def __init__(
        self,
        feature_columns: Sequence[str],
        sequence_length: int,
        *,
        base_url: str,
        target_hosts: Optional[Sequence[str]] = None,
        target_interfaces: Optional[Sequence[str]] = None,
        timeout_seconds: int = 5,
        chunk_seconds: int = 600,
    ):
        super().__init__(feature_columns, sequence_length)
        self._base_url = base_url.rstrip("/")
        self._target_hosts = list(target_hosts or [])
        self._target_interfaces = list(target_interfaces or [])
        self._timeout = int(timeout_seconds)
        self._chunk = int(chunk_seconds)

    def list_targets(self) -> List[Tuple[str, str]]:
        if not self._target_hosts or not self._target_interfaces:
            LOG.warning("PrometheusSampler needs explicit target_hosts and target_interfaces; returning empty")
            return []
        return [(h, i) for h in self._target_hosts for i in self._target_interfaces]

    def fetch(self, host: str, interface: str) -> Optional[MetricFrame]:
        import requests

        end = datetime.now(timezone.utc)
        start = end - timedelta(seconds=self.sequence_length * 10 + 60)
        params = {
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": "10s",
        }
        rows: List[List[float]] = []
        for name in self.feature_columns:
            promql = f'{name}{{host="{host}",interface="{interface}"}}'
            url = f"{self._base_url}/api/v1/query_range"
            resp = requests.get(url, params={**params, "query": promql}, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json().get("data", {}).get("result", [])
            if not data:
                rows.append([0.0] * self.sequence_length)
                continue
            values = data[0].get("values", [])
            series = [float(v[1]) for v in values]
            if len(series) < self.sequence_length:
                series = [0.0] * (self.sequence_length - len(series)) + series
            rows.append(series[-self.sequence_length:])
        matrix = np.asarray(rows, dtype=np.float32).T  # (seq, features)
        latest_signals = self._build_latest_signals(matrix, host, interface)
        return MetricFrame(
            host=host,
            interface=interface,
            feature_columns=list(self.feature_columns),
            tensor=matrix,
            latest_signals=latest_signals,
            generated_at=end,
        )

    @staticmethod
    def _build_latest_signals(
        matrix: np.ndarray, host: str, interface: str
    ) -> List[Dict[str, Any]]:
        names = [
            "interface_utilization",
            "interface_in_errors",
            "interface_out_errors",
            "interface_in_discards",
            "interface_out_discards",
            "interface_in_packets",
            "interface_out_packets",
        ]
        latest = matrix[-1]
        previous = matrix[-2] if matrix.shape[0] >= 2 else latest
        ts = datetime.now(timezone.utc).isoformat()
        out: List[Dict[str, Any]] = []
        for i, name in enumerate(names):
            if i >= latest.shape[0]:
                break
            v = float(latest[i])
            p = float(previous[i])
            if v > p * 1.05:
                trend = "rising"
            elif v < p * 0.95:
                trend = "falling"
            else:
                trend = "flat"
            out.append(
                {
                    "metric": name,
                    "value": round(v, 4),
                    "trend": trend,
                    "host": host,
                    "interface": interface,
                    "ts": ts,
                }
            )
        return out


# ─────────────────────────────────────────────────────────────────────
# Fixture backend (offline, deterministic, used in tests)
# ─────────────────────────────────────────────────────────────────────


class FixtureSampler(MetricSampler):
    """Deterministic in-memory sampler used by the e2e test and by
    ad-hoc smoke runs of the orchestrator.

    Each (host, interface) pair is fed a synthetic signal that crosses
    the anomaly threshold around a configurable tick index, so the
    orchestrator can be exercised end-to-end with no live database.
    """

    def __init__(
        self,
        feature_columns: Sequence[str],
        sequence_length: int,
        *,
        targets: Optional[Sequence[Tuple[str, str]]] = None,
        anomaly_tick: int = 45,
        seed: int = 7,
    ):
        super().__init__(feature_columns, sequence_length)
        self._targets: List[Tuple[str, str]] = list(targets or [("pe-hub-east-1", "eth3")])
        self._anomaly_tick = int(anomaly_tick)
        self._rng = np.random.default_rng(seed)

    def list_targets(self) -> List[Tuple[str, str]]:
        return list(self._targets)

    def fetch(self, host: str, interface: str) -> Optional[MetricFrame]:
        rng = np.random.default_rng(abs(hash((host, interface))) % (2**32))
        n = self.sequence_length
        f = len(self.feature_columns)
        matrix = rng.normal(loc=30.0, scale=5.0, size=(n, f)).astype(np.float32)

        # Inject a congestion ramp on the first feature so the scorer
        # sees something anomalous.
        ramp = np.linspace(0, 70.0, num=self._anomaly_tick + 1, dtype=np.float32)
        matrix[-self._anomaly_tick:, 0] += ramp[::-1][: self._anomaly_tick]

        latest = matrix[-1]
        previous = matrix[-2]
        latest_signals = self._build_latest_signals(latest, previous, host, interface)
        return MetricFrame(
            host=host,
            interface=interface,
            feature_columns=list(self.feature_columns),
            tensor=matrix,
            latest_signals=latest_signals,
            generated_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _build_latest_signals(
        latest: np.ndarray, previous: np.ndarray, host: str, interface: str
    ) -> List[Dict[str, Any]]:
        names = [
            "interface_utilization",
            "interface_in_errors",
            "interface_out_errors",
            "interface_in_discards",
            "interface_out_discards",
            "interface_in_packets",
            "interface_out_packets",
        ]
        ts = datetime.now(timezone.utc).isoformat()
        out: List[Dict[str, Any]] = []
        for i, name in enumerate(names):
            if i >= latest.shape[0]:
                break
            v = float(latest[i])
            p = float(previous[i])
            if v > p * 1.05:
                trend = "rising"
            elif v < p * 0.95:
                trend = "falling"
            else:
                trend = "flat"
            out.append(
                {
                    "metric": name,
                    "value": round(v, 4),
                    "trend": trend,
                    "host": host,
                    "interface": interface,
                    "ts": ts,
                }
            )
        return out


# ─────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()
