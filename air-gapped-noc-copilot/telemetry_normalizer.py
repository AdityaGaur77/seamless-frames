#!/usr/bin/env python3
"""
Air-Gapped NOC Copilot - Telemetry Stream Normalizer
Aligns and normalizes SNMP, NetFlow, and Syslog streams into
a unified time-series dataset for Prometheus/TimescaleDB.
"""

import json
import time
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values, Json
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, push_to_gateway

# ── Logging Configuration ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("telemetry_normalizer")


# ════════════════════════════════════════════════════════════════
# Data Models
# ════════════════════════════════════════════════════════════════

@dataclass
class NormalizedMetric:
    """Normalized metric data structure."""
    timestamp: datetime
    source_host: str
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    source_type: str = ""  # snmp, netflow, syslog


@dataclass
class InterfaceMetrics:
    """Aggregated interface metrics."""
    host: str
    interface: str
    timestamp: datetime
    in_octets: float = 0.0
    out_octets: float = 0.0
    in_errors: float = 0.0
    out_errors: float = 0.0
    in_discards: float = 0.0
    out_discards: float = 0.0
    in_packets: float = 0.0
    out_packets: float = 0.0
    speed: float = 0.0
    admin_status: int = 0
    oper_status: int = 0
    utilization: float = 0.0


@dataclass
class RoutingMetrics:
    """Aggregated routing protocol metrics."""
    host: str
    timestamp: datetime
    protocol: str  # ospf, bgp, mpls_ldp
    neighbor: str = ""
    state: str = ""
    state_changes: int = 0
    retransmissions: int = 0
    updates_in: int = 0
    updates_out: int = 0
    fsm_transitions: int = 0


@dataclass
class IPSecMetrics:
    """Aggregated IPSec tunnel metrics."""
    host: str
    tunnel_id: str
    timestamp: datetime
    local_addr: str = ""
    remote_addr: str = ""
    encrypted_pkts: float = 0.0
    decrypted_pkts: float = 0.0
    in_errors: float = 0.0
    out_errors: float = 0.0


@dataclass
class SyslogEvent:
    """Normalized syslog event."""
    timestamp: datetime
    source_host: str
    facility: int
    severity: int
    app_name: str
    message: str
    tags: Dict[str, str] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════
# Database Schema Manager
# ════════════════════════════════════════════════════════════════

class TimescaleDBManager:
    """Manages TimescaleDB connection and schema setup."""

    def __init__(self, host: str, port: int, dbname: str, user: str, password: str):
        self.conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=user, password=password
        )
        self.conn.autocommit = True
        self._setup_schema()

    def _setup_schema(self):
        """Create hypertables for time-series data."""
        cursor = self.conn.cursor()
        
        # Enable TimescaleDB extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
        
        # ── Interface Metrics Table ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interface_metrics (
                time TIMESTAMPTZ NOT NULL,
                host TEXT NOT NULL,
                interface TEXT NOT NULL,
                in_octets DOUBLE PRECISION,
                out_octets DOUBLE PRECISION,
                in_errors DOUBLE PRECISION,
                out_errors DOUBLE PRECISION,
                in_discards DOUBLE PRECISION,
                out_discards DOUBLE PRECISION,
                in_packets DOUBLE PRECISION,
                out_packets DOUBLE PRECISION,
                speed DOUBLE PRECISION,
                utilization DOUBLE PRECISION,
                admin_status INTEGER,
                oper_status INTEGER
            );
        """)
        
        # ── Routing Metrics Table ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS routing_metrics (
                time TIMESTAMPTZ NOT NULL,
                host TEXT NOT NULL,
                protocol TEXT NOT NULL,
                neighbor TEXT,
                state TEXT,
                state_changes INTEGER,
                retransmissions INTEGER,
                updates_in INTEGER,
                updates_out INTEGER,
                fsm_transitions INTEGER
            );
        """)
        
        # ── IPSec Metrics Table ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ipsec_metrics (
                time TIMESTAMPTZ NOT NULL,
                host TEXT NOT NULL,
                tunnel_id TEXT NOT NULL,
                local_addr TEXT,
                remote_addr TEXT,
                encrypted_pkts DOUBLE PRECISION,
                decrypted_pkts DOUBLE PRECISION,
                in_errors DOUBLE PRECISION,
                out_errors DOUBLE PRECISION
            );
        """)
        
        # ── Syslog Events Table ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS syslog_events (
                time TIMESTAMPTZ NOT NULL,
                host TEXT NOT NULL,
                facility INTEGER,
                severity INTEGER,
                app_name TEXT,
                message TEXT,
                tags JSONB
            );
        """)
        
        # ── Normalized Metrics Table ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS normalized_metrics (
                time TIMESTAMPTZ NOT NULL,
                source_host TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value DOUBLE PRECISION,
                unit TEXT,
                tags JSONB,
                source_type TEXT
            );
        """)
        
        # ── Create Hypertables ──
        hypertables = [
            "interface_metrics",
            "routing_metrics",
            "ipsec_metrics",
            "syslog_events",
            "normalized_metrics",
        ]
        
        for table in hypertables:
            try:
                cursor.execute(f"""
                    SELECT create_hypertable('{table}', 'time',
                        if_not_exists => TRUE,
                        chunk_time_interval => INTERVAL '1 hour'
                    );
                """)
            except Exception as e:
                logger.warning(f"Hypertable creation for {table}: {e}")
        
        # ── Create Indexes ──
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_interface_host_interface 
            ON interface_metrics (host, interface, time DESC);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_routing_host_protocol 
            ON routing_metrics (host, protocol, time DESC);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ipsec_host_tunnel 
            ON ipsec_metrics (host, tunnel_id, time DESC);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_syslog_host_severity 
            ON syslog_events (host, severity, time DESC);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_normalized_name_time 
            ON normalized_metrics (metric_name, time DESC);
        """)
        
        # ── Continuous Aggregates for ML Training ──
        try:
            cursor.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS mv_5min_interface_stats
                WITH (timescaledb.continuous) AS
                SELECT
                    time_bucket('5 minutes', time) AS bucket,
                    host,
                    interface,
                    AVG(utilization) AS avg_utilization,
                    MAX(utilization) AS max_utilization,
                    SUM(in_errors) AS total_in_errors,
                    SUM(out_errors) AS total_out_errors,
                    SUM(in_discards) AS total_in_discards,
                    SUM(out_discards) AS total_out_discards,
                    AVG(in_octets) AS avg_in_octets,
                    AVG(out_octets) AS avg_out_octets
                FROM interface_metrics
                GROUP BY bucket, host, interface
                WITH NO DATA;
            """)
        except Exception as e:
            logger.warning(f"Materialized view creation: {e}")
        
        cursor.close()
        logger.info("TimescaleDB schema initialized")

    def insert_interface_metrics(self, metrics: List[InterfaceMetrics]):
        """Insert interface metrics into TimescaleDB."""
        cursor = self.conn.cursor()
        values = [
            (
                m.timestamp, m.host, m.interface,
                m.in_octets, m.out_octets,
                m.in_errors, m.out_errors,
                m.in_discards, m.out_discards,
                m.in_packets, m.out_packets,
                m.speed, m.utilization,
                m.admin_status, m.oper_status,
            )
            for m in metrics
        ]
        execute_values(
            cursor,
            """
            INSERT INTO interface_metrics
            (time, host, interface, in_octets, out_octets, in_errors, out_errors,
             in_discards, out_discards, in_packets, out_packets, speed, utilization,
             admin_status, oper_status)
            VALUES %s
            """,
            values,
        )
        cursor.close()

    def insert_routing_metrics(self, metrics: List[RoutingMetrics]):
        """Insert routing metrics into TimescaleDB."""
        cursor = self.conn.cursor()
        values = [
            (
                m.timestamp, m.host, m.protocol,
                m.neighbor, m.state, m.state_changes,
                m.retransmissions, m.updates_in, m.updates_out,
                m.fsm_transitions,
            )
            for m in metrics
        ]
        execute_values(
            cursor,
            """
            INSERT INTO routing_metrics
            (time, host, protocol, neighbor, state, state_changes,
             retransmissions, updates_in, updates_out, fsm_transitions)
            VALUES %s
            """,
            values,
        )
        cursor.close()

    def insert_ipsec_metrics(self, metrics: List[IPSecMetrics]):
        """Insert IPSec metrics into TimescaleDB."""
        cursor = self.conn.cursor()
        values = [
            (
                m.timestamp, m.host, m.tunnel_id,
                m.local_addr, m.remote_addr,
                m.encrypted_pkts, m.decrypted_pkts,
                m.in_errors, m.out_errors,
            )
            for m in metrics
        ]
        execute_values(
            cursor,
            """
            INSERT INTO ipsec_metrics
            (time, host, tunnel_id, local_addr, remote_addr,
             encrypted_pkts, decrypted_pkts, in_errors, out_errors)
            VALUES %s
            """,
            values,
        )
        cursor.close()

    def insert_syslog_events(self, events: List[SyslogEvent]):
        """Insert syslog events into TimescaleDB."""
        cursor = self.conn.cursor()
        values = [
            (
                e.timestamp, e.source_host,
                e.facility, e.severity, e.app_name,
                e.message, Json(e.tags),
            )
            for e in events
        ]
        execute_values(
            cursor,
            """
            INSERT INTO syslog_events
            (time, host, facility, severity, app_name, message, tags)
            VALUES %s
            """,
            values,
        )
        cursor.close()

    def insert_normalized_metrics(self, metrics: List[NormalizedMetric]):
        """Insert normalized metrics into TimescaleDB."""
        cursor = self.conn.cursor()
        values = [
            (
                m.timestamp, m.source_host, m.metric_name,
                m.value, m.unit, Json(m.tags), m.source_type,
            )
            for m in metrics
        ]
        execute_values(
            cursor,
            """
            INSERT INTO normalized_metrics
            (time, source_host, metric_name, value, unit, tags, source_type)
            VALUES %s
            """,
            values,
        )
        cursor.close()


# ════════════════════════════════════════════════════════════════
# Metric Normalizer
# ════════════════════════════════════════════════════════════════

class MetricNormalizer:
    """Normalizes raw SNMP/NetFlow/Syslog data into unified metrics."""

    SNMP_TO_NORMALIZED = {
        "if_in_octets": ("interface.bytes.in", "bytes"),
        "if_out_octets": ("interface.bytes.out", "bytes"),
        "if_in_errors": ("interface.errors.in", "count"),
        "if_out_errors": ("interface.errors.out", "count"),
        "if_in_discards": ("interface.discards.in", "count"),
        "if_out_discards": ("interface.discards.out", "count"),
        "if_in_unicast_packets": ("interface.packets.in", "count"),
        "if_out_unicast_packets": ("interface.packets.out", "count"),
        "if_speed_bps": ("interface.speed", "bps"),
        "if_oper_status": ("interface.status.oper", ""),
        "if_admin_status": ("interface.status.admin", ""),
        "ospf_neighbor_state": ("routing.ospf.neighbor_state", ""),
        "ospf_dead_timer": ("routing.ospf.dead_timer", "seconds"),
        "ospf_retransmissions": ("routing.ospf.retransmissions", "count"),
        "bgp_peer_state": ("routing.bgp.peer_state", ""),
        "bgp_fsm_transitions": ("routing.bgp.fsm_transitions", "count"),
        "bgp_in_updates": ("routing.bgp.updates.in", "count"),
        "bgp_out_updates": ("routing.bgp.updates.out", "count"),
        "mpls_ldp_state": ("routing.mpls_ldp.state", ""),
        "mpls_ldp_state_changes": ("routing.mpls_ldp.state_changes", "count"),
        "ipsec_encrypted_pkts": ("ipsec.packets.encrypted", "count"),
        "ipsec_decrypted_pkts": ("ipsec.packets.decrypted", "count"),
        "ipsec_in_errors": ("ipsec.errors.in", "count"),
        "ipsec_out_errors": ("ipsec.errors.out", "count"),
        "ping_average_response_ms": ("latency.ping.average", "ms"),
    }

    OGGPF_STATES = {
        1: "down",
        2: "attempt",
        3: "init",
        4: "2way",
        5: "exstart",
        6: "exchange",
        7: "loading",
        8: "full",
    }

    BGP_STATES = {
        1: "idle",
        2: "connect",
        3: "active",
        4: "opensent",
        5: "openconfirm",
        6: "established",
    }

    def normalize_snmp_metric(
        self, raw_name: str, value: Any, host: str, tags: Dict[str, str]
    ) -> Optional[NormalizedMetric]:
        """Normalize a single SNMP metric."""
        if raw_name not in self.SNMP_TO_NORMALIZED:
            return None

        normalized_name, unit = self.SNMP_TO_NORMALIZED[raw_name]
        float_value = float(value) if value is not None else 0.0

        # Decode protocol states
        if raw_name == "ospf_neighbor_state":
            float_value = float(self.OGGPF_STATES.get(int(value), 0))
            tags["state_name"] = self.OGGPF_STATES.get(int(value), "unknown")
        elif raw_name == "bgp_peer_state":
            float_value = float(self.BGP_STATES.get(int(value), 0))
            tags["state_name"] = self.BGP_STATES.get(int(value), "unknown")

        return NormalizedMetric(
            timestamp=datetime.now(timezone.utc),
            source_host=host,
            metric_name=normalized_name,
            value=float_value,
            unit=unit,
            tags=tags,
            source_type="snmp",
        )

    def calculate_utilization(self, metrics: Dict[str, float]) -> float:
        """Calculate interface utilization percentage."""
        if "if_in_octets" in metrics and "if_speed_bps" in metrics:
            speed = float(metrics.get("if_speed_bps", 0))
            if speed > 0:
                return (float(metrics["if_in_octets"]) * 8 / speed) * 100
        return 0.0


# ════════════════════════════════════════════════════════════════
# Syslog Parser
# ════════════════════════════════════════════════════════════════

class SyslogParser:
    """Parses syslog messages into structured events."""

    SEVERITY_MAP = {
        "emerg": 0, "alert": 1, "crit": 2, "err": 3,
        "warning": 4, "notice": 5, "info": 6, "debug": 7,
    }

    FACILITY_MAP = {
        "kern": 0, "user": 1, "mail": 2, "daemon": 3,
        "auth": 4, "syslog": 5, "lpr": 6, "news": 7,
    }

    BGP_KEYWORDS = [
        "BGP", "bgp", "UPDATE", "KEEPALIVE", "OPEN", "NOTIFICATION",
        "CEASE", "ROUTE-REFRESH", "adjacency", "neighbor",
    ]

    OSPF_KEYWORDS = [
        "OSPF", "ospf", "AREA", "router", "LSA", "adjacency",
        "interface", "neighbor", "HELLO", "DD", "LSR", "LSU", "LSACK",
    ]

    MPLS_KEYWORDS = [
        "MPLS", "mpls", "LDP", "ldp", "label", "session",
        "neighbor", "LSP", "tunnel",
    ]

    def parse(self, raw_message: str, source_host: str) -> SyslogEvent:
        """Parse a raw syslog message into structured event."""
        timestamp = datetime.now(timezone.utc)
        
        # Extract severity and facility from message if available
        severity = 6  # Default: info
        facility = 16  # Default: local0
        
        # Try to identify event category
        app_name = "unknown"
        tags = {"raw": raw_message[:500]}
        
        if any(kw in raw_message for kw in self.BGP_KEYWORDS):
            app_name = "bgp"
            tags["category"] = "routing"
            tags["protocol"] = "bgp"
        elif any(kw in raw_message for kw in self.OSPF_KEYWORDS):
            app_name = "ospf"
            tags["category"] = "routing"
            tags["protocol"] = "ospf"
        elif any(kw in raw_message for kw in self/MPLS_KEYWORDS):
            app_name = "mpls"
            tags["category"] = "mpls"
            tags["protocol"] = "mpls_ldp"
        elif "IPSEC" in raw_message or "ipsec" in raw_message:
            app_name = "ipsec"
            tags["category"] = "security"
        
        return SyslogEvent(
            timestamp=timestamp,
            source_host=source_host,
            facility=facility,
            severity=severity,
            app_name=app_name,
            message=raw_message,
            tags=tags,
        )


# ════════════════════════════════════════════════════════════════
# Main Telemetry Collector
# ════════════════════════════════════════════════════════════════

class TelemetryCollector:
    """Main telemetry collection and normalization engine."""

    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.normalizer = MetricNormalizer()
        self.syslog_parser = SyslogParser()
        self.db = TimescaleDBManager(
            host=config.get("db_host", "timescaledb"),
            port=int(config.get("db_port", 5432)),
            dbname=config.get("db_name", "noc_telemetry"),
            user=config.get("db_user", "postgres"),
            password=config.get("db_password", "noc_copilot_db"),
        )
        self._running = False
        self._buffer: List[NormalizedMetric] = []
        self._buffer_lock = threading.Lock()
        
        # Prometheus metrics for monitoring
        self.registry = CollectorRegistry()
        self.metrics_processed = Counter(
            "telemetry_metrics_processed_total",
            "Total metrics processed",
            ["source_type"],
            registry=self.registry,
        )
        self.metrics_errors = Counter(
            "telemetry_errors_total",
            "Total processing errors",
            ["source_type"],
            registry=self.registry,
        )
        self.current_buffer_size = Gauge(
            "telemetry_buffer_size",
            "Current buffer size",
            registry=self.registry,
        )

    def process_snmp_data(self, raw_data: Dict[str, Any], host: str):
        """Process raw SNMP data and normalize."""
        try:
            normalized_metrics = []
            for key, value in raw_data.items():
                metric = self.normalizer.normalize_snmp_metric(
                    key, value, host, {"device": host}
                )
                if metric:
                    normalized_metrics.append(metric)
                    self.metrics_processed.labels(source_type="snmp").inc()
            
            # Batch insert
            if normalized_metrics:
                with self._buffer_lock:
                    self._buffer.extend(normalized_metrics)
                
        except Exception as e:
            logger.error(f"SNMP processing error for {host}: {e}")
            self.metrics_errors.labels(source_type="snmp").inc()

    def process_netflow_data(self, flow_data: Dict[str, Any]):
        """Process raw NetFlow/IPFIX data."""
        try:
            # Convert flow data to normalized metrics
            metrics = []
            
            # Extract flow key metrics
            source_ip = flow_data.get("src_addr", "")
            dest_ip = flow_data.get("dst_addr", "")
            bytes_count = flow_data.get("bytes", 0)
            packets_count = flow_data.get("packets", 0)
            protocol = flow_data.get("protocol", 0)
            
            timestamp = datetime.now(timezone.utc)
            
            # Create flow volume metrics
            metrics.append(NormalizedMetric(
                timestamp=timestamp,
                source_host=f"flow-{source_ip}",
                metric_name="netflow.bytes",
                value=float(bytes_count),
                unit="bytes",
                tags={
                    "src": source_ip,
                    "dst": dest_ip,
                    "protocol": str(protocol),
                },
                source_type="netflow",
            ))
            
            metrics.append(NormalizedMetric(
                timestamp=timestamp,
                source_host=f"flow-{source_ip}",
                metric_name="netflow.packets",
                value=float(packets_count),
                unit="count",
                tags={
                    "src": source_ip,
                    "dst": dest_ip,
                    "protocol": str(protocol),
                },
                source_type="netflow",
            ))
            
            self.metrics_processed.labels(source_type="netflow").inc()
            
        except Exception as e:
            logger.error(f"NetFlow processing error: {e}")
            self.metrics_errors.labels(source_type="netflow").inc()

    def process_syslog(self, raw_message: str, source_host: str):
        """Process raw syslog message."""
        try:
            event = self.syslog_parser.parse(raw_message, source_host)
            
            # Store in TimescaleDB
            self.db.insert_syslog_events([event])
            
            # Create normalized metric for critical syslog events
            if event.severity <= 3:  # err or worse
                self.db.insert_normalized_metrics([
                    NormalizedMetric(
                        timestamp=event.timestamp,
                        source_host=source_host,
                        metric_name=f"syslog.severity.{event.severity}",
                        value=1.0,
                        unit="count",
                        tags=event.tags,
                        source_type="syslog",
                    )
                ])
            
            self.metrics_processed.labels(source_type="syslog").inc()
            
        except Exception as e:
            logger.error(f"Syslog processing error for {source_host}: {e}")
            self.metrics_errors.labels(source_type="syslog").inc()

    def flush_buffer(self):
        """Flush buffered metrics to database."""
        with self._buffer_lock:
            if not self._buffer:
                return
            
            batch = self._buffer[:1000]
            self._buffer = self._buffer[1000:]
        
        try:
            # Group by metric type for efficient insertion
            interface_metrics = []
            routing_metrics = []
            ipsec_metrics = []
            
            for metric in batch:
                if metric.metric_name.startswith("interface."):
                    # Convert to InterfaceMetrics
                    pass
                elif metric.metric_name.startswith("routing."):
                    pass
                elif metric.metric_name.startswith("ipsec."):
                    pass
            
            # Insert all normalized metrics
            self.db.insert_normalized_metrics(batch)
            
            logger.debug(f"Flushed {len(batch)} metrics to database")
            
        except Exception as e:
            logger.error(f"Buffer flush error: {e}")
            self.metrics_errors.labels(source_type="flush").inc()

    def start_collection(self):
        """Start the telemetry collection loop."""
        self._running = True
        logger.info("Starting telemetry collection...")
        
        while self._running:
            try:
                self.flush_buffer()
                self.current_buffer_size.set(len(self._buffer))
                time.sleep(10)
            except KeyboardInterrupt:
                self._running = False
            except Exception as e:
                logger.error(f"Collection loop error: {e}")
                time.sleep(5)

    def stop_collection(self):
        """Stop the telemetry collection loop."""
        self._running = False
        self.flush_buffer()
        logger.info("Telemetry collection stopped")


# ════════════════════════════════════════════════════════════════
# Prometheus Exporter
# ════════════════════════════════════════════════════════════════

class PrometheusExporter:
    """Exports normalized metrics to Prometheus format."""
    
    def __init__(self, db: TimescaleDBManager):
        self.db = db
    
    def query_latest_metrics(self, metric_name: str, limit: int = 1000) -> List[Dict]:
        """Query latest metrics from TimescaleDB."""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT time, source_host, metric_name, value, unit, tags
            FROM normalized_metrics
            WHERE metric_name = %s
            ORDER BY time DESC
            LIMIT %s
        """, (metric_name, limit))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        return results
    
    def to_prometheus_format(self, metrics: List[Dict]) -> str:
        """Convert metrics to Prometheus exposition format."""
        lines = []
        for metric in metrics:
            name = metric["metric_name"].replace(".", "_")
            value = metric["value"]
            host = metric["source_host"]
            unit = metric.get("unit", "")
            
            # Prometheus metric format
            if unit:
                lines.append('{__name__="%s",unit="%s",host="%s"} %s' % (name, unit, host, value))
            else:
                lines.append('{__name__="%s",host="%s"} %s' % (name, host, value))
        
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# Entry Point
# ════════════════════════════════════════════════════════════════

def main():
    """Main entry point for the telemetry normalizer."""
    config = {
        "db_host": "timescaledb",
        "db_port": "5432",
        "db_name": "noc_telemetry",
        "db_user": "postgres",
        "db_password": "noc_copilot_db",
    }
    
    logger.info("Initializing telemetry normalizer...")
    collector = TelemetryCollector(config)
    
    try:
        collector.start_collection()
    except KeyboardInterrupt:
        collector.stop_collection()
    
    logger.info("Telemetry normalizer shut down")


if __name__ == "__main__":
    main()
