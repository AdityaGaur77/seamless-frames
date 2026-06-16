-- ╔═══════════════════════════════════════════════════════════════╗
-- ║ Air-Gapped NOC Copilot - TimescaleDB Initialization          ║
-- ║ Schema and Hypertables for Network Telemetry                 ║
-- ╚═══════════════════════════════════════════════════════════════╝

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ════════════════════════════════════════════════════════════════
-- Core Telemetry Tables
-- ════════════════════════════════════════════════════════════════

-- Interface Metrics (SNMP-derived)
CREATE TABLE IF NOT EXISTS interface_metrics (
    time        TIMESTAMPTZ NOT NULL,
    host        TEXT NOT NULL,
    interface   TEXT NOT NULL,
    in_octets   DOUBLE PRECISION,
    out_octets  DOUBLE PRECISION,
    in_errors   DOUBLE PRECISION,
    out_errors  DOUBLE PRECISION,
    in_discards  DOUBLE PRECISION,
    out_discards DOUBLE PRECISION,
    in_packets  DOUBLE PRECISION,
    out_packets DOUBLE PRECISION,
    speed       DOUBLE PRECISION,
    utilization DOUBLE PRECISION,
    admin_status INTEGER,
    oper_status  INTEGER
);

-- Routing Protocol Metrics
CREATE TABLE IF NOT EXISTS routing_metrics (
    time            TIMESTAMPTZ NOT NULL,
    host            TEXT NOT NULL,
    protocol        TEXT NOT NULL,
    neighbor        TEXT,
    state           TEXT,
    state_changes   INTEGER,
    retransmissions INTEGER,
    updates_in      INTEGER,
    updates_out     INTEGER,
    fsm_transitions INTEGER
);

-- IPSec Tunnel Metrics
CREATE TABLE IF NOT EXISTS ipsec_metrics (
    time            TIMESTAMPTZ NOT NULL,
    host            TEXT NOT NULL,
    tunnel_id       TEXT NOT NULL,
    local_addr      TEXT,
    remote_addr     TEXT,
    encrypted_pkts  DOUBLE PRECISION,
    decrypted_pkts  DOUBLE PRECISION,
    in_errors       DOUBLE PRECISION,
    out_errors      DOUBLE PRECISION
);

-- Syslog Events
CREATE TABLE IF NOT EXISTS syslog_events (
    time        TIMESTAMPTZ NOT NULL,
    host        TEXT NOT NULL,
    facility    INTEGER,
    severity    INTEGER,
    app_name    TEXT,
    message     TEXT,
    tags        JSONB
);

-- NetFlow/IPFIX Records
CREATE TABLE IF NOT EXISTS netflow_records (
    time            TIMESTAMPTZ NOT NULL,
    source_ip       TEXT,
    dest_ip         TEXT,
    source_port     INTEGER,
    dest_port       INTEGER,
    protocol        INTEGER,
    bytes           BIGINT,
    packets         BIGINT,
    flow_start      TIMESTAMPTZ,
    flow_end        TIMESTAMPTZ,
    tags            JSONB
);

-- Normalized Metrics (Unified Format)
CREATE TABLE IF NOT EXISTS normalized_metrics (
    time         TIMESTAMPTZ NOT NULL,
    source_host  TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    value        DOUBLE PRECISION,
    unit         TEXT,
    tags         JSONB,
    source_type  TEXT
);

-- ════════════════════════════════════════════════════════════════
-- Prediction and Alert Tables
-- ════════════════════════════════════════════════════════════════

-- Predictive Alerts
CREATE TABLE IF NOT EXISTS predictive_alerts (
    time            TIMESTAMPTZ NOT NULL,
    alert_id        UUID DEFAULT gen_random_uuid(),
    alert_type      TEXT NOT NULL,
    severity        INTEGER,
    confidence      DOUBLE PRECISION,
    source_host     TEXT,
    affected_scope  JSONB,
    predicted_issue TEXT,
    root_cause      TEXT,
    recommended_actions TEXT,
    time_to_impact  DOUBLE PRECISION,
    status          TEXT DEFAULT 'active',
    acknowledged    BOOLEAN DEFAULT FALSE
);

-- Model Training Data
CREATE TABLE IF NOT EXISTS training_data (
    time            TIMESTAMPTZ NOT NULL,
    feature_set     JSONB NOT NULL,
    label           TEXT,
    scenario        TEXT,
    fault_type      TEXT,
    fault_injected  BOOLEAN DEFAULT FALSE
);

-- Incident History
CREATE TABLE IF NOT EXISTS incidents (
    time            TIMESTAMPTZ NOT NULL,
    incident_id     UUID DEFAULT gen_random_uuid(),
    title           TEXT NOT NULL,
    description     TEXT,
    severity        INTEGER,
    status          TEXT DEFAULT 'open',
    affected_hosts  JSONB,
    root_cause      TEXT,
    resolution      TEXT,
    copilot_suggestion TEXT,
    actual_outcome  TEXT,
    prediction_lead_time DOUBLE PRECISION
);

-- ════════════════════════════════════════════════════════════════
-- Create Hypertables
-- ════════════════════════════════════════════════════════════════

SELECT create_hypertable('interface_metrics', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
SELECT create_hypertable('routing_metrics', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
SELECT create_hypertable('ipsec_metrics', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
SELECT create_hypertable('syslog_events', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
SELECT create_hypertable('netflow_records', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
SELECT create_hypertable('normalized_metrics', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
SELECT create_hypertable('predictive_alerts', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');
SELECT create_hypertable('training_data', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');
SELECT create_hypertable('incidents', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');

-- ════════════════════════════════════════════════════════════════
-- Create Indexes
-- ════════════════════════════════════════════════════════════════

CREATE INDEX IF NOT EXISTS idx_interface_host_time ON interface_metrics (host, time DESC);
CREATE INDEX IF NOT EXISTS idx_interface_host_interface ON interface_metrics (host, interface, time DESC);
CREATE INDEX IF NOT EXISTS idx_routing_host_protocol ON routing_metrics (host, protocol, time DESC);
CREATE INDEX IF NOT EXISTS idx_ipsec_host_tunnel ON ipsec_metrics (host, tunnel_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_syslog_host_time ON syslog_events (host, time DESC);
CREATE INDEX IF NOT EXISTS idx_netflow_src_dst ON netflow_records (source_ip, dest_ip, time DESC);
CREATE INDEX IF NOT EXISTS idx_normalized_metric_time ON normalized_metrics (metric_name, time DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON predictive_alerts (severity, time DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON predictive_alerts (alert_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents (status, time DESC);

-- ════════════════════════════════════════════════════════════════
-- Continuous Aggregates for ML Training
-- ════════════════════════════════════════════════════════════════

-- 5-minute interface statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_5min_interface_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', time) AS bucket,
    host,
    interface,
    AVG(utilization) AS avg_utilization,
    MAX(utilization) AS max_utilization,
    MIN(utilization) AS min_utilization,
    SUM(in_errors) AS total_in_errors,
    SUM(out_errors) AS total_out_errors,
    SUM(in_discards) AS total_in_discards,
    SUM(out_discards) AS total_out_discards,
    AVG(in_octets) AS avg_in_octets,
    AVG(out_octets) AS avg_out_octets
FROM interface_metrics
GROUP BY bucket, host, interface
WITH NO DATA;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('mv_5min_interface_stats',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes'
);

-- 5-minute routing stability metrics
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_5min_routing_stability
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', time) AS bucket,
    host,
    protocol,
    COUNT(*) AS total_updates,
    SUM(state_changes) AS total_state_changes,
    AVG(retransmissions) AS avg_retransmissions,
    MAX(fsm_transitions) AS max_fsm_transitions
FROM routing_metrics
GROUP BY bucket, host, protocol
WITH NO DATA;

-- ════════════════════════════════════════════════════════════════
-- Data Retention Policies
-- ════════════════════════════════════════════════════════════════

-- Raw data: 30 days
SELECT add_retention_policy('interface_metrics', INTERVAL '30 days');
SELECT add_retention_policy('routing_metrics', INTERVAL '30 days');
SELECT add_retention_policy('ipsec_metrics', INTERVAL '30 days');
SELECT add_retention_policy('syslog_events', INTERVAL '30 days');
SELECT add_retention_policy('netflow_records', INTERVAL '30 days');
SELECT add_retention_policy('normalized_metrics', INTERVAL '30 days');

-- Aggregated data: 90 days
SELECT add_retention_policy('mv_5min_interface_stats', INTERVAL '90 days');
SELECT add_retention_policy('mv_5min_routing_stability', INTERVAL '90 days');

-- Alerts: 180 days
SELECT add_retention_policy('predictive_alerts', INTERVAL '180 days');
SELECT add_retention_policy('incidents', INTERVAL '180 days');

-- Training data: unlimited (no retention policy)
