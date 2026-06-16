# `m3/telemetry/` — Phase 1/2 support: SNMP/NetFlow \u2192 Prometheus pipeline

Supports the mimo v2.5 Containerlab topology by providing the actual
configuration and Python collectors that bind into the existing
`telegraf-collector` and `prometheus` nodes. The topology is owned by the
other agent; this folder is the **bridge** from packet \u2192 time-series.

## Layout

```
m3/telemetry/
  README.md
  config/
    telegraf.conf           # bound to telegraf-collector at /etc/telegraf/telegraf.conf
    prometheus.yml          # bound to prometheus at /etc/prometheus/prometheus.yml
    snmp_targets.yaml       # device list polled by the SNMP input plugin
  collectors/
    snmp_collector.py       # pure-Python alternative to telegraf's snmp input
    netflow_collector.py    # pure-Python NetFlow v5/v9 + IPFIX parser
    mapper.py               # OID \u2192 metric name + label mapping table
```

## How the pieces fit

```
Containerlab nodes            collectors in this folder
==================            =========================
PE/CE/P routers  \u2014\u2014SNMP\u2014\u2192   telegraf (inputs.snmp)   \u2502
                                snmp_collector.py          \u2502
                                                         \u2502
PE/CE/P routers  \u2014\u2014NetFlow\u2192  telegraf (inputs.netflow) \u2502
                                netflow_collector.py       \u2502
                                                         \u2502
Telegraf / collectors \u2014prom\u2014\u2192  prometheus (scrape)   \u2502
                                                         \u2502
                                              TimescaleDB   \u2502 (long-term storage)
                                                         \u2502
                                              Predictive    \u2502 (Phase 3)
                                              engine       \u2502
                                                         \u2502
                                              Copilot (m3/  \u2502 (Phase 4-6)
                                              rag, m3/
                                              prompts)
```

The `snmp_targets.yaml` file is the **single source of truth** for the
device list; the topology's `mgmt-ipv4` addresses are mirrored here so
the SNMP input and the Python collector stay in lockstep with what
containerlab brought up.

## How to bind into the existing Containerlab topology

Add these binds to the `telegraf-collector` and `prometheus` nodes in
`topology.clab.yml` (left untouched by mimo v2.5, but the bind lines
below are *additive* and safe to merge):

```yaml
    telegraf-collector:
      binds:
        - m3/telemetry/config/telegraf.conf:/etc/telegraf/telegraf.conf
        - m3/telemetry/config/snmp_targets.yaml:/etc/telegraf/snmp_targets.yaml
    prometheus:
      binds:
        - m3/telemetry/config/prometheus.yml:/etc/prometheus/prometheus.yml
```

`snmp_collector.py` and `netflow_collector.py` are run as alternative
collectors if the operator prefers pure Python over Telegraf. They
expose a Prometheus scrape endpoint on `:9126` and `:9127` respectively
and write the same metric names as the Telegraf path.
