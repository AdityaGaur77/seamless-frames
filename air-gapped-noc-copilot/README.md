# Air-Gapped NOC Copilot

Predictive network fault detection for air-gapped MPLS/SD-WAN environments. Uses LSTM/TCN models for time-series forecasting, RAG for evidence retrieval, and a local LLM (Ollama) for generating NOC-ready incident reports — all without internet access.

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Docker + Docker Compose | v24+ / v2.20+ | Container stack |
| Python | 3.10+ | Model training |
| Ollama | latest | Local LLM inference |

## Project Structure

```
air-gapped-noc-copilot/
├── controller/                 # Orchestrator (Phase 3-5)
│   ├── orchestrator.py         # Main loop: sample → score → RAG → LLM → validate
│   ├── metric_sampler.py       # TimescaleDB + Prometheus live backends
│   ├── infer_client.py         # Ollama + OpenAI-compatible LLM clients
│   ├── model_scorer.py         # PyTorch inference wrapper
│   ├── alert_builder.py        # Alert payload construction
│   ├── rag_bridge.py           # RAG evidence retrieval
│   ├── prompt_orchestrator.py  # LLM prompt assembly
│   ├── response_gate.py        # LLM response validation
│   └── config.py               # YAML + env-var configuration
├── m3/                         # Knowledge modules
│   ├── rag/                    # RAG index, chunker, query engine
│   ├── prompts/                # System prompt, few-shot examples, schema
│   ├── playbooks/              # Fault injection scenarios
│   └── telemetry/              # SNMP/NetFlow collectors
├── configs/                    # Generated FRR device configs
├── docker-compose.yml          # Full production stack
├── Dockerfile                  # Orchestrator container
├── init-timescaledb.sql        # Database schema + hypertables
├── telegraf.conf               # Telegraf SNMP/ping/NetFlow config
├── prometheus.yml              # Prometheus scrape + recording rules
├── data_preprocessor.py        # Feature engineering pipeline
├── lstm_model.py               # LSTM with attention
├── tcn_model.py                # TCN + hybrid TCN-LSTM
├── train_models.py             # Training pipeline
└── requirements.txt            # Python dependencies
```

## Quick Start

### Option A: Full Docker Stack (Production)

```bash
# 1. Pull Ollama and download a model (on the Docker host)
ollama pull llama3:latest

# 2. Start the full stack
docker compose up --build -d

# 3. Verify all services are healthy
docker compose ps

# 4. Check orchestrator logs
docker compose logs -f orchestrator
```

This spins up:
- **TimescaleDB** on `127.0.0.1:5432`
- **Prometheus** on `127.0.0.1:9090`
- **Telegraf** (SNMP collector)
- **Orchestrator** (the NOC Copilot brain)
- **Grafana** on `127.0.0.1:3000`
- **Node Exporter** on `127.0.0.1:9100`

All ports are bound to `127.0.0.1` — nothing is exposed to the network.

### Option B: Local Development (No Docker)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Train the models
python train_models.py --models lstm tcn hybrid

# 3. Start TimescaleDB (Docker only for the DB)
docker compose up timescaledb -d

# 4. Run the orchestrator (uses fixture data if no DB)
python -m controller.orchestrator --once
```

## Training Models

```bash
# Train all three architectures
python train_models.py --models lstm tcn hybrid

# Train only LSTM
python train_models.py --models lstm

# Custom data and output
python train_models.py --data your_telemetry.csv --output ./checkpoints
```

Output:
- `best_model.pth` — best checkpoint (by validation loss)
- `best_model.pth.scalers.pkl` — fitted feature scalers
- `models/training_summary.json` — metrics comparison

## Configuration

### YAML Config

Edit `controller/controller_config.yaml`:

```yaml
sampler:
  backend: timescaledb    # timescaledb | prometheus | fixture
  db_host: 127.0.0.1
  db_port: 5432

llm:
  backend: ollama         # ollama | openai
  ollama_url: http://127.0.0.1:11434
  ollama_model: llama3:latest
```

### Environment Variable Overrides

When running in Docker, env vars override the YAML file:

| Env Var | Config Path | Example |
|---------|-------------|---------|
| `ORCHESTRATOR_DB_HOST` | `sampler.db_host` | `172.20.0.120` |
| `ORCHESTRATOR_DB_PORT` | `sampler.db_port` | `5432` |
| `ORCHESTRATOR_PROMETHEUS_URL` | `sampler.prometheus_url` | `http://172.20.0.110:9090` |
| `ORCHESTRATOR_OLLAMA_URL` | `llm.ollama_url` | `http://host.docker.internal:11434` |
| `ORCHESTRATOR_OLLAMA_MODEL` | `llm.ollama_model` | `llama3:latest` |
| `ORCHESTRATOR_LLM_BACKEND` | `llm.backend` | `ollama` |

## LLM Backends

### Ollama (Recommended)

```yaml
llm:
  backend: ollama
  ollama_url: http://127.0.0.1:11434
  ollama_model: llama3:latest
```

Supports `/api/chat` (multi-turn) and `/api/generate` (single-turn). Auto-routes based on message count.

### OpenAI-Compatible (llama.cpp / vLLM)

```yaml
llm:
  backend: openai
  base_url: http://127.0.0.1:8080
  api_path: /v1/chat/completions
  model_name: airgap-noc
```

## Orchestrator Pipeline

Each cycle:

```
1. sampler.fetch_all()         → list[MetricFrame]
2. for each frame:
   a. scorer.score()           → forecast + anomaly prob + TTI
   b. threshold gate           → skip RAG if anomaly < 0.5
   c. build_alert_payload()    → structured alert
   d. rag_bridge.retrieve()    → evidence chunks
   e. prompt_orchestrator()    → chat messages
   f. llm.chat()               → NOC-ready response
   g. response_gate.validate() → validated output
3. publish JSON to sink (stdout or file)
```

Run modes:
```bash
# Run forever (default)
python -m controller.orchestrator

# Single cycle
python -m controller.orchestrator --once

# 5 cycles then exit
python -m controller.orchestrator --max-cycles 5
```

## Database Schema

`init-timescaledb.sql` creates:

| Table | Purpose |
|-------|---------|
| `interface_metrics` | SNMP interface counters |
| `routing_metrics` | OSPF/BGP state |
| `ipsec_metrics` | IPSec tunnel health |
| `syslog_events` | Device syslog |
| `netflow_records` | Flow records |
| `predictive_alerts` | Model-generated alerts |
| `incidents` | Incident history |

Continuous aggregates: `mv_5min_interface_stats`, `mv_5min_routing_stability`.

## Stopping

```bash
# Stop all services
docker compose down

# Stop and remove volumes (fresh start)
docker compose down -v
```
