# Air-Gapped NOC Copilot

Predictive network fault detection system for air-gapped MPLS/SD-WAN environments.

## Project Structure

```
air-gapped-noc-copilot/
├── topology.clab.yml          # Containerlab topology definition
├── daemons                    # FRR daemons configuration
├── node-frr.conf             # Base FRR configuration template
├── configure_routing.py       # BGP/OSPF/MPLS configuration generator
├── configure_ipsec_overlay.py # IPSec overlay configuration generator
├── deploy_topology.sh         # Topology deployment script
├── telegraf.conf              # Telemetry collection configuration
├── prometheus.yml             # Prometheus scrape configuration
├── docker-compose.yml         # Telemetry stack deployment
├── init-timescaledb.sql       # Database schema initialization
├── telemetry_normalizer.py    # Telemetry stream normalization
├── data_preprocessor.py       # ML data preprocessing pipeline
├── lstm_model.py              # LSTM prediction model
├── tcn_model.py               # TCN prediction model
├── train_models.py            # Model training script
├── requirements.txt           # Python dependencies
└── configs/                   # Generated device configurations
```

## Quick Start

### 1. Deploy Network Topology

```bash
# Deploy Containerlab topology
chmod +x deploy_topology.sh
./deploy_topology.sh
```

### 2. Start Telemetry Stack

```bash
cd telemetry-stack
docker-compose up -d
```

### 3. Train Models

```bash
pip install -r requirements.txt
python train_models.py --data synthetic_telemetry.csv --models lstm tcn hybrid
```

## Architecture

### Phase 1: Network Simulation
- Multi-site MPLS/SD-WAN topology with Containerlab
- FRR-based routing (BGP, OSPF, MPLS LDP)
- IPSec overlay tunnels for SD-WAN emulation

### Phase 2: Telemetry Pipeline
- SNMP polling for interface/routing metrics
- NetFlow/IPFIX collection for traffic analysis
- Syslog aggregation for event correlation
- TimescaleDB for time-series storage

### Phase 3: Predictive Models
- LSTM with attention for sequence forecasting
- TCN for local pattern detection
- Hybrid TCN-LSTM for combined analysis
- Multi-task learning (utilization, anomaly, TTI)

## Generated Files

- `configs/*.conf` - FRR configurations for all nodes
- `configs/*_ipsec.conf` - IPSec overlay configurations
- `models/*.pth` - Trained model weights
- `models/metadata.json` - Feature and training metadata
