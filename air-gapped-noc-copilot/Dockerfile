# ╔═══════════════════════════════════════════════════════════════╗
# ║ Air-Gapped NOC Copilot — Orchestrator (Phase 3)              ║
# ║ Multi-stage build: Python 3.11-slim + PyTorch CPU            ║
# ╚═══════════════════════════════════════════════════════════════╝
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for psycopg2-binary and general runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq-dev \
        gcc \
        curl \
        tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies first (layer cache) ──────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────
# Project root contents
COPY lstm_model.py tcn_model.py data_preprocessor.py ./
COPY best_model.pth ./

# Controller package
COPY controller/ controller/

# M3 modules (RAG, prompts, playbooks, telemetry schemas)
COPY m3/ m3/

# Config files
COPY telegraf.conf prometheus.yml ./
COPY init-timescaledb.sql ./

# ── Runtime ──────────────────────────────────────────────────
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "controller.orchestrator", \
     "--config", "controller/controller_config.yaml"]
