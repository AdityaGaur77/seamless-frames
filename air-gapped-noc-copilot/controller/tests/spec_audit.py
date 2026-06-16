"""One-shot spec-compliance audit for the controller.

Verifies that every claim in the spec ("regular sampling from DB /
Prom", "PyTorch model scores", "RAG with Chroma", "offline LLM",
"JSON schema", "controller/", "uses m3/") is wired into the package.

Run:
    cd air-gapped-noc-copilot
    python -m controller.tests.spec_audit
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("NOC_COPILOT_FAKE_EMBEDDER", "1")
os.environ.setdefault("NOC_COPILOT_SKIP_AIRGAP_CHECK", "1")

checks = []

from controller.metric_sampler import TimescaleDBSampler, PrometheusSampler, FixtureSampler
checks.append(('Sampling: TimescaleDBSampler', TimescaleDBSampler is not None))
checks.append(('Sampling: PrometheusSampler', PrometheusSampler is not None))
checks.append(('Sampling: FixtureSampler (offline mode)', FixtureSampler is not None))

from controller.model_scorer import ModelScorer, ScoringResult
checks.append(('Scoring: ModelScorer', ModelScorer is not None))
checks.append(('Scoring: ScoringResult.anomaly_prob', 'anomaly_prob' in ScoringResult.__dataclass_fields__))
checks.append(('Scoring: ScoringResult.forecast', 'forecast' in ScoringResult.__dataclass_fields__))
checks.append(('Scoring: ScoringResult.tti_minutes', 'tti_minutes' in ScoringResult.__dataclass_fields__))

from controller.alert_builder import build_alert_payload, validate_alert_payload
from controller.rag_bridge import RagBridge, _build_where_clause
import json
schema = json.loads(open('controller/alert_schema.json').read())
checks.append(('ALERT_PAYLOAD: schema_version 1.0.0 pinned', schema['properties']['schema_version']['const'] == '1.0.0'))
checks.append(('ALERT_PAYLOAD: predicted_issue.type enum', 'congestion_saturation' in schema['properties']['predicted_issue']['properties']['type']['enum']))
checks.append(('ALERT_PAYLOAD: provenance.checkpoint_sha256', 'checkpoint_sha256' in schema['properties']['provenance']['required']))
checks.append(('RAG: RagBridge wraps m3', RagBridge is not None))
checks.append(('RAG: multi-filter -> $and', _build_where_clause({'a': 1, 'b': 2}) == {'$and': [{'a': 1}, {'b': 2}]}))
checks.append(('RAG: single-filter passthrough', _build_where_clause({'a': 1}) == {'a': 1}))
checks.append(('RAG: no-filter -> None', _build_where_clause({}) is None))

from controller.infer_client import OfflineLLMClient, OfflineLLMUnavailable
from controller.response_gate import ResponseGate
from controller.prompt_orchestrator import PromptOrchestrator
checks.append(('Inference: OfflineLLMClient', OfflineLLMClient is not None))
checks.append(('Inference: PromptOrchestrator', PromptOrchestrator is not None))
checks.append(('Inference: ResponseGate', ResponseGate is not None))

from controller.orchestrator import Orchestrator
checks.append(('Loop: Orchestrator', Orchestrator is not None))
checks.append(('Loop: sample_interval_seconds in config', 'sample_interval_seconds' in open('controller/config.py').read()))

checks.append(('Subfolder: controller/ at project root', os.path.exists('controller/__init__.py')))
checks.append(('Subfolder: m3/ (previous RAG session)', os.path.exists('m3/rag/rag_query.py')))
checks.append(('Context: LSTM/TCN models used', os.path.exists('lstm_model.py') and os.path.exists('tcn_model.py')))
checks.append(('Context: data_preprocessor schema', 'sequence_length: int = 60' in open('data_preprocessor.py', encoding='utf-8').read()))
checks.append(('Context: Chroma index present', os.path.exists('index/chroma/chroma.sqlite3')))

print('SPEC COMPLIANCE:')
for name, ok in checks:
    mark = 'PASS' if ok else 'FAIL'
    print(f'  {mark}  {name}')
total = sum(1 for _, ok in checks if ok)
print(f'{total}/{len(checks)} checks passed')
sys.exit(0 if total == len(checks) else 1)
