# `m3/playbooks/` — Scenario Validation Playbooks (Phase 6)

Four end-to-end fault-injection scenarios that exercise the full Copilot
stack from packet → alert → grounded NOC recommendation. Each playbook is
a self-contained document that tells the operator:

1. What the scenario simulates and why.
2. The exact fault-injection primitives used (traffic control, BGP
   daemon commands, controller config edits).
3. The expected predictive engine output (alert_id, risk_band, signals,
   time-to-impact).
4. The expected Copilot response shape (predicted_issue, cited
   runbook, recommended actions).
5. The validation metrics to capture (lead time, explanation
   grounding %, action applicability).

A single `fault_injection_runner.py` automates the four scenarios against
a Containerlab instance; `validation_metrics.py` rolls up the per-scenario
metrics into the dashboard report required by the project spec.

## The four scenarios

| # | Name | File | Failure class |
|---|------|------|---------------|
| 1 | Progressive congestion on hub-spoke link | `scenario_01_congestion.md` | Link saturation precursor |
| 2 | BGP route flap with downstream path reroute cascade | `scenario_02_bgp_flap.md` | Routing instability precursor |
| 3 | Intermittent MPLS underlay failure with tunnel degradation | `scenario_03_mpls_underlay.md` | Underlay loss precursor |
| 4 | Controller misconfiguration leading to policy drift | `scenario_04_policy_drift.md` | Configuration drift precursor |

## File layout

```
m3/playbooks/
  README.md
  scenario_01_congestion.md
  scenario_02_bgp_flap.md
  scenario_03_mpls_underlay.md
  scenario_04_policy_drift.md
  fault_injection_runner.py     # drives the four scenarios in sequence
  validation_metrics.py         # computes lead time, grounding %, action applicability
```

## How the playbooks map to the spec's success criteria

The project spec says success is measured by:

| Success criterion (spec) | How this folder measures it |
|---|---|
| Prediction accuracy and lead time | `validation_metrics.py::lead_time_minutes` per scenario; alarm-to-impact gap. |
| Explanation quality, grounded, no hallucination | `validation_metrics.py::grounding_recall_at_k`; `validation_metrics.py::fabrication_rate` (catches chunk_ids that don't exist in the index). |
| Operator-relevant | `validation_metrics.py::action_applicability` (does the cited runbook chunk actually contain the recommended action?). |
| Security & air-gap compliance | Every script refuses to run if `runtime.refuse_network_calls=true` and DNS to a public host resolves. |

## Running a single scenario

```bash
python -m m3.playbooks.fault_injection_runner \
    --scenario 1 \
    --clab-topology ../topology.clab.yml \
    --config  m3/rag/rag_config.yaml \
    --prometheus http://172.20.0.110:9090 \
    --copilot-url http://127.0.0.1:8080/v1/chat \
    --out  m3/playbooks/runs/scenario_01.json
```

The runner prints a per-step trace to stderr and writes the full
alert + Copilot response + metrics JSON to `--out`. The validator
(`m3/prompts/schema_validator.py`) is called inline on every Copilot
response; if validation fails, the run is recorded with
`copilot_unavailable: true`.
