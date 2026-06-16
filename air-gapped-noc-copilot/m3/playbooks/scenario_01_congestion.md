# Scenario 1 — Progressive congestion on a hub-spoke link

## Goal

Validate that the Copilot predicts *progressive* link saturation on the
`pe-hub-east-1 \u2194 p-core-1` 40G uplink **before** it crosses the SLA
threshold, and that the recommended action cites a real, applicable
runbook step.

## Fault model

A 30-minute workload ramp on `ce-branch-3` (branch-sfo) drives the
hub-spoke uplink from a baseline ~30 % utilisation to > 95 %. The ramp is
shaped by a piecewise linear traffic generator that respects the
QoS policy in `corpus/topology/topology_reference.yaml` (`qos-cust-blue`):

| Minute | Offered load (Mbps) |
|---|---|
| 0 \u2013 5 | 12 000 (baseline) |
| 6 \u2013 12 | 18 000 (linear ramp) |
| 13 \u2013 22 | 26 000 (linear ramp) |
| 23 \u2013 30 | 34 000 (linear ramp; expected to saturate) |

Voice and video share the same physical link; the QoS policy gives them
priority but the link's physical capacity is the hard limit.

## Expected predictive signal

The Copilot's ensemble congestion model should produce (approximately):

| Time (min) | `if_out_util_pct` on eth3 | `queue_depth_bytes[AF21-data]` | Risk band |
|---|---|---|---|
| 0 | 31 | 60 000 | NORMAL |
| 5 | 41 | 110 000 | NORMAL |
| 10 | 58 | 220 000 | WATCH |
| 15 | 71 | 380 000 | ELEVATED |
| 20 | 84 | 510 000 | ELEVATED |
| 25 | 92 | 670 000 | CRITICAL |
| 30 | 96 | 800 000 | CRITICAL (impact) |

The first ELEVATED alert is the key validation point. The Copilot should
emit a `P3` alert with `risk_band: ELEVATED` at **T+15 minutes**, with a
predicted `time_to_impact_minutes` in the 10\u201320 minute range. That gives
the operator 15 minutes of lead time before the actual SLA breach at T+30.

## Expected Copilot response shape

```text
schema_version: 1.0.0
answer_grounded: true
predicted_issue.type: congestion_saturation
predicted_issue.target.device: pe-hub-east-1
predicted_issue.target.interface_or_peer: eth3
predicted_issue.target.vrf: cust-blue
predicted_issue.time_to_impact_minutes: ~12
root_cause_hypothesis.summary: cites the rising if_out_util_pct and queue depth,
                                 and references the AF21 queue starvation risk
evidence_chunks[*]: MUST include the topology chunk for pe-hub-east-1 and
                    a runbook chunk that contains a reroute procedure
recommended_actions[0].action: reroute_te
recommended_actions[0].target: pe-hub-east-1 eth3
recommended_actions[0].risk: low
recommended_actions[0].linked_runbook_chunk_id: must point to a real chunk
operator_questions.q1: "The pe-hub-east-1 \u2194 p-core-1 link will saturate in ~12 minutes,
                         dropping voice and video on cust-blue."
```

## Fault-injection primitives

| Layer | Tool | Command (executed inside the containerlab namespace) |
|---|---|---|
| Traffic | `tc` (in `ce-branch-3`) | `tc qdisc add dev eth5 root tbf rate 34000Mbit burst 32Mb latency 50ms` (piecewise reconfigured by the runner). |
| Verification | SNMP poll from telegraf | `snmpget -v2c -c public 172.20.0.20 .1.3.6.1.2.1.2.2.1.10.3` (ifInOctets on eth3) and `.1.3.6.1.2.1.2.2.1.16.3` (ifOutOctets). |
| Cross-check | Prometheus query | `rate(if_out_octets_total{device="pe-hub-east-1",interface="eth3"}[1m]) * 8 / 40e9 * 100` (utilisation %). |

## Validation metrics captured

| Metric | How | Pass criterion |
|---|---|---|
| `lead_time_minutes` | Time of first ELEVATED alert minus T+0 of the workload ramp. | `>= 12` |
| `time_to_impact_error_minutes` | `|predicted_tti - actual_tti|`. | `<= 5` |
| `grounding_recall_at_k` | Does the runbook chunk cited by the action contain the reroute step? | `== 1.0` |
| `fabrication_rate` | Did the model cite any `chunk_id` not in the index? | `== 0.0` |
| `action_applicability` | Would following `recommended_actions[0]` have prevented the breach (counterfactual check via the runner)? | `>= 0.8` |

## Rollback / cleanup

After the run, the runner removes the `tc` rules and resets
`pe-hub-east-1`'s `mpls traffic-eng interface eth3 metric` to its
default. The Prometheus recording rules are reset by the test harness.
