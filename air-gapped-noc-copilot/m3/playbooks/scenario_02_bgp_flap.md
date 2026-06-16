# Scenario 2 — BGP route flap with downstream path reroute cascade

## Goal

Validate that the Copilot detects **precursor** route-flap behaviour on
an iBGP session between a branch CE and a hub PE, **before** the
session actually fails, and that the recommendation isolates the
affected site to prevent the downstream cascade from spilling into the
datacenter VRF.

## Fault model

A controlled route-flap is induced on the BGP session
`ce-branch-1 \u2194 pe-hub-east-1` (cust-blue). The session is held
`Established` but the CE periodically withdraws and re-advertises its
two most specific prefixes (`10.10.1.0/24`, `10.10.1.128/25`) at a
rising cadence:

| Minute | Cadence (s) | Notes |
|---|---|---|
| 0 \u2013 5 | 60 | baseline (1 flap / min) |
| 6 \u2013 12 | 20 | rising |
| 13 \u2013 18 | 6 | rising |
| 19 \u2013 22 | 2 | high; downstream `pe-dc-1` will see the cascade |
| 23+ | 0.5 | session will fail around T+24 |

The objective is to detect the rising cadence **before** the session
fails. The downstream PE-DCs (`pe-dc-1`, `pe-dc-2`) see the cascade via
iBGP route-reflector propagation; `pe-dc-1` will start preferring
`pe-hub-west-1` for `10.10.1.0/24` as `pe-hub-east-1`'s path goes
unstable, then the BGP path selection flips back when the session
finally fails \u2014 a classic precursor of a full PE failure.

## Expected predictive signal

The Copilot's `bgp-flap-detector` and `path-asymmetry-detector` models
should produce:

| Time (min) | `bgp_session_flap_rate` (flaps/min) | `best_path_changes_total` (on pe-dc-1) | Risk band |
|---|---|---|---|
| 0 | 1 | 0 | NORMAL |
| 10 | 3 | 0 | WATCH |
| 15 | 10 | 1 | ELEVATED |
| 20 | 30 | 4 | ELEVATED |
| 23 | 60 | 9 | CRITICAL |
| 24 | session down | \u2014 | CRITICAL (impact) |

The Copilot should emit a P2 alert at **T+15** with `risk_band: ELEVATED`
and `time_to_impact_minutes` around 8.

## Expected Copilot response shape

```text
schema_version: 1.0.0
answer_grounded: true
predicted_issue.type: bgp_session_flap
predicted_issue.target.device: ce-branch-1
predicted_issue.target.interface_or_peer: pe-hub-east-1 (cust-blue)
predicted_issue.time_to_impact_minutes: ~8
root_cause_hypothesis.summary:
    cites bgp_session_flap_rate and best_path_changes_total
evidence_chunks[*]:
    - BGP-007 runbook (neighbor stuck in Active / flap remediation)
    - topology chunk for the branch-hub-DC path
    - INC history chunk if any prior flap on this peer exists
recommended_actions[0].action: lower_bfd_timer
recommended_actions[0].target: ce-branch-1 peer pe-hub-east-1
recommended_actions[0].linked_runbook_chunk_id: must point to BGP-007
recommended_actions[1].action: preempt_failover
recommended_actions[1].target: pe-dc-1 cust-blue traffic steering
operator_questions.q1: "ce-branch-1 BGP session will fail in ~8 minutes; downstream
                         PE path selection will cascade to pe-dc-1."
operator_questions.q3: "Lower BFD timers per BGP-007 and pre-emptively steer
                         pe-dc-1 cust-blue traffic to pe-hub-west-1."
```

## Fault-injection primitives

| Layer | Tool | Command |
|---|---|---|
| Flap induction | `vtysh` (in `ce-branch-1`) | per minute: `clear ip bgp 10.255.1.1 soft out` (withdrawn prefixes re-advertise). |
| BFD verification | SNMP poll | `snmpget ... .1.3.6.1.4.1.9.9.187.0.x` (BFD session state). |
| Cross-check | Prometheus | `rate(bgp_session_flap_total{peer="ce-branch-1"}[1m])` and `increase(best_path_changes_total{device="pe-dc-1"}[1m])`. |

## Validation metrics

| Metric | Pass criterion |
|---|---|
| `lead_time_minutes` | `>= 8` |
| `path_cascade_contained` | Did following `recommended_actions[1]` actually prevent the downstream path change from reaching `pe-dc-1`? | `== true` |
| `grounding_recall_at_k` | `== 1.0` (BGP-007 must be cited) |
| `action_applicability` | Would `lower_bfd_timer` on this peer have surfaced the flap as a BFD-down event? | `>= 0.8` |

## Rollback / cleanup

Restore the original flap cadence to baseline, then clear any
`clear ip bgp` soft-out operations left in flight. Reset the BFD timer
to its baseline (50/50/3).
