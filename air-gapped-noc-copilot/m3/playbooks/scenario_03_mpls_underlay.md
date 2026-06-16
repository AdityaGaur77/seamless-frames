# Scenario 3 — Intermittent MPLS underlay failure with tunnel degradation

## Goal

Validate that the Copilot detects the **multi-signal precursor** of an
MPLS underlay failure (rising CRC, falling LDP holdtime, rising SD-WAN
tunnel jitter) **before** voice impact on the affected SD-WAN tunnel,
and that the cited runbook (`MPLS-003`) is the correct one and the
recommended action sequence is operationally safe.

## Fault model

The fibre path between `pe-hub-east-1` and `p-core-1` (link
`pe-hub-east-1--p-core-1`, 40G) is degraded by injecting progressive CRC
errors and small packet drops using `netem` on the containerlab veth
pair. The drop probability is ramped piecewise:

| Minute | Drop probability | CRC error rate (/s) |
|---|---|---|
| 0 \u2013 5 | 0.001 | 0.5 |
| 6 \u2013 12 | 0.005 | 4 |
| 13 \u2013 20 | 0.020 | 28 |
| 21 \u2013 25 | 0.050 | 80 (LDP holdtime dropping) |
| 26+ | 0.080 | 150 (voice impact on the SD-WAN tunnel) |

The SD-WAN IPSec tunnel `sdwan-nyc-sfo` rides this underlay, so its
jitter p95 rises in lockstep. Voice packets on the tunnel are dropped at
~0.5 % once the underlay loss exceeds 5 %.

## Expected predictive signal

| Time (min) | `if_in_errors_rate` (CRC) | `ldp_session_holdtime` (s) | `sdwan_tunnel_jitter_ms` p95 | Risk band |
|---|---|---|---|---|
| 0 | 0.5 | 15.0 | 3.5 | NORMAL |
| 10 | 4 | 14.8 | 4.0 | WATCH |
| 17 | 28 | 13.5 | 8.0 | ELEVATED |
| 23 | 80 | 11.0 | 14.0 | ELEVATED |
| 25 | 80 | 10.0 | 18.0 | CRITICAL |
| 26 | 150 | \u2014 | 28.0 (voice impact) | CRITICAL (impact) |

The Copilot's `tunnel-health-v2` model should fuse all three signals
into a single `risk_band` and emit a P1 alert at **T+17** with
`time_to_impact_minutes` in the 7\u201310 minute window.

## Expected Copilot response shape

```text
schema_version: 1.0.0
answer_grounded: true
predicted_issue.type: underlay_packet_loss
predicted_issue.target.device: pe-hub-east-1
predicted_issue.target.interface_or_peer: eth1
predicted_issue.time_to_impact_minutes: ~8
root_cause_hypothesis.summary:
    rising CRC + falling LDP holdtime + rising tunnel jitter is the textbook
    precursor for MPLS underlay fibre degradation; INC-2026-03-14-001 is the
    same signal pattern
evidence_chunks[*]:
    - MPLS-003 runbook (with rekey retry + TE reroute steps)
    - INC-2026-03-14-001 incident detail (RCA: fibre splice)
    - topology chunk for the link
recommended_actions[0].action: open_change_ticket      (BEFORE reroute, per R9)
recommended_actions[0].linked_runbook_chunk_id: must be a real MPLS-003 chunk
recommended_actions[1].action: reroute_te
recommended_actions[1].target: pe-hub-east-1 eth1
recommended_actions[2].action: page_oncall
recommended_actions[2].target: facilities-team
operator_questions.q3: "Open a change ticket, raise the TE cost on pe-hub-east-1
                         eth1 by 1000 per MPLS-003, and page facilities to
                         dispatch an OTDR for the east ring."
```

## Fault-injection primitives

| Layer | Tool | Command (on the `pe-hub-east-1:eth1` veth) |
|---|---|---|
| Drop + CRC | `tc netem` | `tc qdisc add dev eth1 root netem loss 0.5% corrupt 0.1%` (ramped by the runner). |
| Verification | SNMP | `.1.3.6.1.2.1.2.2.1.14.1` (ifInErrors). |
| LDP | `vtysh` | `show mpls ldp neighbor` for holdtime. |
| Tunnel | SNMP / NetFlow | `sdwan_tunnel_jitter_ms` from the SD-WAN controller exporter. |

## Validation metrics

| Metric | Pass criterion |
|---|---|
| `lead_time_minutes` | `>= 7` |
| `multisignal_fusion_correct` | Did the model fuse the three signals (CRC, LDP, jitter) and not just one? | `== true` |
| `incident_precedent_cited` | Did the model cite INC-2026-03-14-001? | `== true` |
| `action_ordering_compliant` | Was the change ticket ordered **before** the TE reroute, per R9? | `== true` |
| `grounding_recall_at_k` | `== 1.0` |
| `fabrication_rate` | `== 0.0` |

## Rollback / cleanup

`tc qdisc del dev eth1 root` to remove the netem rule. Restore
`mpls traffic-eng interface eth1 metric` to default. Confirm
`if_in_errors_rate` returns to baseline within 60 seconds.
