---
id: INC-2026-03-14-001
date: "2026-03-14"
duration_minutes: 47
affected_sites: [hub-east, branch-nyc, branch-sfo]
mttr_minutes: 32
root_cause_class: "underlay_loss_progression"
linked_runbook_id: "MPLS-003"
signals: ["if_in_errors_rate", "ldp_session_holdtime", "sdwan_tunnel_jitter_ms"]
---

## Symptom

From 09:14 UTC the Copilot's predictive engine reported a slowly rising
`if_in_errors_rate` on the `pe-hub-east-1 \u2194 p-core-1` 40G link. Over the next 18
minutes the `ldp_session_holdtime` started decreasing linearly. The Copilot
emitted a P3 alert at 09:21 UTC with `risk_band: ELEVATED` and a predicted
`time_to_impact_minutes` of 27. At 09:32 UTC the link crossed the 2 % loss
threshold and the SD-WAN tunnel between `ce-branch-1` and `ce-branch-3`
started dropping voice packets. The Copilot escalated to P1 and surfaced the
MPLS-003 runbook, which was followed verbatim. Fibre OTDR at 10:01 UTC found
a degraded splice 4.2 km from the east hub. The link was rerouted to a
secondary path and the splice was repaired during the maintenance window.

## Root cause

A fibre splice in the east ring degraded over several days, producing CRC
errors that gradually exceeded the link's FEC correction capacity. The Copilot
had flagged the rising error rate 11 minutes before the threshold was crossed
and 18 minutes before user-visible voice impact.

## Detection signal

- `if_in_errors_rate` (CRC, fragment, giant) on the 40G link rose from baseline
  0.3 / s to 41 / s over 18 minutes.
- `ldp_session_holdtime` on the P-PE adjacency decreased from 15 s to 8 s.
- `sdwan_tunnel_jitter_ms` for the `sdwan-nyc-sfo` tunnel rose from 4 ms p95
  to 18 ms p95.
- The Copilot's `risk_band` progression was: `NORMAL` (09:00) \u2192 `WATCH`
  (09:14) \u2192 `ELEVATED` (09:21) \u2192 `CRITICAL` (09:32).

## Remediation

1. Copilot surfaced the MPLS-003 runbook at 09:32 UTC.
2. Operator increased the TE cost of the affected link by 1000 at 09:35 UTC.
3. Traffic rerouted to the secondary path within 90 s.
4. Voice impact cleared at 09:36 UTC.
5. Facilities dispatched at 09:40 UTC; OTDR completed at 10:01 UTC.
6. Splice repaired during the 11:00 UTC maintenance window.
7. TE cost reverted after a 24-hour soak with no error rate increase.

## Prevention

- DOM polling cadence was increased from 5 min to 60 s.
- The Copilot's `tunnel_health_degradation` model was retrained with this
  incident added to the labelled set; precision@1h-lead-time improved from
  0.71 to 0.84.
- The MPLS-003 runbook was updated to add a fibre OTDR step in the procedure
  and was re-reviewed on 2026-05-10.
