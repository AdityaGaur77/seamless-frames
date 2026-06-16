# Scenario 4 — Controller misconfiguration leading to policy drift

## Goal

Validate that the Copilot detects **policy drift** between the SD-WAN
controller's intended state and the device's actual rendered state, and
that the recommended action is the correct sequence
(`open_change_ticket` \u2192 `pin_policy` \u2192 `rekey_tunnel` if needed),
not a blind re-deploy that would mask the drift source.

## Fault model

The SD-WAN controller is reconfigured to attach a new policy
`qos-cust-blue-v2` to the `cust-blue` VRF on `pe-hub-east-1`. The new
policy has a **different** `bw_pct` for the `voice` class (5 % instead
of 10 %), and adds a `scavenger` class with 15 %. The controller is
*not* yet pushed to the device; the device still has the original
`qos-cust-blue`. The Copilot's reconciliation loop detects the drift
within 2 minutes.

After 10 minutes, the operator (simulated) inadvertently pushes the
controller's intended config to the device, masking the source of the
drift. The Copilot's evidence chunk for the runbook
(`SDWAN-012`) warns against doing this and recommends pinning the
controller's intended state in the audit log instead.

## Expected predictive signal

| Time (min) | `policy_drift_score` | `policy_expected_class_count != policy_actual_class_count` | Risk band |
|---|---|---|---|
| 0 | 0.0 | false | NORMAL |
| 2 | 0.55 | true | ELEVATED |
| 10 | 0.55 | true (still drifted) | ELEVATED |
| 12 | 0.55 | true (operator pushed the drift) | ELEVATED |
| 13 | 0.55 | false (drift now in the rendered config) | WATCH (if not pinned) |

The Copilot should emit a P2 alert at **T+2** with `time_to_impact_minutes`
in the 30 \u2013 60 minute window (because voice impact depends on the actual
load profile; the drift itself is a P2 because of the policy change).

## Expected Copilot response shape

```text
schema_version: 1.0.0
answer_grounded: true
predicted_issue.type: policy_drift
predicted_issue.target.device: pe-hub-east-1
predicted_issue.target.interface_or_peer: vrf=cust-blue
predicted_issue.time_to_impact_minutes: ~45
root_cause_hypothesis.summary:
    controller intended state for policy qos-cust-blue on pe-hub-east-1
    does not match the device's rendered state; the voice class bandwidth
    is 5% in the controller but 10% on the device, and a new scavenger
    class at 15% exists in the controller only
evidence_chunks[*]:
    - SDWAN-012 runbook (drift detection and rollback)
    - topology chunk for the policy
    - controller_intended_state_hash from the local vector store
recommended_actions[0].action: open_change_ticket    (BEFORE any push)
recommended_actions[1].action: pin_policy             (pin controller state, do not re-deploy)
recommended_actions[2].action: escalate_to_team      (SD-WAN architect)
operator_questions.q3: "Open a change ticket, pin the controller's intended
                         state for qos-cust-blue on pe-hub-east-1, and
                         escalate to the SD-WAN architect. Do NOT push the
                         controller's intended state to the device \u2014 the
                         SDWAN-012 runbook warns that this would mask the
                         source of the drift."
```

## Fault-injection primitives

| Layer | Tool | Command |
|---|---|---|
| Controller state edit | SD-WAN controller REST (simulated by editing the YAML in `corpus/topology/policies`) | bump `bw_pct` of `voice` class to 5, add `scavenger` class. |
| Reconciliation loop | Cron in the Copilot runner | runs every 60 seconds, diffs `controller_intended_state_hash` against `device_actual_state_hash` from SNMP/`show policy-map`. |
| Verification | SNMP | `policy_drift_score` (Copilot-computed) and `policy_class_count{policy="qos-cust-blue"}`. |

## Validation metrics

| Metric | Pass criterion |
|---|---|
| `drift_detected_within_seconds` | Time from controller edit to first ELEVATED alert. | `<= 150` |
| `recommended_action_not_redeploy` | The model did **not** recommend `rekey_tunnel` or a blind `pin_policy` without first recommending `open_change_ticket` + `escalate_to_team`. | `== true` |
| `controller_state_pinned_in_evidence` | `provenance.evidence_chunks` includes the topology chunk with the intended policy. | `== true` |
| `grounding_recall_at_k` | `== 1.0` (SDWAN-012 must be cited) |
| `fabrication_rate` | `== 0.0` |

## Rollback / cleanup

Revert the controller's policy edit (back to 10 % voice, no extra
scavenger). The runner's reconciliation loop should resolve
`policy_drift_score` to 0.0 within 60 seconds and the alert should
auto-clear.
