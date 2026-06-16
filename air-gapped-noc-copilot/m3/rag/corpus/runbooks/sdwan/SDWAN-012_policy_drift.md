---
id: SDWAN-012
title: "SD-WAN policy drift detection and rollback"
protocol: SDWAN
severity: P2
last_reviewed: "2026-05-15"
approved_by: "sdwan-architect"
applies_to: ["ce-*"]
---

# Symptom

The SD-WAN controller's intended policy (e.g. `qos-cust-blue` attached to VRF
`cust-blue` on PE `pe-hub-east-1`) is not what is currently rendered on the device.
Symptoms include:

- A QoS class that should be present (`voice`, `video`, `data`, `scavenger`) is
  missing or has the wrong bandwidth percent.
- A VPN membership change: a branch is in a VRF that the controller does not
  advertise.
- A tunnel policy that has been locally overridden on a CE.

# Detection signal

- `policy_drift_score` (Copilot-derived metric) > 0.4 for > 10 minutes.
- `policy_expected_class_count != policy_actual_class_count` for any policy
  attached to a VRF.
- `controller_intended_state_hash != device_actual_state_hash` reported by the
  controller's reconcile loop.

# Step-by-step

1. From the Copilot's structured alert, identify the device, the policy, and the
   drift field(s).
2. On the device, run `show policy-map` and `show running-config | include policy`
   to confirm the drift.
3. Cross-check the controller's intended config in the local topology store
   (`corpus/topology/topology_reference.yaml` \u2192 `policies[]`).
4. If the controller's intended config is the source of truth, trigger a
   controller-driven re-deploy: `controller-cli policy apply --device <id>
   --policy <policy-id>`.
5. If the device's local config is the source of truth (controller was wrong),
   open a change ticket and update the controller's intended state to match;
   re-pin the controller config so future drifts trip the alarm.
6. If neither side matches the runbook baseline, escalate to the SD-WAN
   architect; do not "fix" by hand-editing the device while the controller is
   still in reconciliation mode.

# Escalation

- **P1** if the drift removes the `voice` class from a `cust-blue` policy on a
  hub PE.
- **P2** otherwise.

# Prevention

- Pin all policies in the controller and reject edits that change `bw_pct`
  values outside the approved band.
- Run a nightly reconciliation job and surface any drift in the Copilot's
  morning digest.
- Maintain a `controller_intended_state_hash` per (device, policy) pair in the
  local vector store; the Copilot uses this for fast retrieval of the exact
  intended config when answering a drift alert.
