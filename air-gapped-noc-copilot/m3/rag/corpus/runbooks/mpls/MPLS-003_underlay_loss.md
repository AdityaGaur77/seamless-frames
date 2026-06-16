---
id: MPLS-003
title: "MPLS underlay packet loss and tunnel degradation"
protocol: MPLS
severity: P1
last_reviewed: "2026-05-10"
approved_by: "mpls-architect"
applies_to: ["p-*", "pe-*"]
---

# Symptom

Packet loss on an MPLS underlay link progresses from <0.1 % to >2 % over a 15 \u2013 30
minute window. The loss is visible on the LDP session between the affected P/PE and
on the SD-WAN IPSec tunnels that ride that underlay as an increase in rekey retries
and a slow rise in tunnel jitter. No alarms have fired on the link \u2014 the threshold
breach has not yet occurred.

# Detection signal

- `if_in_errors_rate` on the underlay interface rising for > 10 minutes.
- `ldp_session_holdtime` decreasing linearly for > 5 minutes.
- `sdwan_tunnel_jitter_ms` rising for > 10 minutes on tunnels whose underlay includes
  the affected link.
- The Copilot's tunnel health degradation model flags the affected tunnel as
  `risk_band: ELEVATED` with `lead_time_minutes >= 15`.

# Step-by-step

1. Identify the underlay link using the affected P/PE loopbacks:
   `show mpls ldp neighbor` and `show mpls interface`.
2. Check `show interface counters errors` on both ends of the link.
3. Check `show logging` for `LINK-3-UPDOWN` and `LINK-4-ERROR` events on the
   interface in the last 30 minutes.
4. Check the optical layer (DOM) for the affected SFP/QSFP if available:
   `show interface transceiver detail`.
5. If the link is fibre, schedule a fibre OTDR test with the facilities team within
   the Copilot's recommended lead time.
6. Apply a soft pre-emptive reroute by raising the TE cost of the affected link by
   1000 using `mpls traffic-eng interface <intf> metric 2000` (rollback step 8).
7. Monitor `if_in_errors_rate` and `ldp_session_holdtime` for 10 minutes; if both
   stabilise or improve, leave the reroute in place until the fibre is repaired.
8. Rollback: revert the TE cost to the original value with
   `no mpls traffic-eng interface <intf> metric` and reapply the default config.

# Escalation

- **P1** if any `cust-blue` tunnel is in the `risk_band: CRITICAL` band.
- **P2** if only `mgmt` traffic is affected.
- **P3** if the loss is contained to a single non-redundant link and no tunnels
  are degraded.

# Prevention

- Run DOM polling every 60 seconds and forward to the Copilot's anomaly detector.
- Maintain dual underlays on all SD-WAN edge devices; the topology must have
  `redundancy_class >= 2` for the `cust-blue` VRF.
- Quarterly OTDR sweep of all backbone fibre spans.
