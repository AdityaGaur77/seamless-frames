---
id: BGP-007
title: "BGP neighbor stuck in Active"
protocol: BGP
severity: P2
last_reviewed: "2026-04-22"
approved_by: "neteng-lead"
applies_to: ["pe-*", "ce-*"]
---

# Symptom

A configured BGP neighbor on a PE or CE device stays in the **Active** state for
more than 3 minutes (the standard hold time window). The peer is configured but no
TCP/179 session is ever established. In SD-WAN edge deployments, this often shows up
as a precursor to a full route-flap cascade.

# Detection signal

- `bgp_4_bgp_session_state{peer=*,af=ipv4} == 2` (Active) for > 180s.
- `bgp_session_established_total{peer=*} rate == 0` for the affected peer.
- No `BGP-5-ADJCHANGE` log lines in the last 5 minutes.

# Step-by-step

1. Confirm the peer IP and ASN with `show ip bgp summary` on the affected device.
2. Verify L3 reachability to the peer with `ping <peer-ip> source <local-iface>`
   (10 packets). If the ping fails, escalate to the underlay runbook MPLS-003.
3. Verify TCP/179 reachability with `show tcp vrf <vrf>` \u2014 look for an entry where
   the local port is 179 and the state is not ESTABLISHED.
4. Check that the peer ASN matches what is configured locally. A single digit typo
   will hold the session in Active indefinitely.
5. If the peer is correct, check inbound ACLs on the peer device for `deny tcp any
   any eq 179`.
6. Capture an `tcpdump` on the local interface for 60 seconds while attempting to
   bring the session up, and look for SYN retransmissions.
7. If the session still does not establish, roll back any recent configuration
   changes on either side and escalate to the BGP-L3 team.

# Escalation

- **P2** if a production VRF (`cust-blue`, `cust-green`) is affected.
- **P3** if only the `mgmt` VRF is affected.
- **P4** if the peer is a known lab / test neighbor and the impact is contained.

# Prevention

- Deploy BFD (`bfd profile bgp-fast` with `interval 50 min_rx 50 multiplier 3`) on all
  production BGP sessions. BFD-detected failures are surfaced as the
  `bgp_session_down_total` rate in the Copilot's predictive engine and trigger
  pre-emptive reroute before user-visible impact.
- Pin the BGP configuration in the controller and reject drift. The Copilot's policy
  drift playbook (`PD-001`) covers detection and response.
- Run a weekly `show ip bgp summary` audit and surface any neighbour that has been
  in `Active` for > 60 seconds at any point in the last 7 days.
