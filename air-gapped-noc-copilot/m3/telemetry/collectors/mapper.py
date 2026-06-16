"""OID \u2192 Prometheus metric mapping table.

Centralised so the SNMP collector, the NetFlow collector, and the
predictive engine all use the same metric names and label sets.

Conventions:
  - All metrics are lower_snake_case.
  - Counter metrics end in `_total`; gauges have no suffix.
  - Every metric carries the labels: `device`, `site` (if known),
    `role` (if known), `interface_or_peer` (if applicable), `vrf`
    (if applicable).
  - Units in the metric name (e.g. `_octets`, `_seconds`, `_ms`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OidMap:
    metric: str
    oid: str
    type: str
    description: str
    label_from_oid: str | None = None


IFACE_OIDS: list[OidMap] = [
    OidMap("interface_ifInOctets_total", ".1.3.6.1.2.1.2.2.1.10", "counter", "Bytes received on the interface."),
    OidMap("interface_ifOutOctets_total", ".1.3.6.1.2.1.2.2.1.16", "counter", "Bytes sent on the interface."),
    OidMap("interface_ifInErrors_total", ".1.3.6.1.2.1.2.2.1.14", "counter", "Inbound packets that contained errors."),
    OidMap("interface_ifOutErrors_total", ".1.3.6.1.2.1.2.2.1.20", "counter", "Outbound packets that contained errors."),
    OidMap("interface_ifInDiscards_total", ".1.3.6.1.2.1.2.2.1.13", "counter", "Inbound packets discarded."),
    OidMap("interface_ifOutDiscards_total", ".1.3.6.1.2.1.2.2.1.19", "counter", "Outbound packets discarded."),
    OidMap("interface_ifHCInOctets_total", ".1.3.6.1.2.1.31.1.1.1.6", "counter", "High-capacity 64-bit inbound octet counter."),
    OidMap("interface_ifHCOutOctets_total", ".1.3.6.1.2.1.31.1.1.1.10", "counter", "High-capacity 64-bit outbound octet counter."),
    OidMap("interface_ifHighSpeed", ".1.3.6.1.2.1.31.1.1.1.15", "gauge", "Nominal interface speed in Mbps."),
    OidMap("interface_ifSpeed", ".1.3.6.1.2.1.2.2.1.5", "gauge", "Interface speed in bits per second."),
    OidMap("interface_ifAdminStatus", ".1.3.6.1.2.1.2.2.1.7", "gauge", "Admin status (1=up,2=down,3=testing,4=unknown,5=dormant,6=notPresent,7=lowerLayerDown)."),
    OidMap("interface_ifOperStatus", ".1.3.6.1.2.1.2.2.1.8", "gauge", "Operational status (same enum as ifAdminStatus)."),
    OidMap("interface_ifName", ".1.3.6.1.2.1.31.1.1.1.1", "label", "Interface name."),
    OidMap("interface_ifAlias", ".1.3.6.1.2.1.31.1.1.1.18", "label", "Interface alias / description."),
    OidMap("interface_ifDescr", ".1.3.6.1.2.1.2.2.1.2", "label", "Interface description (IF-MIB)."),
]


BGP_OIDS: list[OidMap] = [
    OidMap("bgp_peer_state", ".1.3.6.1.2.1.15.3.1.1.2", "gauge", "BGP peer FSM state (1=idle,2=connect,3=active,4=opensent,5=openconfirm,6=established)."),
    OidMap("bgp_peer_remote_addr", ".1.3.6.1.2.1.15.3.1.1.7", "label", "Remote BGP peer IP."),
    OidMap("bgp_peer_remote_as", ".1.3.6.1.2.1.15.3.1.1.9", "label", "Remote BGP peer ASN."),
    OidMap("bgp_peer_in_updates_total", ".1.3.6.1.2.1.15.3.1.1.10", "counter", "Number of BGP UPDATE messages received."),
    OidMap("bgp_peer_out_updates_total", ".1.3.6.1.2.1.15.3.1.1.11", "counter", "Number of BGP UPDATE messages sent."),
    OidMap("bgp_peer_in_total_messages_total", ".1.3.6.1.2.1.15.3.1.1.12", "counter", "Total BGP messages received."),
    OidMap("bgp_peer_out_total_messages_total", ".1.3.6.1.2.1.15.3.1.1.13", "counter", "Total BGP messages sent."),
    OidMap("bgp_peer_fsm_established_time_seconds", ".1.3.6.1.2.1.15.3.1.1.16", "gauge", "Time the peer has been in established state, in seconds."),
    OidMap("bgp_peer_in_update_elapsed_time_seconds", ".1.3.6.1.2.1.15.3.1.1.19", "gauge", "Time since the last BGP UPDATE was received, in seconds."),
]


LDP_OIDS: list[OidMap] = [
    OidMap("ldp_session_state", ".1.3.6.1.4.1.9.9.187.1.2.2.1.3", "gauge", "LDP session state (0=unknown,1=operational,2=non-existent)."),
    OidMap("ldp_session_holdtime_seconds", ".1.3.6.1.4.1.9.9.187.1.2.2.1.7", "gauge", "LDP session negotiated hold time in seconds."),
    OidMap("ldp_session_uptime_seconds", ".1.3.6.1.4.1.9.9.187.1.2.2.1.9", "gauge", "Time since the LDP session came up, in seconds."),
    OidMap("ldp_session_peer_label_space_id", ".1.3.6.1.4.1.9.9.187.1.2.2.1.2", "label", "LDP peer LSR identifier."),
]


SDWAN_TUNNEL_OIDS: list[OidMap] = [
    OidMap("sdwan_tunnel_up", ".1.3.6.1.4.1.9.9.187.0.1", "gauge", "SD-WAN tunnel up (1) or down (0)."),
    OidMap("sdwan_tunnel_jitter_ms", ".1.3.6.1.4.1.9.9.187.0.2", "gauge", "SD-WAN tunnel jitter in milliseconds (p95 over the last interval)."),
    OidMap("sdwan_tunnel_loss_pct", ".1.3.6.1.4.1.9.9.187.0.3", "gauge", "SD-WAN tunnel loss percentage over the last interval."),
    OidMap("sdwan_tunnel_rekey_retries_total", ".1.3.6.1.4.1.9.9.187.0.4", "counter", "Number of SD-WAN tunnel rekey retries."),
]


ALL_OIDS: list[OidMap] = IFACE_OIDS + BGP_OIDS + LDP_OIDS + SDWAN_TUNNEL_OIDS


TOPOLOGY_FILE_OIDS: list[OidMap] = [
    OidMap("system_hostname", ".1.3.6.1.2.1.1.5.0", "label", "System name (hostname)."),
    OidMap("system_uptime_seconds", ".1.3.6.1.2.1.1.3.0", "gauge", "System uptime in hundredths of a second."),
]
