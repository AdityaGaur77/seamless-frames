#!/usr/bin/env python3
"""
Air-Gapped NOC Copilot - SD-WAN IPSec Overlay Configuration
Generates IPSec tunnel configurations between CE (branch) and PE (hub) routers.
Uses strongSwan-style syntax compatible with FRR's built-in IPSec support.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class IPSecTunnel:
    """Represents a single IPSec tunnel endpoint."""
    local_addr: str
    remote_addr: str
    local_id: str
    remote_id: str
    spi: str
    encryption: str = "aes-256-gcm"
    integrity: str = "sha256"
    dh_group: int = 20
    lifetime_sec: int = 3600
    tunnel_local: str = ""
    tunnel_remote: str = ""
    local_subnet: str = ""
    remote_subnet: str = ""
    metric: int = 10
    pre_shared_key: str = "N0c_C0p1l0t_2024!"
    dpd_delay: int = 10
    dpd_timeout: int = 30


@dataclass
class SDWANSite:
    """SD-WAN site definition with hub-spoke topology."""
    site_id: int
    name: str
    role: str  # "hub" or "spoke"
    loopback: str
    tunnels: List[IPSecTunnel] = field(default_factory=list)


# ── SD-WAN Topology Definition ──
SDWAN_SITES = {
    "pe-hub-east-1": SDWANSite(
        site_id=1,
        name="hub-east",
        role="hub",
        loopback="10.255.0.10",
        tunnels=[
            IPSecTunnel(
                local_addr="10.255.0.10",
                remote_addr="10.255.0.40",
                local_id="hub-east",
                remote_id="branch-1",
                spi="0xA001",
                tunnel_local="172.16.1.1",
                tunnel_remote="172.16.1.2",
                local_subnet="10.0.1.0/24",
                remote_subnet="10.0.11.0/24",
            ),
            IPSecTunnel(
                local_addr="10.255.0.10",
                remote_addr="10.255.0.41",
                local_id="hub-east",
                remote_id="branch-2",
                spi="0xA002",
                tunnel_local="172.16.2.1",
                tunnel_remote="172.16.2.2",
                local_subnet="10.0.1.0/24",
                remote_subnet="10.0.12.0/24",
            ),
        ],
    ),
    "pe-hub-west-1": SDWANSite(
        site_id=2,
        name="hub-west",
        role="hub",
        loopback="10.255.0.11",
        tunnels=[
            IPSecTunnel(
                local_addr="10.255.0.11",
                remote_addr="10.255.0.42",
                local_id="hub-west",
                remote_id="branch-3",
                spi="0xA003",
                tunnel_local="172.16.3.1",
                tunnel_remote="172.16.3.2",
                local_subnet="10.0.2.0/24",
                remote_subnet="10.0.13.0/24",
            ),
            IPSecTunnel(
                local_addr="10.255.0.11",
                remote_addr="10.255.0.43",
                local_id="hub-west",
                remote_id="branch-4",
                spi="0xA004",
                tunnel_local="172.16.4.1",
                tunnel_remote="172.16.4.2",
                local_subnet="10.0.2.0/24",
                remote_subnet="10.0.14.0/24",
            ),
        ],
    ),
    "ce-branch-1": SDWANSite(
        site_id=11,
        name="branch-1",
        role="spoke",
        loopback="10.255.0.40",
        tunnels=[
            IPSecTunnel(
                local_addr="10.255.0.40",
                remote_addr="10.255.0.10",
                local_id="branch-1",
                remote_id="hub-east",
                spi="0xB001",
                tunnel_local="172.16.1.2",
                tunnel_remote="172.16.1.1",
                local_subnet="10.0.11.0/24",
                remote_subnet="10.0.1.0/24",
            ),
        ],
    ),
    "ce-branch-2": SDWANSite(
        site_id=12,
        name="branch-2",
        role="spoke",
        loopback="10.255.0.41",
        tunnels=[
            IPSecTunnel(
                local_addr="10.255.0.41",
                remote_addr="10.255.0.10",
                local_id="branch-2",
                remote_id="hub-east",
                spi="0xB002",
                tunnel_local="172.16.2.2",
                tunnel_remote="172.16.2.1",
                local_subnet="10.0.12.0/24",
                remote_subnet="10.0.1.0/24",
            ),
        ],
    ),
    "ce-branch-3": SDWANSite(
        site_id=13,
        name="branch-3",
        role="spoke",
        loopback="10.255.0.42",
        tunnels=[
            IPSecTunnel(
                local_addr="10.255.0.42",
                remote_addr="10.255.0.11",
                local_id="branch-3",
                remote_id="hub-west",
                spi="0xB003",
                tunnel_local="172.16.3.2",
                tunnel_remote="172.16.3.1",
                local_subnet="10.0.13.0/24",
                remote_subnet="10.0.2.0/24",
            ),
        ],
    ),
    "ce-branch-4": SDWANSite(
        site_id=14,
        name="branch-4",
        role="spoke",
        loopback="10.255.0.43",
        tunnels=[
            IPSecTunnel(
                local_addr="10.255.0.43",
                remote_addr="10.255.0.11",
                local_id="branch-4",
                remote_id="hub-west",
                spi="0xB004",
                tunnel_local="172.16.4.2",
                tunnel_remote="172.16.4.1",
                local_subnet="10.0.14.0/24",
                remote_subnet="10.0.2.0/24",
            ),
        ],
    ),
}


def generate_ipsec_tunnel_config(tunnel: IPSecTunnel, node_name: str) -> str:
    """Generate FRR IPSec tunnel configuration for a single tunnel."""
    tunnel_name = f"ipsec-tunnel-{tunnel.remote_id}"
    
    lines = [
        f"! ── IPSec Tunnel: {tunnel.local_id} <-> {tunnel.remote_id} ──",
        f"interface {tunnel_name}",
        f" description IPSec overlay to {tunnel.remote_id}",
        f" ip address {tunnel.tunnel_local}/30",
        f" ip ospf network point-to-point",
        f" tunnel source {tunnel.local_addr}",
        f" tunnel destination {tunnel.remote_addr}",
        f" tunnel mode ipsec ipv4",
        f" tunnel protection ipsec profile sd-wan-profile",
        f" ip ospf cost {tunnel.metric}",
        f" no shutdown",
        f"!",
        "",
    ]
    return "\n".join(lines)


def generate_crypto_config(node_name: str, site: SDWANSite) -> str:
    """Generate strongSwan/IPSec crypto configuration."""
    lines = [
        f"! ── IPSec Crypto Configuration for {node_name} ──",
        f"!",
    ]
    
    for tunnel in site.tunnels:
        lines.extend([
            f"! Tunnel to {tunnel.remote_id}",
            f"crypto ikev2 proposal sd-wan-proposal-{tunnel.remote_id}",
            f" encryption {tunnel.encryption}",
            f" integrity {tunnel.integrity}",
            f" group {tunnel.dh_group}",
            f"!",
            f"",
            f"crypto ikev2 policy sd-wan-policy-{tunnel.remote_id}",
            f" proposal sd-wan-proposal-{tunnel.remote_id}",
            f" match address local {tunnel.local_addr}",
            f" match address remote {tunnel.remote_addr}",
            f" match identity local id {tunnel.local_id}",
            f" match identity remote id {tunnel.remote_id}",
            f"!",
            f"",
            f"crypto ipsec profile sd-wan-profile-{tunnel.remote_id}",
            f" ikev2-profile sd-wan-policy-{tunnel.remote_id}",
            f" pfs group {tunnel.dh_group}",
            f" set security-association lifetime seconds {tunnel.lifetime_sec}",
            f"!",
            f"",
            f"! Pre-shared key for {tunnel.remote_id}",
            f"crypto ikev2 key {tunnel.remote_id}-key",
            f" address {tunnel.remote_addr}",
            f" pre-shared-key {tunnel.pre_shared_key}",
            f"!",
            f"",
        ])
    
    return "\n".join(lines)


def generate_ipsec_interfaces(node_name: str, site: SDWANSite) -> str:
    """Generate IPSec tunnel interface configurations."""
    lines = []
    
    for tunnel in site.tunnels:
        tunnel_name = f"ipsec-tun-{tunnel.remote_id}"
        lines.extend([
            f"! ── IPSec Tunnel Interface: {tunnel_name} ──",
            f"interface {tunnel_name}",
            f" description SD-WAN tunnel to {tunnel.remote_id}",
            f" ip address {tunnel.tunnel_local}/30",
            f" tunnel source {tunnel.local_addr}",
            f" tunnel destination {tunnel.remote_addr}",
            f" tunnel mode gre ipsec",
            f" tunnel protection ipsec profile sd-wan-profile-{tunnel.remote_id}",
            f" ip ospf network point-to-point",
            f" ip ospf cost {tunnel.metric}",
            f" keepalive 10 3",
            f" no shutdown",
            f"!",
            "",
        ])
    
    return "\n".join(lines)


def generate_static_routing(node_name: str, site: SDWANSite) -> str:
    """Generate static routes for IPSec overlay reachability."""
    lines = [
        f"! ── Static Routes for IPSec Overlay ──",
    ]
    
    for tunnel in site.tunnels:
        # Route remote subnets via the IPSec tunnel
        tunnel_name = f"ipsec-tun-{tunnel.remote_id}"
        lines.extend([
            f"ip route {tunnel.remote_subnet} {tunnel.tunnel_remote} {tunnel_name}",
        ])
    
    lines.append(f"!")
    return "\n".join(lines)


def generate_bgp_overlay_config(node_name: str, site: SDWANSite) -> str:
    """Generate BGP configuration for IPSec overlay routing."""
    if site.role == "hub":
        asn = 65010
        neighbors = []
        for tunnel in site.tunnels:
            neighbors.append((tunnel.remote_id, tunnel.tunnel_remote, 65020))
    else:
        asn = 65020
        neighbors = []
        for tunnel in site.tunnels:
            neighbors.append((tunnel.remote_id, tunnel.tunnel_remote, 65010))
    
    lines = [
        f"! ── BGP Overlay Routing for {node_name} ──",
        f"router bgp {asn}",
        f" bgp router-id {site.loopback}",
    ]
    
    for neighbor_name, neighbor_addr, neighbor_asn in neighbors:
        lines.extend([
            f" neighbor {neighbor_addr} remote-as {neighbor_asn}",
            f" neighbor {neighbor_addr} description {neighbor_name}",
            f" neighbor {neighbor_addr} update-source {site.loopback}",
            f" neighbor {neighbor_addr} fall-over",
            f" neighbor {neighbor_addr} timers 10 30",
        ])
    
    lines.extend([
        f" !",
        f" address-family ipv4 unicast",
        f"  network {site.loopback}/32",
    ])
    
    if site.role == "hub":
        for tunnel in site.tunnels:
            lines.append(f"  neighbor {tunnel.tunnel_remote} activate")
    else:
        for tunnel in site.tunnels:
            lines.append(f"  neighbor {tunnel.tunnel_remote} activate")
    
    lines.extend([
        f" exit-address-family",
        f"!",
    ])
    
    return "\n".join(lines)


def generate_qos_config(node_name: str, site: SDWANSite) -> str:
    """Generate QoS policies for IPSec overlay traffic."""
    lines = [
        f"! ── QoS Policies for SD-WAN Overlay ──",
        f"!",
        f"policy-map sd-wan-qos",
        f" class voice",
        f"  priority percent 30",
        f"  set dscp ef",
        f" class video",
        f"  bandwidth percent 25",
        f"  set dscp af41",
        f" class mission-critical",
        f"  bandwidth percent 20",
        f"  set dscp af31",
        f" class scavenger",
        f"  bandwidth percent 10",
        f"  set dscp cs1",
        f" class class-default",
        f"  bandwidth percent 15",
        f"  fair-queue",
        f"!",
    ]
    
    # Apply QoS to IPSec tunnel interfaces
    for tunnel in site.tunnels:
        tunnel_name = f"ipsec-tun-{tunnel.remote_id}"
        lines.extend([
            f"interface {tunnel_name}",
            f" service-policy output sd-wan-qos",
            f"!",
        ])
    
    return "\n".join(lines)


def generate_full_ipsec_config(node_name: str, site: SDWANSite) -> str:
    """Generate complete IPSec overlay configuration for a node."""
    sections = [
        f"!",
        f"! === IPSec Overlay Configuration: {node_name} ({site.role}) ===",
        f"! Site ID: {site.site_id}",
        f"! Loopback: {site.loopback}",
        f"!",
        "",
        generate_crypto_config(node_name, site),
        "",
        generate_ipsec_interfaces(node_name, site),
        "",
        generate_static_routing(node_name, site),
        "",
        generate_bgp_overlay_config(node_name, site),
        "",
        generate_qos_config(node_name, site),
        "",
        "! ── IPSec Monitoring and DPD ──",
        f"crypto logging",
        f"crypto isakmp keepalive 10 3",
        f"!",
        f"! ── NAT Traversal (for behind-NAT branches) ──",
        f"crypto nat-traversal enable",
        f"!",
    ]
    return "\n".join(sections)


def generate_ipsec_summary():
    """Generate summary report of IPSec overlay configuration."""
    lines = [
        "# SD-WAN IPSec Overlay Configuration Summary",
        "",
        "## Topology",
        "",
        "```",
        "                  [Hub-East]                    [Hub-West]",
        "                  (pe-hub-east-1)               (pe-hub-west-1)",
        "                     /    \\                       /    \\",
        "                    /      \\                     /      \\",
        "            IPSec / IPSec  IPSec / IPSec  IPSec / IPSec  IPSec / IPSec",
        "                 /          \\                     /          \\",
        "                /            \\                   /            \\",
        "        [Branch-1]      [Branch-2]        [Branch-3]      [Branch-4]",
        "        (ce-branch-1)   (ce-branch-2)     (ce-branch-3)   (ce-branch-4)",
        "```",
        "",
        "## Tunnel Details",
        "",
        "| Tunnel | Local | Remote | SPI | Encryption | Lifetime |",
        "|--------|-------|--------|-----|------------|----------|",
    ]
    
    for node_name, site in SDWAN_SITES.items():
        for tunnel in site.tunnels:
            lines.append(
                f"| {site.name}<->{tunnel.remote_id} | {tunnel.local_addr} | "
                f"{tunnel.remote_addr} | {tunnel.spi} | {tunnel.encryption} | "
                f"{tunnel.lifetime_sec}s |"
            )
    
    lines.extend([
        "",
        "## IP Addressing",
        "",
        "| Device | Loopback | Tunnel Local | Subnet |",
        "|--------|----------|--------------|--------|",
    ])
    
    for node_name, site in SDWAN_SITES.items():
        for tunnel in site.tunnels:
            lines.append(
                f"| {node_name} | {site.loopback} | {tunnel.tunnel_local} | "
                f"{tunnel.local_subnet} |"
            )
    
    return "\n".join(lines)


def main():
    output_dir = Path(__file__).parent / "configs"
    output_dir.mkdir(exist_ok=True)
    
    for node_name, site in SDWAN_SITES.items():
        config = generate_full_ipsec_config(node_name, site)
        config_path = output_dir / f"{node_name}_ipsec.conf"
        config_path.write_text(config, encoding='utf-8')
        print(f"[+] Generated IPSec config: {config_path}")
    
    # Generate summary
    summary = generate_ipsec_summary()
    summary_path = output_dir / "IPSEC_SUMMARY.md"
    summary_path.write_text(summary, encoding='utf-8')
    print(f"[+] Summary written: {summary_path}")
    
    print(f"\n[OK] IPSec overlay configuration complete for {len(SDWAN_SITES)} nodes.")


if __name__ == "__main__":
    main()
