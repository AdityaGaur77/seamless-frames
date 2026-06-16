#!/usr/bin/env python3
"""
Air-Gapped NOC Copilot - BGP/OSPF/MPLS Routing Configuration Generator
Generates FRR configs for all nodes in the MPLS SD-WAN topology.
"""

import os
import sys
from pathlib import Path

# ── Topology Definition ──
TOPOLOGY = {
    "p-core-1": {
        "role": "P",
        "router_id": "10.0.0.1",
        "loopback": "10.255.0.1",
        "interfaces": {
            "eth1": {"ip": "10.1.0.1/30", "peer": "p-core-2", "type": "core"},
            "eth3": {"ip": "10.1.0.5/30", "peer": "pe-hub-east-1", "type": "core"},
            "eth5": {"ip": "10.1.0.9/30", "peer": "pe-dc-1", "type": "core"},
            "eth10": {"ip": "172.20.0.10/24", "peer": "mgmt", "type": "mgmt"},
        },
    },
    "p-core-2": {
        "role": "P",
        "router_id": "10.0.0.2",
        "loopback": "10.255.0.2",
        "interfaces": {
            "eth2": {"ip": "10.1.0.2/30", "peer": "p-core-1", "type": "core"},
            "eth4": {"ip": "10.1.0.13/30", "peer": "pe-hub-west-1", "type": "core"},
            "eth6": {"ip": "10.1.0.17/30", "peer": "pe-dc-2", "type": "core"},
            "eth10": {"ip": "172.20.0.11/24", "peer": "mgmt", "type": "mgmt"},
        },
    },
    "pe-hub-east-1": {
        "role": "PE",
        "router_id": "10.0.0.10",
        "loopback": "10.255.0.10",
        "interfaces": {
            "eth1": {"ip": "10.1.0.6/30", "peer": "p-core-1", "type": "core"},
            "eth3": {"ip": "10.1.0.21/30", "peer": "pe-dc-1", "type": "core"},
            "eth5": {"ip": "10.10.1.1/30", "peer": "ce-branch-1", "type": "branch"},
            "eth6": {"ip": "10.10.2.1/30", "peer": "ce-branch-2", "type": "branch"},
            "eth10": {"ip": "172.20.0.20/24", "peer": "mgmt", "type": "mgmt"},
        },
        "vpn_targets": {"export": "65001:100", "import": "65001:100"},
    },
    "pe-hub-west-1": {
        "role": "PE",
        "router_id": "10.0.0.11",
        "loopback": "10.255.0.11",
        "interfaces": {
            "eth2": {"ip": "10.1.0.14/30", "peer": "p-core-2", "type": "core"},
            "eth4": {"ip": "10.1.0.25/30", "peer": "pe-dc-2", "type": "core"},
            "eth7": {"ip": "10.10.3.1/30", "peer": "ce-branch-3", "type": "branch"},
            "eth8": {"ip": "10.10.4.1/30", "peer": "ce-branch-4", "type": "branch"},
            "eth10": {"ip": "172.20.0.21/24", "peer": "mgmt", "type": "mgmt"},
        },
        "vpn_targets": {"export": "65001:200", "import": "65001:200"},
    },
    "pe-dc-1": {
        "role": "PE",
        "router_id": "10.0.0.20",
        "loopback": "10.255.0.20",
        "interfaces": {
            "eth3": {"ip": "10.1.0.10/30", "peer": "p-core-1", "type": "core"},
            "eth5": {"ip": "10.1.0.22/30", "peer": "pe-hub-east-1", "type": "core"},
            "eth10": {"ip": "172.20.0.30/24", "peer": "mgmt", "type": "mgmt"},
        },
        "vpn_targets": {"export": "65001:300", "import": "65001:300"},
    },
    "pe-dc-2": {
        "role": "PE",
        "router_id": "10.0.0.21",
        "loopback": "10.255.0.21",
        "interfaces": {
            "eth4": {"ip": "10.1.0.18/30", "peer": "p-core-2", "type": "core"},
            "eth6": {"ip": "10.1.0.26/30", "peer": "pe-hub-west-1", "type": "core"},
            "eth10": {"ip": "172.20.0.31/24", "peer": "mgmt", "type": "mgmt"},
        },
        "vpn_targets": {"export": "65001:400", "import": "65001:400"},
    },
    "ce-branch-1": {
        "role": "CE",
        "router_id": "10.0.0.40",
        "loopback": "10.255.0.40",
        "interfaces": {
            "eth5": {"ip": "10.10.1.2/30", "peer": "pe-hub-east-1", "type": "wan"},
            "eth6": {"ip": "192.168.10.1/24", "peer": "ce-branch-3", "type": "branch"},
            "eth10": {"ip": "172.20.0.40/24", "peer": "mgmt", "type": "mgmt"},
        },
    },
    "ce-branch-2": {
        "role": "CE",
        "router_id": "10.0.0.41",
        "loopback": "10.255.0.41",
        "interfaces": {
            "eth6": {"ip": "10.10.2.2/30", "peer": "pe-hub-east-1", "type": "wan"},
            "eth7": {"ip": "192.168.20.1/24", "peer": "ce-branch-4", "type": "branch"},
            "eth10": {"ip": "172.20.0.41/24", "peer": "mgmt", "type": "mgmt"},
        },
    },
    "ce-branch-3": {
        "role": "CE",
        "router_id": "10.0.0.42",
        "loopback": "10.255.0.42",
        "interfaces": {
            "eth7": {"ip": "10.10.3.2/30", "peer": "pe-hub-west-1", "type": "wan"},
            "eth5": {"ip": "192.168.10.2/24", "peer": "ce-branch-1", "type": "branch"},
            "eth10": {"ip": "172.20.0.42/24", "peer": "mgmt", "type": "mgmt"},
        },
    },
    "ce-branch-4": {
        "role": "CE",
        "router_id": "10.0.0.43",
        "loopback": "10.255.0.43",
        "interfaces": {
            "eth8": {"ip": "10.10.4.2/30", "peer": "pe-hub-west-1", "type": "wan"},
            "eth6": {"ip": "192.168.20.2/24", "peer": "ce-branch-2", "type": "branch"},
            "eth10": {"ip": "172.20.0.43/24", "peer": "mgmt", "type": "mgmt"},
        },
    },
}

# ── OSPF Area Assignments ──
CORE_OSPF_AREA = "0.0.0.0"
VPN_OSPF_AREA = "0.0.0.1"


def generate_interface_config(node_name: str, node: dict) -> str:
    """Generate interface configuration block."""
    lines = []
    for ifname, ifdata in node["interfaces"].items():
        if ifdata["type"] == "mgmt":
            continue
        lines.append(f"interface {ifname}")
        lines.append(f" description {ifdata['peer']}:{ifdata['type']}")
        lines.append(f" ip address {ifdata['ip']}")
        if node["role"] in ("P", "PE"):
            lines.append(f" ip ospf network point-to-point")
        lines.append(f" no shutdown")
        lines.append(f"!")
    return "\n".join(lines)


def generate_ospf_config(node_name: str, node: dict) -> str:
    """Generate OSPF routing configuration."""
    lines = []
    lines.append(f"router ospf")
    lines.append(f" ospf router-id {node['router_id']}")
    
    # Advertise core interfaces in area 0
    for ifname, ifdata in node["interfaces"].items():
        if ifdata["type"] in ("core",) and ifdata["type"] != "mgmt":
            network = ifdata["ip"].split("/")[0]
            # Calculate wildcard mask
            prefix = int(ifdata["ip"].split("/")[1])
            wildcard = ".".join(
                str(~int(o) & 0xFF)
                for o in str(
                    sum(
                        int(b) << (24 - 8 * i)
                        for i, b in enumerate(
                            network.split(".")
                        )
                    )
                ).split(".")
            ) if prefix == 24 else f"0.0.0.{((1 << (32 - prefix)) - 1) & 255}"
            lines.append(f" network {network}/{prefix} area {CORE_OSPF_AREA}")
    
    # Advertise branch/WAN interfaces in OSPF area 1 for VPN
    if node["role"] == "PE":
        for ifname, ifdata in node["interfaces"].items():
            if ifdata["type"] in ("branch", "wan"):
                network = ifdata["ip"].split("/")[0]
                prefix = int(ifdata["ip"].split("/")[1])
                lines.append(f" network {network}/{prefix} area {VPN_OSPF_AREA}")
    
    # Redistribute connected for loopback reachability
    lines.append(f" redistribute connected")
    lines.append(f"!")
    return "\n".join(lines)


def generate_mpls_config(node_name: str, node: dict) -> str:
    """Generate MPLS LDP configuration for P and PE routers."""
    if node["role"] not in ("P", "PE"):
        return ""
    
    lines = []
    lines.append(f"mpls ldp")
    lines.append(f" router-id {node['router_id']}")
    lines.append(f" address-family ipv4")
    for ifname, ifdata in node["interfaces"].items():
        if ifdata["type"] == "core":
            lines.append(f"  interface {ifname}")
    lines.append(f" exit-address-family")
    lines.append(f"!")
    
    # Enable MPLS on core interfaces
    for ifname, ifdata in node["interfaces"].items():
        if ifdata["type"] == "core":
            lines.append(f"interface {ifname}")
            lines.append(f" mpls ip")
            lines.append(f"!")
    
    return "\n".join(lines)


def generate_mplsvpnh_config(node_name: str, node: dict) -> str:
    """Generate MPLS L2VPN/VPLS or EVPN configuration for PE routers."""
    if node["role"] != "PE":
        return ""
    
    vt = node.get("vpn_targets", {"export": "65001:100", "import": "65001:100"})
    lines = []
    
    # BGP VPNv4 address family
    lines.append(f"router bgp 65000")
    lines.append(f" bgp router-id {node['router_id']}")
    lines.append(f" neighbor 10.255.0.1 remote-as 65000")
    lines.append(f" neighbor 10.255.0.2 remote-as 65000")
    lines.append(f" !")
    lines.append(f" address-family vpnv4 unicast")
    lines.append(f"  neighbor 10.255.0.1 activate")
    lines.append(f"  neighbor 10.255.0.2 activate")
    lines.append(f"  route-target import {vt['import']}")
    lines.append(f"  route-target export {vt['export']}")
    lines.append(f" exit-address-family")
    lines.append(f"!")
    
    return "\n".join(lines)


def generate_static_routes(node_name: str, node: dict) -> str:
    """Generate static routes for management reachability."""
    lines = []
    for ifname, ifdata in node["interfaces"].items():
        if ifdata["type"] == "mgmt":
            lines.append(f"ip route 172.20.0.0/24 {ifname}")
    return "\n".join(lines)


def generate_frr_config(node_name: str, node: dict) -> str:
    """Generate complete FRR configuration for a node."""
    sections = [
        f"!",
        f"! === {node_name} ({node['role']}) ===",
        f"! Router ID: {node['router_id']}",
        f"!",
        "",
        "! ── Loopback Interface ──",
        f"interface lo",
        f" ip address {node['loopback']}/32",
        f"!",
        "",
        "! ── Physical Interfaces ──",
        generate_interface_config(node_name, node),
        "",
        "! ── OSPF Routing ──",
        generate_ospf_config(node_name, node),
        "",
        "! ── MPLS Configuration ──",
        generate_mpls_config(node_name, node),
        "",
        "! ── MPLS VPN (PE only) ──",
        generate_mplsvpnh_config(node_name, node),
        "",
        "! ── Static Routes ──",
        generate_static_routes(node_name, node),
        "",
        "! ── BFD for fast failure detection ──",
        f"bfd",
        f"!",
    ]
    return "\n".join(sections)


def main():
    output_dir = Path(__file__).parent / "configs"
    output_dir.mkdir(exist_ok=True)
    
    for node_name, node in TOPOLOGY.items():
        config = generate_frr_config(node_name, node)
        config_path = output_dir / f"{node_name}.conf"
        config_path.write_text(config, encoding='utf-8')
        print(f"[+] Generated config: {config_path}")
    
    # Generate a summary manifest
    manifest_path = output_dir / "MANIFEST.md"
    with open(manifest_path, "w", encoding='utf-8') as f:
        f.write("# Generated FRR Configurations\n\n")
        f.write("| Node | Role | Router ID | Loopback | OSPF Area |\n")
        f.write("|------|------|-----------|----------|----------|\n")
        for name, node in TOPOLOGY.items():
            f.write(
                f"| {name} | {node['role']} | {node['router_id']} | "
                f"{node['loopback']} | {CORE_OSPF_AREA} (core), "
                f"{VPN_OSPF_AREA} (VPN) |\n"
            )
    print(f"[+] Manifest written: {manifest_path}")
    print(f"\n[OK] All {len(TOPOLOGY)} node configurations generated successfully.")


if __name__ == "__main__":
    main()
