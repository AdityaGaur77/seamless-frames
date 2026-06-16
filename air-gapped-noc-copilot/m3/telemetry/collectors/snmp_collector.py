"""Pure-Python SNMP collector for the air-gapped NOC lab.

Polls the IF-MIB, BGP4-MIB, MPLS-LDP-MIB, and a small SD-WAN tunnel
MIB from each device listed in `m3/telemetry/config/snmp_targets.yaml`
and exposes the results on a Prometheus scrape endpoint.

This is an *alternative* to Telegraf's `inputs.snmp` plugin. Sites that
prefer a single small binary and a Python deployment pipeline use this;
sites that already run Telegraf use the `telegraf.conf` in
`m3/telemetry/config/`. Both produce the **same** metric names so the
downstream Copilot is unaffected.

Air-gap: uses the system `snmpget` / `snmpbulkwalk` binaries via
`subprocess`, no PySNMP dependency on outbound wheels.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from prometheus_client import Gauge, start_http_server

from .mapper import IFACE_OIDS, BGP_OIDS, LDP_OIDS, SDWAN_TUNNEL_OIDS, TOPOLOGY_FILE_OIDS, OidMap

LOG = logging.getLogger("noc_copilot.snmp")

_GAUGES: Dict[str, Gauge] = {}


def _gauge(name: str, documentation: str) -> Gauge:
    g = _GAUGES.get(name)
    if g is None:
        g = Gauge(name, documentation, ["device", "site", "role", "interface_or_peer", "vrf"])
        _GAUGES[name] = g
    return g


@dataclass
class Device:
    name: str
    address: str
    version: int = 2
    community: str = "noc-ro"
    site: str = "unknown"
    role: str = "unknown"
    vrf: str = "default"


def _load_targets(path: Path) -> List[Device]:
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    devs: List[Device] = []
    for a in data.get("agents", []):
        host = a["address"].split(":")[0]
        devs.append(Device(
            name=a["name"],
            address=host,
            version=int(a.get("version", 2)),
            community=a.get("community", "noc-ro"),
            site=a.get("site", "unknown"),
            role=a.get("role", "unknown"),
            vrf=a.get("vrf", "default"),
        ))
    return devs


def _snmpget(host: str, oid: str, version: int, community: str, timeout: float = 3.0) -> Optional[str]:
    args = ["snmpget", "-v", str(version), "-c", community, "-t", str(int(timeout)), "-r", "1", host, oid]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout + 1)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        LOG.debug("snmpget failed for %s %s: %s", host, oid, e)
        return None
    if out.returncode != 0:
        return None
    m = re.search(r"=\s*(.*)$", out.stdout.strip())
    return m.group(1).strip() if m else None


def _snmpwalk(host: str, base_oid: str, version: int, community: str, timeout: float = 5.0) -> Dict[str, str]:
    args = ["snmpbulkwalk", "-v", str(version), "-c", community, "-t", str(int(timeout)), "-r", "1", "-Cr25", host, base_oid]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout + 1)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        LOG.warning("snmpbulkwalk failed for %s %s: %s", host, base_oid, e)
        return {}
    if out.returncode != 0:
        return {}
    parsed: Dict[str, str] = {}
    for line in out.stdout.splitlines():
        m = re.match(r"\.(?P<oid>[\d\.]+)\s*=\s*(?P<type>\w+:\s*)?(?P<value>.*)$", line.strip())
        if m:
            parsed[m.group("oid")] = m.group("value").strip()
    return parsed


def _coerce(value: str, target: str) -> Optional[float]:
    value = value.strip().strip('"')
    if not value:
        return None
    if target == "counter":
        m = re.search(r"Counter(?:\d+):\s*(\d+)", value) or re.search(r"Counter64:\s*(\d+)", value) or re.search(r"(\d+)", value)
        if m is None:
            return None
        return float(m.group(m.lastindex) if m.lastindex else m.group(0))
    if target == "gauge":
        m = re.search(r"(?:Gauge\d+|INTEGER|INTEGER32|TIMETICKS|COUNTER\d*):\s*(\d+)", value, re.IGNORECASE) or re.search(r"(\d+)\s*$", value)
        if m is None:
            return None
        return float(m.group(m.lastindex) if m.lastindex else m.group(0))
    return None


def _label_value(value: str) -> str:
    return value.strip().strip('"')


def _emit(dev: Device, om: OidMap, raw: str, if_index: Optional[str] = None, if_name: Optional[str] = None) -> None:
    common_labels = {
        "device": dev.name,
        "site": dev.site,
        "role": dev.role,
        "interface_or_peer": if_name or if_index or "",
        "vrf": dev.vrf,
    }
    if om.type == "label":
        g = _gauge(f"{om.metric}_value", om.description)
        g.labels(**common_labels).set(0.0)
        from prometheus_client import Info
        if f"info_{om.metric}" not in _GAUGES:
            _GAUGES[f"info_{om.metric}"] = Info(om.metric, om.description, ["device", "site", "role", "interface_or_peer", "vrf"])
        _GAUGES[f"info_{om.metric}"].labels(**common_labels).info({om.metric: _label_value(raw)})
        return
    val = _coerce(raw, om.type)
    if val is None:
        return
    g = _gauge(om.metric, om.description)
    g.labels(**common_labels).set(val)


def _poll_device(dev: Device) -> None:
    raw = _snmpget(dev.address, ".1.3.6.1.2.1.1.5.0", dev.version, dev.community)
    if raw:
        _emit(dev, TOPOLOGY_FILE_OIDS[0], raw)

    if_walk = _snmpwalk(dev.address, ".1.3.6.1.2.1.31.1.1.1.1", dev.version, dev.community)
    if_index_to_name: Dict[str, str] = {}
    for oid, val in if_walk.items():
        m = re.match(r"\.1\.3\.6\.1\.2\.1\.31\.1\.1\.1\.1\.(?P<i>\d+)", oid)
        if m:
            if_index_to_name[m.group("i")] = _label_value(val)

    for om in IFACE_OIDS:
        if om.type == "label":
            walk = _snmpwalk(dev.address, om.oid, dev.version, dev.community)
            for sub_oid, val in walk.items():
                m = re.match(re.escape(om.oid) + r"\.(?P<i>\d+)", sub_oid)
                if m:
                    if_name = if_index_to_name.get(m.group("i"), "")
                    _emit(dev, om, val, if_index=m.group("i"), if_name=if_name)
        else:
            walk = _snmpwalk(dev.address, om.oid, dev.version, dev.community)
            for sub_oid, val in walk.items():
                m = re.match(re.escape(om.oid) + r"\.(?P<i>\d+)", sub_oid)
                if m:
                    if_name = if_index_to_name.get(m.group("i"), "")
                    _emit(dev, om, val, if_index=m.group("i"), if_name=if_name)

    for om in BGP_OIDS:
        walk = _snmpwalk(dev.address, om.oid, dev.version, dev.community)
        for sub_oid, val in walk.items():
            m = re.match(re.escape(om.oid) + r"\.(?P<i>[\d\.]+)", sub_oid)
            if m:
                _emit(dev, om, val, if_index=m.group("i"))

    for om in LDP_OIDS:
        walk = _snmpwalk(dev.address, om.oid, dev.version, dev.community)
        for sub_oid, val in walk.items():
            m = re.match(re.escape(om.oid) + r"\.(?P<i>[\d\.]+)", sub_oid)
            if m:
                _emit(dev, om, val, if_index=m.group("i"))


def main() -> int:
    p = argparse.ArgumentParser(description="Air-gapped SNMP collector for the NOC Copilot")
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--interval", type=int, default=10)
    p.add_argument("--port", type=int, default=9126)
    p.add_argument("--once", action="store_true", help="Poll each device once and exit (for tests)")
    args = p.parse_args()

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    devs = _load_targets(args.targets)
    LOG.info("Loaded %d devices from %s", len(devs), args.targets)

    start_http_server(args.port)
    LOG.info("Prometheus scrape endpoint on :%d/metrics", args.port)

    if args.once:
        for d in devs:
            _poll_device(d)
        return 0

    while True:
        t0 = time.time()
        for d in devs:
            try:
                _poll_device(d)
            except Exception as e:  # noqa: BLE001
                LOG.error("poll %s failed: %s", d.name, e)
        elapsed = time.time() - t0
        sleep_for = max(0.5, args.interval - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    sys.exit(main())
