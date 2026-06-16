"""Pure-Python NetFlow v5/v9 + IPFIX collector for the air-gapped NOC lab.

Listens on UDP 2055 (NetFlow) and 4739 (IPFIX), decodes the templates
and the data records, rolls them up into per-(device, interface,
VRF) flow counters, and exposes them on a Prometheus scrape endpoint.

This is an *alternative* to Telegraf's `inputs.netflow` plugin. The
metric names match what Telegraf exports, so the Copilot is unaffected
by which collector is in use.

Air-gap: no cloud SDKs, no `requests`. Pure-stdlib + `prometheus_client`.
"""

from __future__ import annotations

import argparse
import logging
import socket
import struct
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from prometheus_client import Counter, Gauge, start_http_server

LOG = logging.getLogger("noc_copilot.netflow")

_FLOWS: Dict[Tuple[str, str, str, str], "FlowRecord"] = {}
_TEMPLATES: Dict[Tuple[str, int, int], List[Tuple[int, int]]] = {}

NETFLOW_V5_HEADER_FMT = ">HHIIIIBBH"
NETFLOW_V5_RECORD_FMT = ">IIIHHIIIIHHBBBHHHHBB"
NETFLOW_V5_RECORD_SIZE = struct.calcsize(NETFLOW_V5_RECORD_FMT)

V9_TEMPLATE_FLOWSET_ID = 0
V9_OPTION_TEMPLATE_FLOWSET_ID = 1

IPFIX_TEMPLATE_FLOWSET_ID = 2
IPFIX_OPTION_TEMPLATE_FLOWSET_ID = 3

KNOWN_FIELD_TYPES = {
    1: ("IN_BYTES", ">I"),
    2: ("IN_PKTS", ">I"),
    4: ("PROTOCOL", "B"),
    5: ("SRC_TOS", "B"),
    6: ("TCP_FLAGS", "B"),
    7: ("L4_SRC_PORT", ">H"),
    8: ("L4_DST_PORT", ">H"),
    10: ("INPUT_SNMP", ">H"),
    11: ("L4_SRC_PORT", ">H"),
    12: ("L4_DST_PORT", ">H"),
    14: ("OUTPUT_SNMP", ">H"),
    15: ("IPV4_SRC_ADDR", "4s"),
    16: ("IPV4_DST_ADDR", "4s"),
    21: ("LAST_SWITCHED", ">I"),
    22: ("FIRST_SWITCHED", ">I"),
    23: ("OUT_BYTES", ">I"),
    24: ("OUT_PKTS", ">I"),
    27: ("IPV4_NEXT_HOP", "4s"),
    28: ("BGP_IPV4_NEXT_HOP", "4s"),
    29: ("MPLS_LABEL_1", ">I"),
    34: ("PACKET_LENGTH", ">H"),
    35: ("IPV4_IDENT", ">H"),
    41: ("INTERFACE_ENGINE_ID", ">I"),
    42: ("INTERFACE_ENGINE_TYPE", "B"),
    43: ("INGRESS_INTERFACE", ">I"),
    47: ("MPLS_TOP_LABEL_TYPE", "B"),
    56: ("INGRESS_VRFID", ">I"),
    58: ("EGRESS_VRFID", ">I"),
    59: ("VRF_NAME", "s"),
    70: ("MPLS_LABEL_STACK", "s"),
    130: ("FLOW_DIRECTION", "B"),
    150: ("FLOW_START_MILLISECONDS", ">Q"),
    151: ("FLOW_END_MILLISECONDS", ">Q"),
    152: ("FLOW_START_MICROSECONDS", ">Q"),
    153: ("FLOW_END_MICROSECONDS", ">Q"),
}


@dataclass
class FlowRecord:
    octets: int = 0
    packets: int = 0
    last_seen: float = 0.0


_GAUGES: Dict[str, Gauge] = {}
_COUNTERS: Dict[str, Counter] = {}


def _gauge(name: str, documentation: str) -> Gauge:
    g = _GAUGES.get(name)
    if g is None:
        g = Gauge(name, documentation, ["device", "src_if", "dst_if", "vrf", "protocol", "peer_addr"])
        _GAUGES[name] = g
    return g


def _counter(name: str, documentation: str) -> Counter:
    c = _COUNTERS.get(name)
    if c is None:
        c = Counter(name, documentation, ["device", "src_if", "dst_if", "vrf", "protocol", "peer_addr"])
        _COUNTERS[name] = c
    return c


def _ip4(b: bytes) -> str:
    if len(b) != 4:
        return ""
    return ".".join(str(x) for x in b)


def _ip4_from_int(v: int) -> str:
    try:
        import socket
        return socket.inet_ntoa(struct.pack("!I", int(v) & 0xFFFFFFFF))
    except (OSError, struct.error, ValueError):
        return ""


def _decode_field(fmt: str, buf: bytes, offset: int) -> Tuple[object, int]:
    if fmt == "4s":
        return _ip4(buf[offset:offset + 4]), offset + 4
    if fmt == "B":
        return buf[offset], offset + 1
    if fmt == ">H":
        return struct.unpack(">H", buf[offset:offset + 2])[0], offset + 2
    if fmt == ">I":
        return struct.unpack(">I", buf[offset:offset + 4])[0], offset + 4
    if fmt == ">Q":
        return struct.unpack(">Q", buf[offset:offset + 8])[0], offset + 8
    if fmt == "s":
        length = buf[offset]
        return buf[offset + 1:offset + 1 + length].decode("utf-8", "replace"), offset + 1 + length
    return None, offset


def _ingest_v5(payload: bytes, device: str) -> None:
    if len(payload) < struct.calcsize(NETFLOW_V5_HEADER_FMT):
        return
    (version, count, sys_uptime, unix_secs, unix_nsecs, flow_seq, engine_type, engine_id,
     sample_interval) = struct.unpack(NETFLOW_V5_HEADER_FMT, payload[:struct.calcsize(NETFLOW_V5_HEADER_FMT)])
    body = payload[struct.calcsize(NETFLOW_V5_HEADER_FMT):]
    for i in range(count):
        rec = body[i * NETFLOW_V5_RECORD_SIZE:(i + 1) * NETFLOW_V5_RECORD_SIZE]
        if len(rec) < NETFLOW_V5_RECORD_SIZE:
            break
        (src_addr, dst_addr, nxt_hop, i_if, o_if, packets, octets, first, last,
         src_port, dst_port, tcp_flags, ip_proto, tos, pad1, src_as, dst_as,
         pad2, src_mask, dst_mask) = struct.unpack(NETFLOW_V5_RECORD_FMT, rec)
        key = (device, str(i_if), str(o_if), "default", str(ip_proto), _ip4_from_int(nxt_hop))
        rec_obj = _FLOWS.setdefault(key, FlowRecord())
        rec_obj.octets += octets
        rec_obj.packets += packets
        rec_obj.last_seen = time.time()
        g = _gauge("netflow_peer_octets_total", "NetFlow octets by (device, interface).")
        g.labels(device=device, src_if=str(i_if), dst_if=str(o_if), vrf="default", protocol=str(ip_proto), peer_addr=_ip4_from_int(nxt_hop)).set(rec_obj.octets)
        c = _counter("netflow_peer_packets_total", "NetFlow packets by (device, interface).")
        c.labels(device=device, src_if=str(i_if), dst_if=str(o_if), vrf="default", protocol=str(ip_proto), peer_addr=_ip4_from_int(nxt_hop)).inc(packets)


def _ingest_v9_or_ipfix(payload: bytes, device: str, is_ipfix: bool) -> None:
    if len(payload) < 4:
        return
    if is_ipfix:
        version, length = struct.unpack(">HH", payload[:4])
        body = payload[4:length]
        msg_offset = 0
    else:
        version, count, sys_uptime, unix_secs, seq, source_id = struct.unpack(">HHIIII", payload[:20])
        body = payload[20:]
        msg_offset = 0

    while msg_offset + 4 <= len(body):
        if is_ipfix:
            flowset_id = struct.unpack(">H", body[msg_offset:msg_offset + 2])[0]
            set_length = struct.unpack(">H", body[msg_offset + 2:msg_offset + 4])[0]
        else:
            flowset_id, set_length = struct.unpack(">HH", body[msg_offset:msg_offset + 4])
        set_data = body[msg_offset + 4:msg_offset + set_length]
        msg_offset += set_length

        if flowset_id in (V9_TEMPLATE_FLOWSET_ID, IPFIX_TEMPLATE_FLOWSET_ID) or flowset_id in (V9_OPTION_TEMPLATE_FLOWSET_ID, IPFIX_OPTION_TEMPLATE_FLOWSET_ID):
            offset = 0
            while offset + 4 <= len(set_data):
                template_id = struct.unpack(">H", set_data[offset:offset + 2])[0]
                field_count = struct.unpack(">H", set_data[offset + 2:offset + 4])[0]
                offset += 4
                fields: List[Tuple[int, int]] = []
                for _ in range(field_count):
                    if offset + 4 > len(set_data):
                        break
                    if is_ipfix and (struct.unpack(">H", set_data[offset:offset + 2])[0] in (0, 1)):
                        ftype, flen, _, _ = struct.unpack(">HHBB", set_data[offset:offset + 4])
                        offset += 4
                        fields.append((ftype, flen))
                    else:
                        ftype, flen = struct.unpack(">HH", set_data[offset:offset + 4])
                        offset += 4
                        fields.append((ftype, flen))
                _TEMPLATES[(device, source_id if not is_ipfix else 0, template_id)] = fields
            continue

        template = _TEMPLATES.get((device, source_id if not is_ipfix else 0, flowset_id))
        if template is None:
            continue
        offset = 0
        while offset + 4 <= len(set_data):
            i_if = 0
            o_if = 0
            octets = 0
            packets = 0
            proto = 0
            vrf = "default"
            nxt = ""
            cursor = offset
            for ftype, flen in template:
                if cursor + flen > len(set_data):
                    break
                if ftype == 1:
                    octets = struct.unpack(">I", set_data[cursor:cursor + 4])[0]
                elif ftype == 2:
                    packets = struct.unpack(">I", set_data[cursor:cursor + 4])[0]
                elif ftype == 4:
                    proto = set_data[cursor]
                elif ftype == 10:
                    i_if = struct.unpack(">H", set_data[cursor:cursor + 2])[0]
                elif ftype == 14:
                    o_if = struct.unpack(">H", set_data[cursor:cursor + 2])[0]
                elif ftype == 15:
                    nxt = _ip4(set_data[cursor:cursor + 4])
                elif ftype == 27:
                    nxt = _ip4(set_data[cursor:cursor + 4])
                elif ftype == 56:
                    vrf = str(struct.unpack(">I", set_data[cursor:cursor + 4])[0])
                elif ftype == 58:
                    vrf = str(struct.unpack(">I", set_data[cursor:cursor + 4])[0])
                elif ftype == 59:
                    raw_v = set_data[cursor:cursor + flen]
                    vrf = raw_v.lstrip(b"\x00").decode("utf-8", "replace") or "default"
                cursor += flen
            key = (device, str(i_if), str(o_if), vrf, str(proto), nxt)
            fr = _FLOWS.setdefault(key, FlowRecord())
            fr.octets += octets
            fr.packets += packets
            fr.last_seen = time.time()
            _gauge("netflow_peer_octets_total", "NetFlow octets by (device, interface).").labels(
                device=device, src_if=str(i_if), dst_if=str(o_if), vrf=vrf, protocol=str(proto), peer_addr=nxt
            ).set(fr.octets)
            _counter("netflow_peer_packets_total", "NetFlow packets by (device, interface).").labels(
                device=device, src_if=str(i_if), dst_if=str(o_if), vrf=vrf, protocol=str(proto), peer_addr=nxt
            ).inc(packets)
            offset = cursor


def _listen_loop(host: str, port: int, is_ipfix: bool) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind((host, port))
    LOG.info("Listening on %s:%d (%s)", host, port, "IPFIX" if is_ipfix else "NetFlow")
    while True:
        data, addr = sock.recvfrom(65535)
        device = addr[0]
        if len(data) < 2:
            continue
        version = struct.unpack(">H", data[:2])[0]
        if version == 5 and not is_ipfix:
            _ingest_v5(data, device)
        elif version == 9 and not is_ipfix:
            _ingest_v9_or_ipfix(data, device, is_ipfix=False)
        elif version == 10 and is_ipfix:
            _ingest_v9_or_ipfix(data, device, is_ipfix=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Air-gapped NetFlow/IPFIX collector for the NOC Copilot")
    p.add_argument("--port", type=int, default=9127, help="Prometheus scrape port")
    p.add_argument("--netflow-port", type=int, default=2055)
    p.add_argument("--ipfix-port", type=int, default=4739)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    start_http_server(args.port)
    LOG.info("Prometheus scrape endpoint on :%d/metrics", args.port)

    import threading
    t1 = threading.Thread(target=_listen_loop, args=(args.host, args.netflow_port, False), daemon=True)
    t2 = threading.Thread(target=_listen_loop, args=(args.host, args.ipfix_port, True), daemon=True)
    t1.start()
    t2.start()
    while True:
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
