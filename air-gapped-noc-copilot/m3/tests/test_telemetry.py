"""Test the pure-Python parts of the telemetry collectors."""
import json
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(r'C:\Users\adity\Downloads\real-port\test\air-gapped-noc-copilot')
sys.path.insert(0, str(ROOT))

from m3.telemetry.collectors.mapper import IFACE_OIDS, BGP_OIDS, LDP_OIDS, SDWAN_TUNNEL_OIDS, ALL_OIDS, OidMap
from m3.telemetry.collectors.snmp_collector import _load_targets, _coerce, _label_value, Device
from m3.telemetry.collectors.netflow_collector import _ingest_v5, _ingest_v9_or_ipfix, _decode_field, _FLOWS, KNOWN_FIELD_TYPES


def test_mapper_consistent_metric_types():
    for o in ALL_OIDS:
        assert o.metric, f"empty metric name on {o}"
        assert o.oid, f"empty OID on {o.metric}"
        assert o.type in ("counter", "gauge", "label")
    print(f"PASS  mapper: {len(ALL_OIDS)} OIDs all have non-empty metric, oid, valid type")


def test_load_targets_yaml():
    targets_path = ROOT / "m3" / "telemetry" / "config" / "snmp_targets.yaml"
    devs = _load_targets(targets_path)
    assert len(devs) == 10, f"expected 10 devices, got {len(devs)}"
    p1 = next(d for d in devs if d.name == "p-core-1")
    assert p1.address == "172.20.0.10"
    assert p1.community == "noc-ro"
    print(f"PASS  _load_targets: {len(devs)} devices loaded from yaml")


def test_coerce_counter_string():
    assert _coerce("Counter32: 12345", "counter") == 12345.0
    assert _coerce("Counter64: 99999999999", "counter") == 99999999999.0
    assert _coerce("Gauge32: 42", "gauge") == 42.0
    assert _coerce("INTEGER: 7", "gauge") == 7.0
    assert _coerce("\"some_string\"", "gauge") is None
    assert _coerce("", "counter") is None
    print("PASS  _coerce: counter / gauge / string / empty handling")


def test_label_value_strips_quotes():
    assert _label_value('"eth0"') == "eth0"
    assert _label_value("eth0") == "eth0"
    assert _label_value("  10.0.0.1  ") == "10.0.0.1"
    print("PASS  _label_value: strips quotes and whitespace")


def test_netflow_v5_roundtrip():
    """Encode a minimal NetFlow v5 packet and decode it."""
    import m3.telemetry.collectors.netflow_collector as nfc
    nfc._FLOWS.clear()
    header = struct.pack(">HHIIIIBBH", 5, 1, 100, 1000000, 0, 1, 0, 0, 30)
    record = struct.pack(">IIIHHIIIIHHBBBHHHHBB",
                         0x0A0A0101, 0x0A0A0202, 0x0A0A0303,
                         1, 2,
                         5, 1500,
                         0, 0,
                         80, 443,
                         0, 6, 0, 0,
                         0, 0,
                         0,
                         0, 0)
    nfc._ingest_v5(header + record, device="10.0.0.99")
    assert any("10.0.0.99" in str(k) for k in nfc._FLOWS), f"expected flow entry for 10.0.0.99, got {nfc._FLOWS}"
    key = next(k for k in nfc._FLOWS if "10.0.0.99" in str(k))
    rec_obj = nfc._FLOWS[key]
    assert rec_obj.octets == 1500, f"octets={rec_obj.octets}"
    assert rec_obj.packets == 5, f"packets={rec_obj.packets}"
    print(f"PASS  netflow v5 roundtrip: octets={rec_obj.octets}, packets={rec_obj.packets}")


def test_netflow_v9_template_then_data():
    """Encode a template, then a data record, and verify it parses."""
    import m3.telemetry.collectors.netflow_collector as nfc
    nfc._FLOWS.clear()
    template_id = 256
    record_count = 0
    sysuptime = 100
    unix_secs = 1000000
    seq = 1
    source_id = 0
    header = struct.pack(">HHIIII", 9, record_count, sysuptime, unix_secs, seq, source_id)
    template_header = struct.pack(">HH", 0, 24)
    template_body = struct.pack(">HH", template_id, 4)
    template_body += struct.pack(">HH", 1, 4)
    template_body += struct.pack(">HH", 2, 4)
    template_body += struct.pack(">HH", 4, 1)
    template_body += struct.pack(">HH", 10, 2)
    nfc._FLOWS.clear()
    nfc._ingest_v9_or_ipfix(header + template_header + template_body, device="10.0.0.100", is_ipfix=False)
    data_set = struct.pack(">HH", template_id, 4 + 4 + 4 + 1 + 2)
    data_body = struct.pack(">I", 1500)
    data_body += struct.pack(">I", 5)
    data_body += struct.pack("B", 6)
    data_body += struct.pack(">H", 1)
    record_count = 1
    header2 = struct.pack(">HHIIII", 9, record_count, sysuptime, unix_secs, seq + 1, source_id)
    nfc._ingest_v9_or_ipfix(header2 + data_set + data_body, device="10.0.0.100", is_ipfix=False)
    assert len(nfc._FLOWS) > 0, f"expected at least one flow, got {nfc._FLOWS}"
    print(f"PASS  netflow v9 template+data: {len(nfc._FLOWS)} flows")


def test_known_field_types_have_required_keys():
    for ftype, spec in KNOWN_FIELD_TYPES.items():
        assert isinstance(ftype, int)
        assert spec, f"empty spec for type {ftype}"
    assert KNOWN_FIELD_TYPES[1][0] == "IN_BYTES"
    assert KNOWN_FIELD_TYPES[15][0] == "IPV4_SRC_ADDR"
    assert KNOWN_FIELD_TYPES[27][0] == "IPV4_NEXT_HOP"
    print(f"PASS  KNOWN_FIELD_TYPES: {len(KNOWN_FIELD_TYPES)} field types registered")


def test_decode_field_ip4():
    val, off = _decode_field("4s", b"\xc0\xa8\x01\x01\xde\xad", 0)
    assert val == "192.168.1.1"
    assert off == 4
    val, off = _decode_field(">I", struct.pack(">I", 12345), 0)
    assert val == 12345
    assert off == 4
    val, off = _decode_field("B", b"\x42", 0)
    assert val == 0x42
    assert off == 1
    print("PASS  _decode_field: IP4 / uint32 / byte")


if __name__ == "__main__":
    test_mapper_consistent_metric_types()
    test_load_targets_yaml()
    test_coerce_counter_string()
    test_label_value_strips_quotes()
    test_netflow_v5_roundtrip()
    test_netflow_v9_template_then_data()
    test_known_field_types_have_required_keys()
    test_decode_field_ip4()
    print("\nAll telemetry tests passed.")
