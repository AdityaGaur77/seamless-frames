#!/usr/bin/env python3
"""Continuously seeds synthetic telemetry into TimescaleDB.

Runs as a Docker sidecar next to the orchestrator.  Every 10 seconds
it inserts one row per (host, interface) so the sampler always finds
fresh data inside its active_window_seconds lookback.
"""
import os
import random
import time

import psycopg2

DB = dict(
    host=os.environ.get("DB_HOST", "172.20.0.120"),
    port=int(os.environ.get("DB_PORT", "5432")),
    dbname=os.environ.get("DB_NAME", "noc_telemetry"),
    user=os.environ.get("DB_USER", "postgres"),
    password=os.environ.get("DB_PASSWORD", "noc_copilot_db"),
)

HOSTS = ["router-core-01", "router-core-02", "switch-access-01"]
INTERFACES = {
    "router-core-01": ["GigabitEthernet0/0/1", "GigabitEthernet0/0/2"],
    "router-core-01:BGP": None,
    "router-core-02": ["GigabitEthernet0/0/1"],
    "router-core-02:BGP": None,
    "switch-access-01": ["GigabitEthernet1/0/1", "GigabitEthernet1/0/2"],
}


def insert_iface(conn, host: str, iface: str, anomalous: bool = False):
    with conn.cursor() as cur:
        if anomalous:
            util = 85 + random.random() * 15
            in_err = random.random() * 500
            out_err = random.random() * 200
        else:
            util = 20 + random.random() * 40
            in_err = random.random() * 10
            out_err = random.random() * 5
        cur.execute(
            """INSERT INTO interface_metrics
               (time, host, interface, in_octets, out_octets,
                in_errors, out_errors, in_discards, out_discards,
                in_packets, out_packets, speed, utilization)
               VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                host, iface,
                1e9 + random.random() * 1e8,
                8e8 + random.random() * 1e8,
                in_err, out_err,
                random.random() * 10, random.random() * 5,
                1e6 + random.random() * 1e5,
                8e5 + random.random() * 1e5,
                1e9, util,
            ),
        )


def insert_routing(conn, host: str, anomalous: bool = False):
    with conn.cursor() as cur:
        if anomalous:
            state = "Active"
            state_chg = random.randint(1, 3)
            retrans = random.randint(50, 200)
        else:
            state = "Established"
            state_chg = 0
            retrans = random.randint(0, 5)
        cur.execute(
            """INSERT INTO routing_metrics
               (time, host, protocol, neighbor, state,
                state_changes, retransmissions, updates_in, updates_out,
                fsm_transitions)
               VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                host, "bgp", "10.0.0.2", state,
                state_chg, retrans,
                random.randint(0, 20), random.randint(0, 15),
                random.randint(0, 5),
            ),
        )


def main():
    print("Waiting 15s for TimescaleDB to become healthy...")
    time.sleep(15)

    while True:
        try:
            conn = psycopg2.connect(**DB)
            conn.autocommit = True
            break
        except Exception as e:
            print(f"DB not ready ({e}), retrying in 5s...")
            time.sleep(5)

    print("Connected. Seeding telemetry every 10s...")
    tick = 0
    while True:
        # Every ~20 ticks (~200s), inject an anomalous spike on one host
        spike = (tick % 20 == 18)

        for host in HOSTS:
            for key, ifaces in list(INTERFACES.items()):
                if not key.startswith(host):
                    continue
                if ifaces is None:
                    # routing row
                    insert_routing(conn, host, anomalous=spike)
                else:
                    for iface in ifaces:
                        insert_iface(conn, host, iface, anomalous=spike)

        conn.commit()
        status = " ** ANOMALOUS SPIKE **" if spike else ""
        print(f"[tick {tick}] seeded{status}", flush=True)
        tick += 1
        time.sleep(10)


if __name__ == "__main__":
    main()
