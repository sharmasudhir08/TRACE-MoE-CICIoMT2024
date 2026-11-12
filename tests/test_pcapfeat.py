"""pcapfeat (CIC-style packet-window features) and pcapcheck (alignment with CIC's CSVs), on synthetic captures.

The rules tested here were recovered from CICIoMT2024's profiling captures (TWIN_PROTOCOL.md §3a), where they
reproduce CIC's CSVs on 38 of 39 features for 100% of ~420,000 rows (Rate 99.97%, timestamp resolution).
"""
import math

import dpkt
import pytest

from trace_iomt import pcapcheck, pcapfeat

DEV_A, DEV_B, ROUTER = b"\x90\x11\x95\x16\x40\x3b", b"\x44\x61\x32\xe8\x88\x2f", b"\x8c\xae\x4c\xdd\x1e\x39"


def eth(src, dst, etype, payload):
    return bytes(dpkt.ethernet.Ethernet(src=src, dst=dst, type=etype, data=payload))


def ipv4(proto, l4=b"", ttl=64, frag_offset=0):
    ip = dpkt.ip.IP(src=b"\x0a\x00\x00\x01", dst=b"\x0a\x00\x00\x02", p=proto, ttl=ttl, data=l4)
    ip.offset = frag_offset          # bytes; > 0 marks a later fragment
    ip.len = len(bytes(ip))
    return ip


def tcp(flags, sport=40000, dport=443):
    return dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, off=5)


def udp(sport=40000, dport=53, data=b"x" * 10):
    u = dpkt.udp.UDP(sport=sport, dport=dport, data=data)
    u.ulen = len(bytes(u))
    return u


def arp():
    return dpkt.arp.ARP()


def write_pcap(path, frames):
    with open(path, "wb") as fh:
        w = dpkt.pcap.Writer(fh)
        for ts, buf in frames:
            w.writepkt(buf, ts=ts)


def tcp_frame(flags=dpkt.tcp.TH_ACK, src=DEV_A, dst=ROUTER, dport=443):
    return eth(src, dst, dpkt.ethernet.ETH_TYPE_IP, ipv4(6, tcp(flags, dport=dport)))


def test_window_features_match_cic_rules(tmp_path):
    frames = [tcp_frame(dpkt.tcp.TH_SYN), tcp_frame(), tcp_frame(dpkt.tcp.TH_ACK | dpkt.tcp.TH_PUSH),
              eth(DEV_A, ROUTER, dpkt.ethernet.ETH_TYPE_IP, ipv4(17, udp())),
              eth(DEV_A, b"\xff" * 6, dpkt.ethernet.ETH_TYPE_ARP, arp())]
    path = tmp_path / "w.pcap"
    write_pcap(path, [(100.0 + i, f) for i, f in enumerate(frames)])
    (row,) = pcapfeat.cic_rows(path)
    lens = [len(f) for f in frames]
    assert row["Number"] == 5 and row["Tot sum"] == sum(lens) and row["Min"] == min(lens)
    assert row["syn_flag_number"] == pytest.approx(0.2) and row["ack_count"] == 2 and row["psh_flag_number"] == 0.2
    assert row["Header_Length"] == pytest.approx((20 * 3 + 8) / 5)          # TCP data offset, 8 for UDP
    assert row["HTTPS"] == pytest.approx(0.6) and row["DNS"] == pytest.approx(0.2)
    assert row["ARP"] == pytest.approx(0.2) and row["IPv"] == pytest.approx(0.8) and row["LLC"] == row["IPv"]
    assert row["Time_To_Live"] == pytest.approx(64 * 4 / 5)                  # non-IP counts as TTL 0
    assert row["Rate"] == pytest.approx(5 / 4) and row["IAT"] == pytest.approx(4 / 5)


def test_protocol_type_counts_arp_as_zero_and_breaks_ties_low(tmp_path):
    arp_heavy = [tcp_frame()] * 3 + [eth(DEV_A, ROUTER, dpkt.ethernet.ETH_TYPE_ARP, arp())] * 4
    tie = [tcp_frame()] * 2 + [eth(DEV_A, ROUTER, dpkt.ethernet.ETH_TYPE_IP, ipv4(17, udp()))] * 2
    for frames, expected in [(arp_heavy, 0), (tie, 6)]:
        path = tmp_path / f"p{expected}.pcap"
        write_pcap(path, [(float(i), f) for i, f in enumerate(frames)])
        assert pcapfeat.cic_rows(path)[0]["Protocol Type"] == expected


def test_iat_counts_gap_from_previous_window(tmp_path):
    path = tmp_path / "iat.pcap"
    write_pcap(path, [(float(t), tcp_frame()) for t in list(range(10)) + [19, 20]])
    first, second = pcapfeat.cic_rows(path)
    assert first["IAT"] == pytest.approx(9 / 10)           # no previous packet: span / n
    assert second["IAT"] == pytest.approx((20 - 9) / 2)    # from the previous window's last packet


def test_ipv6_and_lldp_frames_are_dropped_and_fragment_tail_is_not_udp(tmp_path):
    ip6 = dpkt.ip6.IP6(src=b"\x00" * 16, dst=b"\x00" * 16, nxt=17, hlim=64, data=b"")
    tail = ipv4(17, b"y" * 40, frag_offset=80)              # later fragment: protocol 17, no UDP header
    frames = [tcp_frame(), eth(DEV_A, ROUTER, dpkt.ethernet.ETH_TYPE_IP6, ip6),
              eth(DEV_A, ROUTER, 0x88cc, b"\x00" * 20), eth(DEV_A, ROUTER, dpkt.ethernet.ETH_TYPE_IP, tail)]
    path = tmp_path / "drop.pcap"
    write_pcap(path, [(float(i), f) for i, f in enumerate(frames)])
    (row,) = pcapfeat.cic_rows(path)
    assert row["Number"] == 2 and row["UDP"] == 0 and row["Header_Length"] == pytest.approx(10)


def test_pieces_split_like_tcpdump_c(tmp_path):
    frame = tcp_frame()
    path = tmp_path / "pieces.pcap"
    write_pcap(path, [(float(i), frame) for i in range(25)])
    per_record = pcapfeat.RECORD_HEADER + len(frame)
    limit = pcapfeat.PCAP_HEADER + 7 * per_record           # exceeded after the 8th record
    pieces = list(pcapfeat.read_pieces(path, chunk_bytes=limit))
    assert [len(p) for p in pieces] == [8, 8, 8, 1]
    rows = pcapfeat.cic_rows(path, chunk_bytes=limit)
    assert [r["Number"] for r in rows] == [8, 8, 8, 1] and math.isnan(rows[-1]["Std"])


def test_device_rows_follow_each_device_and_count_shared_packets_twice(tmp_path):
    frames = [tcp_frame(src=DEV_A, dst=ROUTER)] * 12 + [tcp_frame(src=DEV_B, dst=ROUTER)] * 3 \
        + [tcp_frame(src=DEV_A, dst=DEV_B)] * 2
    path = tmp_path / "dev.pcap"
    write_pcap(path, [(float(i), f) for i, f in enumerate(frames)])
    rows = pcapfeat.device_rows(path, {"a": DEV_A.hex(":"), "b": DEV_B.hex(":")})
    assert [(r["device"], r["Number"]) for r in rows] == [("a", 10), ("a", 4), ("b", 5)]


def test_dominant_mac_ignores_infrastructure(tmp_path):
    path = tmp_path / "single.pcap"
    write_pcap(path, [(float(i), tcp_frame(src=DEV_A, dst=ROUTER)) for i in range(5)])
    mac, share = pcapfeat.dominant_mac(path, exclude={ROUTER.hex(":")})
    assert mac == DEV_A.hex(":") and share == 1.0


def _rows(sums, piece):
    return [{"Tot sum": s, "Max": s, "Min": s, "piece": piece, "Number": 10} for s in sums]


def test_align_places_shuffled_and_repetitive_pieces():
    p0, p1 = _rows([5, 5, 5, 5, 5, 6, 7], 0), _rows([5, 5, 5, 5, 5, 8, 9], 1)  # same first rows
    official = [{k: str(v) for k, v in r.items()} for r in p1 + p0]           # CSV wrote piece 1 first
    pairs, unplaced = pcapcheck.align(p0 + p1, official)
    assert unplaced == 0
    assert all(float(o["Tot sum"]) == m["Tot sum"] for o, m in pairs)


def test_agreement_treats_blank_std_as_nan():
    pairs = [({"Std": "", "Rate": "2"}, {"Std": float("nan"), "Rate": 2.0, "Number": 10})]
    assert pcapcheck.agreement(pairs, ["Std", "Rate"]) == {"Std": 1.0, "Rate": 1.0}


def test_device_rows_count_attacker_packets(tmp_path):
    attacker = b"\xdc\xa6\x32\xc9\xe4\xab"
    frames = [tcp_frame(src=attacker, dst=DEV_A)] * 3 + [tcp_frame(src=DEV_A, dst=ROUTER)] * 7
    path = tmp_path / "att.pcap"
    write_pcap(path, [(float(i), f) for i, f in enumerate(frames)])
    (row,) = pcapfeat.device_rows(path, {"a": DEV_A.hex(":")}, attackers={attacker.hex(":")})
    assert row["n_attacker_packets"] == 3 and row["Number"] == 10


def test_device_rows_iat_continues_across_windows_of_one_device(tmp_path):
    # device a's packets at t = 0..9 and 30, interleaved with b's; a's second window starts after t = 9
    frames = [(float(t), tcp_frame(src=DEV_A, dst=ROUTER)) for t in range(10)] + \
             [(20.0, tcp_frame(src=DEV_B, dst=ROUTER)), (30.0, tcp_frame(src=DEV_A, dst=ROUTER))]
    path = tmp_path / "iat_dev.pcap"
    write_pcap(path, frames)
    rows = [r for r in pcapfeat.device_rows(path, {"a": DEV_A.hex(":"), "b": DEV_B.hex(":")}) if r["device"] == "a"]
    assert rows[1]["IAT"] == pytest.approx(30 - 9)


def _extract_script():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("pcap_extract", Path(__file__).resolve().parents[1] / "scripts" / "14_pcap_extract.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extract_writes_labelled_parquet_per_capture(tmp_path):
    import pandas as pd
    mod = _extract_script()
    root = tmp_path / "WiFi_and_MQTT"
    attack = root / "attacks" / "PCAP" / "train" / "TCP_IP-DDoS-ICMP4_train.pcap"
    prof = root / "profiling" / "PCAP" / "Power" / "Dev_Power.pcap"
    for p in (attack, prof):
        p.parent.mkdir(parents=True)
        write_pcap(p, [(float(i), tcp_frame(src=DEV_A, dst=ROUTER)) for i in range(12)])
    out = tmp_path / "out"
    for p in (attack, prof):
        mod.extract_one(p, root, out, "device", {"a": DEV_A.hex(":")}, set())
    a = pd.read_parquet(out / "train" / "TCP_IP-DDoS-ICMP4_train.parquet")
    b = pd.read_parquet(out / "profiling" / "Dev_Power.parquet")
    assert set(a["label19"]) == {"TCP_IP-DDoS-ICMP"} and set(a["split"]) == {"train"} and len(a) == 2
    assert set(b["label19"]) == {"Benign"} and set(b["state"]) == {"Power"}
    with pytest.raises(ValueError, match="PCAP_EXTRACT_NAME"):
        mod.describe(root / "attacks" / "PCAP" / "train" / "odd.pcap", root)


def malformed_ipv4_frame():
    # Ethernet header with type IPv4, then an IP header whose length field (IHL = 1) is invalid
    return DEV_A + ROUTER + b"\x08\x00" + b"\x41" + b"\x00" * 30


def test_malformed_frames_are_skipped_and_counted(tmp_path):
    import collections
    path = tmp_path / "bad.pcap"
    # 802.3 length field + LLC/SNAP carrying a bad IP header: this frame stopped the first scan (2026-11-04)
    llc = DEV_A + ROUTER + (38).to_bytes(2, "big") + b"\xaa\xaa\x03\x00\x00\x00\x08\x00" + b"\x41" + b"\x00" * 29
    write_pcap(path, [(0.0, tcp_frame()), (1.0, malformed_ipv4_frame()), (1.5, llc), (2.0, tcp_frame())])
    stats = collections.Counter()
    (row,) = pcapfeat.cic_rows(path, stats=stats)
    assert row["Number"] == 2 and stats["malformed"] == 1
    stats = collections.Counter()
    rows = pcapfeat.device_rows(path, {"a": DEV_A.hex(":")}, stats=stats)
    assert rows[0]["Number"] == 2 and stats["malformed"] == 1


def test_scan_reads_macs_without_decoding_payload(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("pcap_scan", Path(__file__).resolve().parents[1] / "scripts" / "12_pcap_scan.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    path = tmp_path / "scan.pcap"
    llc = DEV_A + ROUTER + (38).to_bytes(2, "big") + b"\xaa\xaa\x03\x00\x00\x00\x08\x00" + b"\x41" + b"\x00" * 29
    write_pcap(path, [(0.0, llc), (1.0, tcp_frame(src=DEV_B, dst=ROUTER))])
    row = mod.mac_shares(path, {"a": DEV_A.hex(":"), "b": DEV_B.hex(":"), "router": ROUTER.hex(":")})
    assert row["packets"] == 2 and row["a"] == 0.5 and row["b"] == 0.5 and row["router"] == 1.0


def test_scan_resumes_after_a_stop(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts" / "12_pcap_scan.py"
    spec = importlib.util.spec_from_file_location("pcap_scan2", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pcaps = tmp_path / "PCAP" / "train"
    pcaps.mkdir(parents=True)
    for name in ("A_train.pcap", "B_train.pcap"):
        write_pcap(pcaps / name, [(0.0, tcp_frame())])
    cfg = tmp_path / "devices.json"
    cfg.write_text(json.dumps({"devices": {"a": DEV_A.hex(":")}, "infrastructure": {"router": ROUTER.hex(":")}}))
    out = tmp_path / "out"
    calls = []
    real = mod.mac_shares
    monkeypatch.setattr(mod, "mac_shares", lambda f, names: calls.append(f.name) or real(f, names))
    args = ["12_pcap_scan.py", "--pcaps", str(tmp_path / "PCAP"), "--devices", str(cfg), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", args)
    mod.main()
    monkeypatch.setattr(sys, "argv", args)
    mod.main()                                           # second run: nothing left to scan
    assert calls == ["A_train.pcap", "B_train.pcap"]
    assert len((out / "pcap_mac_shares.csv").read_text().strip().splitlines()) == 3


def _mqtt_publish(topic, payload=b"42"):
    t = topic.encode()
    body = len(t).to_bytes(2, "big") + t + payload
    return bytes([0x30, len(body)]) + body


def test_mqtt_topic_parsing():
    assert pcapfeat.mqtt_topic(_mqtt_publish("ward/3/spo2")) == "ward/3/spo2"
    sub = bytes([0x82, 12]) + b"\x00\x01" + (8).to_bytes(2, "big") + b"cmd/pump" + b"\x00"
    assert pcapfeat.mqtt_topic(sub) == "cmd/pump"
    assert pcapfeat.mqtt_topic(bytes([0x30, 10, 0x00, 0x20]) + b"ab") == "?"     # topic length beyond the packet
    assert pcapfeat.mqtt_topic(bytes([0x10, 2, 0, 0])) == ""                       # CONNECT: no topic


def test_device_rows_record_peers_and_topics(tmp_path):
    t = dpkt.tcp.TCP(sport=40000, dport=1883, flags=dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK, off=5,
                     data=_mqtt_publish("ward/3/spo2"))
    mqtt = eth(DEV_A, ROUTER, dpkt.ethernet.ETH_TYPE_IP, ipv4(6, t))
    frames = [mqtt, tcp_frame(src=DEV_A, dst=ROUTER), eth(DEV_B, DEV_A, dpkt.ethernet.ETH_TYPE_ARP, arp())]
    path = tmp_path / "peers.pcap"
    write_pcap(path, [(float(i), f) for i, f in enumerate(frames)])
    (row,) = [r for r in pcapfeat.device_rows(path, {"a": DEV_A.hex(":")})]
    assert row["topics"] == "ward/3/spo2"
    assert row["peers"] == ";".join(sorted({"10.0.0.2", DEV_B.hex(":")}))


def test_all_hosts_mode_covers_every_unicast_host_except_excluded(tmp_path):
    other = b"\x20\x7b\xd2\x23\x55\x40"
    frames = [tcp_frame(src=other, dst=ROUTER)] * 3 + [tcp_frame(src=DEV_A, dst=other)] * 2 \
        + [eth(DEV_A, b"\xff" * 6, dpkt.ethernet.ETH_TYPE_ARP, arp())] \
        + [eth(DEV_A, b"\x01\x00\x5e\x00\x00\xfb", dpkt.ethernet.ETH_TYPE_IP, ipv4(17, udp(dport=5353)))]
    path = tmp_path / "hosts.pcap"
    write_pcap(path, [(float(i), f) for i, f in enumerate(frames)])
    known = {"a": DEV_A.hex(":")}
    only_known = pcapfeat.device_rows(path, known)
    assert [(r["device"], r["Number"]) for r in only_known] == [("a", 4)]
    rows = pcapfeat.device_rows(path, known, all_hosts=True, exclude={ROUTER.hex(":")})
    assert sorted((r["device"], r["Number"]) for r in rows) == [("a", 4), (f"mac:{other.hex(':')}", 5)]
    assert pcapfeat.is_group_mac("ff:ff:ff:ff:ff:ff") and pcapfeat.is_group_mac("01:00:5e:00:00:fb")
    assert not pcapfeat.is_group_mac(DEV_A.hex(":"))
