"""Packet-window features from PCAPs, in two modes (Digital Twin, task 20; added 2026-11-04).

CICIoMT2024 rows are not network flows: each row aggregates 10 consecutive packets. The rules below were
recovered by matching the profiling PCAPs against CIC's own CSVs (TWIN_PROTOCOL.md §3a):
  - only IPv4 and ARP frames are used (IPv6, LLDP and LLC frames are dropped);
  - CIC split every capture into 10,000,000-byte pieces (like `tcpdump -C 10`) and formed windows inside
    each piece, so the last window of a piece can be shorter than 10 packets;
  - IAT is the mean gap between consecutive packets, counting the gap from the previous window's last
    packet; Rate is packets / time span inside the window; Protocol Type is the most frequent protocol number
    over all packets (non-IP = 0), ties going to the lowest; TCP/UDP shares count packets whose transport
    header parses (not trailing IP fragments); application protocols are matched on TCP ports (DNS and DHCP
    on UDP).

Modes:
  cic     reproduces CIC's rows (validation against the official CSVs, gate G1);
  device  windows of 10 consecutive packets that involve one device's MAC, in time order over the whole
          capture (no 10 MB artefact), for the per-device twin.
The six features found only in the attack CSVs (Srate, Drate, Magnitue, Radius, Covariance, Weight) are not
computed yet: they need an attack CSV to recover their definitions.
"""
import collections
import statistics
from dataclasses import dataclass

import dpkt

WINDOW = 10
CHUNK_BYTES = 10_000_000
PCAP_HEADER, RECORD_HEADER = 24, 16
KEEP_ETHERTYPES = (dpkt.ethernet.ETH_TYPE_IP, dpkt.ethernet.ETH_TYPE_ARP)
TCP_APPS = {"HTTP": {80}, "HTTPS": {443}, "Telnet": {23}, "SMTP": {25}, "SSH": {22}, "IRC": {6667, 194}}
UDP_APPS = {"DNS": {53}, "DHCP": {67, 68}}
FLAGS = [("fin", 0x01), ("syn", 0x02), ("rst", 0x04), ("psh", 0x08), ("ack", 0x10), ("ece", 0x40), ("cwr", 0x80)]
COUNTED = ("ack", "syn", "fin", "rst")
# Feature order of the profiling CSVs (39 columns); Time_To_Live is called Duration in the attack CSVs.
FEATURES = ["Header_Length", "Protocol Type", "Time_To_Live", "Rate"] + [f"{n}_flag_number" for n, _ in FLAGS] \
    + [f"{n}_count" for n in COUNTED] + ["HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP",
                                          "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC", "Tot sum", "Min", "Max",
                                          "AVG", "Std", "Tot size", "IAT", "Number", "Variance"]


@dataclass(frozen=True)
class Packet:
    ts: float
    length: int
    src_mac: str
    dst_mac: str
    is_ip: bool
    is_arp: bool
    proto: int          # IP protocol number, 0 if not IP
    ttl: int
    l4_header: int      # TCP data offset in bytes, 8 for UDP, 0 otherwise
    tcp_flags: int
    l4: str             # "tcp", "udp" or "" (transport header parsed)
    apps: frozenset     # application protocols matched by port
    src_ip: str = ""    # dotted IPv4, "" for ARP
    dst_ip: str = ""
    mqtt_topic: str = ""   # topic of an MQTT PUBLISH/SUBSCRIBE, "?" if that packet does not parse


MQTT_PORT = 1883
MQTT_PUBLISH, MQTT_SUBSCRIBE = 3, 8
MAX_TOPIC = 128


def mqtt_topic(payload):
    """Topic of the first MQTT PUBLISH or SUBSCRIBE in a TCP payload; "" for other MQTT packets, "?" when a
    PUBLISH/SUBSCRIBE cannot be parsed (e.g. the MQTT-Malformed_Data attack)."""
    if not payload:
        return ""
    kind = payload[0] >> 4
    if kind not in (MQTT_PUBLISH, MQTT_SUBSCRIBE):
        return ""
    try:
        i, length, shift = 1, 0, 0
        while True:                                   # "remaining length": 1-4 bytes, 7 bits each
            b = payload[i]
            length |= (b & 0x7F) << shift
            i, shift = i + 1, shift + 7
            if not b & 0x80 or shift > 21:
                break
        if kind == MQTT_SUBSCRIBE:
            i += 2                                    # packet identifier
        n = int.from_bytes(payload[i:i + 2], "big")
        topic = payload[i + 2:i + 2 + n]
        if len(topic) != n:
            return "?"
        return topic.decode("utf-8", errors="replace")[:MAX_TOPIC]
    except IndexError:
        return "?"


def _ipv4(raw):
    return ".".join(str(b) for b in raw)


class MalformedFrame(Exception):
    """An IPv4/ARP frame whose headers cannot be decoded (seen in the CICIoMT2024 attack captures)."""


def parse_packet(ts, buf):
    """One kept frame -> Packet, or None for frames CIC does not use. Raises MalformedFrame for an IPv4/ARP
    frame that cannot be decoded; callers skip and count it."""
    if len(buf) < 14 or int.from_bytes(buf[12:14], "big") not in KEEP_ETHERTYPES:
        return None                                  # decided from the raw header, without decoding the payload
    try:
        eth = dpkt.ethernet.Ethernet(buf)
    except (dpkt.UnpackError, dpkt.NeedData) as exc:
        raise MalformedFrame(str(exc)) from exc
    ip = eth.data
    is_ip = isinstance(ip, dpkt.ip.IP)
    if not (is_ip or isinstance(ip, dpkt.arp.ARP)):
        raise MalformedFrame(f"ethertype {eth.type:#06x} but the payload does not decode")   # dpkt keeps raw bytes
    l4 = ip.data if is_ip else None
    tcp, udp = isinstance(l4, dpkt.tcp.TCP), isinstance(l4, dpkt.udp.UDP)
    ports = {l4.sport, l4.dport} if (tcp or udp) else set()
    table = TCP_APPS if tcp else UDP_APPS if udp else {}
    apps = frozenset(name for name, p in table.items() if ports & p)
    topic = mqtt_topic(bytes(l4.data)) if tcp and MQTT_PORT in ports else ""
    return Packet(ts=float(ts), length=len(buf), src_mac=eth.src.hex(":"), dst_mac=eth.dst.hex(":"),
                  is_ip=is_ip, is_arp=isinstance(ip, dpkt.arp.ARP), proto=ip.p if is_ip else 0,
                  ttl=ip.ttl if is_ip else 0, l4_header=l4.off * 4 if tcp else (8 if udp else 0),
                  tcp_flags=l4.flags if tcp else 0, l4="tcp" if tcp else "udp" if udp else "", apps=apps,
                  src_ip=_ipv4(ip.src) if is_ip else "", dst_ip=_ipv4(ip.dst) if is_ip else "", mqtt_topic=topic)


def _parse_counted(ts, buf, stats):
    try:
        return parse_packet(ts, buf)
    except MalformedFrame:
        if stats is not None:
            stats["malformed"] += 1
        return None


def read_pieces(path, chunk_bytes=CHUNK_BYTES, stats=None):
    """Kept packets of a capture, split into pieces as CIC did: a new piece starts once the current one
    (pcap header + records, all frames counted) exceeds chunk_bytes. Malformed frames are skipped and
    counted in stats["malformed"] when a Counter is given."""
    piece, size = [], PCAP_HEADER
    with open(path, "rb") as fh:
        for ts, buf in dpkt.pcap.Reader(fh):
            if size > chunk_bytes:
                yield piece
                piece, size = [], PCAP_HEADER
            size += RECORD_HEADER + len(buf)
            p = _parse_counted(ts, buf, stats)
            if p is not None:
                piece.append(p)
    if piece:
        yield piece


def window_features(win, prev_ts):
    """CIC features of one window. prev_ts: timestamp of the packet before the window (its own first
    packet when there is none), used for IAT."""
    n = len(win)
    lengths = [p.length for p in win]
    protos = collections.Counter(p.proto for p in win)          # ARP and other non-IP count as protocol 0
    top = max(protos.values()) if protos else 0
    span = win[-1].ts - win[0].ts
    f = {"Header_Length": sum(p.l4_header for p in win) / n,
         "Protocol Type": min(k for k, v in protos.items() if v == top),
         "Time_To_Live": sum(p.ttl for p in win) / n,
         "Rate": n / span if span > 0 else 0.0,
         "IAT": (win[-1].ts - prev_ts) / n}
    for name, bit in FLAGS:
        c = sum(1 for p in win if p.tcp_flags & bit)
        f[f"{name}_flag_number"] = c / n
        if name in COUNTED:
            f[f"{name}_count"] = c
    for app in list(TCP_APPS) + list(UDP_APPS):
        f[app] = sum(app in p.apps for p in win) / n
    f["TCP"] = sum(p.l4 == "tcp" for p in win) / n     # parsed transport header: IP fragments after the
    f["UDP"] = sum(p.l4 == "udp" for p in win) / n     # first carry protocol 6/17 but no header
    f["ICMP"] = sum(p.proto == 1 for p in win) / n
    f["IGMP"] = sum(p.proto == 2 for p in win) / n
    f["ARP"] = sum(p.is_arp for p in win) / n
    f["IPv"] = f["LLC"] = sum(p.is_ip for p in win) / n
    f.update({"Tot sum": sum(lengths), "Min": min(lengths), "Max": max(lengths), "AVG": sum(lengths) / n,
              "Tot size": sum(lengths) / n, "Number": n,
              "Std": statistics.stdev(lengths) if n > 1 else float("nan"),
              "Variance": statistics.variance(lengths) if n > 1 else float("nan")})
    return f


def windows_of(packets, size=WINDOW):
    """Consecutive windows of `size` packets (the last one may be shorter) with their features and times."""
    out = []
    for i in range(0, len(packets), size):
        win = packets[i:i + size]
        prev_ts = packets[i - 1].ts if i else win[0].ts
        out.append({**window_features(win, prev_ts), "t_start": win[0].ts, "t_end": win[-1].ts})
    return out


def cic_rows(path, chunk_bytes=CHUNK_BYTES, stats=None):
    """Rows as CIC produced them, in capture order; `piece` gives the 10 MB piece of each row."""
    rows = []
    for k, piece in enumerate(read_pieces(path, chunk_bytes, stats)):
        rows += [{**w, "piece": k} for w in windows_of(piece)]
    return rows


def _peer(p, mac):
    """The other end of a packet as seen by the device with `mac`: its IP address, or its MAC for ARP."""
    if p.src_mac == mac:
        return p.dst_ip or p.dst_mac
    return p.src_ip or p.src_mac


def _device_row(device, mac, win, prev_ts, attackers):
    row = window_features(win, prev_ts)
    row.update(device=device, t_start=win[0].ts, t_end=win[-1].ts,
               n_attacker_packets=sum(1 for p in win if p.src_mac in attackers or p.dst_mac in attackers),
               peers=";".join(sorted({_peer(p, mac) for p in win})),
               topics=";".join(sorted({p.mqtt_topic for p in win if p.mqtt_topic})))
    return row


def is_group_mac(mac):
    """Broadcast/multicast address (I/G bit set): never a device."""
    return int(mac[:2], 16) & 1 == 1


def host_name(mac, known):
    """Name of an endpoint: its inventory name, else 'mac:<address>' for hosts outside the twin."""
    return known.get(mac, f"mac:{mac}")


def device_rows(path, macs, attackers=(), stats=None, all_hosts=False, exclude=()):
    """Per-device rows, streamed: windows of 10 consecutive packets with the device's MAC as source or
    destination, over the whole capture in time order (no 10 MB pieces). `macs` maps name -> MAC. A packet
    between two endpoints counts for both. With all_hosts=True every unicast MAC except `exclude` (e.g. the
    router) is an endpoint, named 'mac:<address>' when it is not in `macs`: a gateway sees every host, and
    hosts outside the twin get the fleet-wide profile. Each row records how many of its packets involve an
    attacker MAC, so attack labels can later be given by file or by attacker presence without
    re-extracting, plus its peers (IP, or MAC for ARP) and MQTT topics for the twin's novelty flags.
    Memory: one buffer of <= 10 packets per endpoint."""
    known = {mac: name for name, mac in macs.items()}
    attackers, exclude = set(attackers), set(exclude)
    buffers, mac_of, prev, rows = {}, {}, {}, []
    with open(path, "rb") as fh:
        for ts, buf in dpkt.pcap.Reader(fh):
            p = _parse_counted(ts, buf, stats)
            if p is None:
                continue
            for mac in {p.src_mac, p.dst_mac}:
                if mac in exclude or is_group_mac(mac) or not (all_hosts or mac in known):
                    continue
                name = host_name(mac, known)
                mac_of[name] = mac
                win = buffers.setdefault(name, [])
                win.append(p)
                if len(win) == WINDOW:
                    rows.append(_device_row(name, mac, win, prev.get(name, win[0].ts), attackers))
                    prev[name], buffers[name] = win[-1].ts, []
    for name in sorted(buffers):
        if buffers[name]:
            win = buffers[name]
            rows.append(_device_row(name, mac_of[name], win, prev.get(name, win[0].ts), attackers))
    return rows


def dominant_mac(path, exclude=()):
    """MAC present in the largest share of a single-device capture's packets (its device), with that share."""
    counts, total = collections.Counter(), 0
    for piece in read_pieces(path, chunk_bytes=float("inf")):
        for p in piece:
            total += 1
            for m in {p.src_mac, p.dst_mac} - set(exclude):
                counts[m] += 1
    if not counts:
        return None, 0.0
    mac, c = counts.most_common(1)[0]
    return mac, c / total
