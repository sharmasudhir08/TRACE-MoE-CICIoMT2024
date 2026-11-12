"""Check pcapfeat's CIC mode against CIC's official CSVs (gate G1 of TWIN_PROTOCOL.md).

CIC wrote the 10 MB pieces of a capture to the CSV in an arbitrary order, but each piece as one contiguous
block. So every piece is located in the CSV by its first rows, then compared row by row.
"""
import collections
import csv
import math

# Columns named differently in the attack CSVs.
ALIASES = {"Duration": "Time_To_Live"}


def _num(v):
    return float("nan") if v in ("", None) else float(v)


def _key(row):
    return tuple(round(_num(row[c]), 2) for c in ("Tot sum", "Max", "Min"))


def read_csv(path):
    with open(path, newline="") as fh:
        return [{ALIASES.get(k, k): v for k, v in r.items()} for r in csv.DictReader(fh)]


def _score(rows, official, start, sample=200):
    n = min(len(rows), sample, len(official) - start)
    return sum(_key(official[start + k]) == _key(rows[k]) for k in range(n)) / max(n, 1)


def align(computed, official, probe=5, accept=0.99):
    """Pairs (official_row, computed_row), placing each computed piece where it occurs in the CSV.

    Capture order is tried first (most files were written in order). Otherwise each piece goes to the
    candidate start whose first rows match and whose whole piece matches best: repetitive traffic (e.g.
    streams of identical packets) makes the first few rows alone ambiguous. Returns (pairs, rows not placed)."""
    by_piece = collections.defaultdict(list)
    for r in computed:
        by_piece[r["piece"]].append(r)
    pieces = [by_piece[k] for k in sorted(by_piece)]
    if len(computed) == len(official) and all(_key(o) == _key(c) for o, c in zip(official, computed)):
        return list(zip(official, computed)), 0
    where = collections.defaultdict(list)
    for i, r in enumerate(official):
        where[_key(r)].append(i)
    pairs, unplaced = [], 0
    for rows in pieces:
        n = min(probe, len(rows))
        cands = [s for s in where.get(_key(rows[0]), [])
                 if s + n <= len(official) and all(_key(official[s + k]) == _key(rows[k]) for k in range(n))]
        best = max(cands, key=lambda s: _score(rows, official, s), default=None)
        if best is None or _score(rows, official, best) < accept:
            unplaced += len(rows)
            continue
        pairs += [(official[best + k], rows[k]) for k in range(len(rows)) if best + k < len(official)]
    return pairs, unplaced


def same(a, b, rel=1e-4, absolute=1e-6):
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= absolute + rel * abs(b)


def same_rate(mine, official, n, timestamp_resolution=2e-6):
    """Rate = n / span. For windows of ~0.1 ms a span difference at the capture's microsecond timestamp
    resolution changes the rate by ~1%, so rates agree if their implied spans differ by <= the resolution."""
    if official == 0 or mine == 0:
        return same(mine, official)
    return abs(n / mine - n / official) <= timestamp_resolution


def agreement(pairs, features):
    """Share of rows on which each feature matches; blank official cells (e.g. Std of one packet) count
    as matching a NaN."""
    hits, total = collections.Counter(), collections.Counter()
    for off, mine in pairs:
        for c in features:
            if c in off and c in mine:
                total[c] += 1
                if c == "Rate":
                    hits[c] += same_rate(float(mine[c]), _num(off[c]), mine["Number"])
                else:
                    hits[c] += same(float(mine[c]), _num(off[c]))
    return {c: hits[c] / total[c] for c in total}
