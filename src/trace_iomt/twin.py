"""Behavioural Digital Twin of the IoMT fleet (paper §III.A-B, TWIN_PROTOCOL.md §5; added 2026-11-04).

Works on per-device packet-window rows from trace_iomt.pcapfeat (device mode). Pieces:
  - profiles: mean and standard deviation of the features per (device, state), per device and for the
    whole fleet, plus the peers and MQTT topics seen; lookups fall back (device, state) -> device -> fleet
    when a profile has fewer than `min_support` rows, and hosts outside the twin use the fleet profile;
  - discrepancy Δ (Eq. 1): per feature, the window's mean absolute deviation from the profile in σ units;
  - novelty flags: share of rows talking to a peer, and whether any MQTT topic, not in the profile;
  - synchronisation (Eq. 2): EMA update of a profile, only for confident-benign windows within the drift
    bound δ and under an hourly update budget per device (the poisoning guard);
  - state estimator: a gradient-boosted tree classifier on summary statistics of a device's last W rows.
Features are compared on a log scale (log1p): counts, rates and byte sums are heavy-tailed, so σ on the raw
scale would be dominated by a few floods.
"""
import collections
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .pcapfeat import FEATURES

STATES = ("Power", "Idle", "Active", "Interactions")
STATE_ALIASES = {"Broker": "Active"}         # the Broker capture records the broker under active MQTT load
FLEET = "__fleet__"
EPSILON = 1e-6


def feature_matrix(df):
    """Rows x features on the twin's scale: log1p of the (non-negative) features; a single-packet row has
    no standard deviation (NaN), counted as 0."""
    x = df[list(FEATURES)].to_numpy(dtype=np.float64)
    return np.log1p(np.nan_to_num(np.clip(x, 0, None), nan=0.0))


def _split(cell):
    return [v for v in str(cell).split(";") if v] if isinstance(cell, str) else []


@dataclass
class Profile:
    mu: np.ndarray
    sigma: np.ndarray
    n: int
    peers: frozenset = frozenset()
    topics: frozenset = frozenset()


def _profile(x, peers, topics):
    return Profile(x.mean(axis=0), x.std(axis=0), len(x), frozenset(peers), frozenset(topics))


@dataclass
class TwinProfiles:
    profiles: dict                       # key: (device, state), (device, None) or (FLEET, None)
    twinned: frozenset
    min_support: int = 200
    fallbacks: collections.Counter = field(default_factory=collections.Counter)

    def lookup_key(self, device, state):
        """(profile key, level) for a device in a state; level is 'state', 'device' or 'fleet'. state=None asks
        for the device-level profile directly."""
        if device in self.twinned:
            if state is not None:
                key = (device, STATE_ALIASES.get(state, state))
                p = self.profiles.get(key)
                if p is not None and p.n >= self.min_support:
                    return key, "state"
            p = self.profiles.get((device, None))
            if p is not None and p.n >= self.min_support:
                return (device, None), "device"
        return (FLEET, None), "fleet"

    def lookup(self, device, state):
        """(profile, level) for a device in a state, counting fallbacks."""
        key, level = self.lookup_key(device, state)
        if level != "state":
            self.fallbacks[level] += 1
        return self.profiles[key], level


def shrink(p, fleet, k):
    """Empirical-Bayes shrinkage toward the fleet profile with prior strength k (rows): a feature that is
    constant in a device's training rows (σ = 0) would otherwise make any small change look like a huge
    deviation. Profiles with n >> k keep their own values."""
    w = p.n / (p.n + k)
    mu = w * p.mu + (1 - w) * fleet.mu
    var = w * p.sigma ** 2 + (1 - w) * fleet.sigma ** 2 + w * (1 - w) * (p.mu - fleet.mu) ** 2
    return Profile(mu, np.sqrt(var), p.n, p.peers, p.topics)


def fit_profiles(rows, twinned, min_support=200, shrinkage=50):
    """Profiles from benign training rows (profiling captures and Benign training traffic only, never test
    files). `rows` needs the FEATURES columns and device, state, peers and topics. Device and (device,
    state) profiles are shrunk toward the fleet profile with strength `shrinkage` rows (0 = off)."""
    x = feature_matrix(rows)
    devices = rows["device"].to_numpy()
    states = rows["state"].map(lambda s: STATE_ALIASES.get(s, s)).to_numpy()
    peers = [_split(v) for v in rows["peers"]]
    topics = [_split(v) for v in rows["topics"]]

    def make(mask):
        idx = np.flatnonzero(mask)
        return _profile(x[idx], {p for i in idx for p in peers[i]}, {t for i in idx for t in topics[i]})

    profiles = {(FLEET, None): make(np.ones(len(rows), dtype=bool))}
    fleet = profiles[(FLEET, None)]
    for d in sorted(set(twinned) & set(devices)):
        is_d = devices == d
        profiles[(d, None)] = make(is_d)
        for s in sorted({s for s in states[is_d] if s in STATES}):
            profiles[(d, s)] = make(is_d & (states == s))
    if shrinkage:
        profiles = {k: (v if k[0] == FLEET else shrink(v, fleet, shrinkage)) for k, v in profiles.items()}
    return TwinProfiles(profiles, frozenset(twinned), min_support)


def discrepancy(x_window, profile):
    """Eq. 1: mean over the window's rows of |x - μ| / (σ + ε), one value per feature."""
    return np.mean(np.abs(x_window - profile.mu) / (profile.sigma + EPSILON), axis=0)


def novelty(peers_cells, topics_cells, profile):
    """(share of rows with a peer the profile never saw, 1.0 if any topic is new or unparsable else 0.0)."""
    rows_new_peer = [any(p not in profile.peers for p in _split(c)) for c in peers_cells]
    topics = {t for c in topics_cells for t in _split(c)}
    new_topic = any(t == "?" or t not in profile.topics for t in topics)
    return (float(np.mean(rows_new_peer)) if rows_new_peer else 0.0), float(new_topic)


class Synchroniser:
    """Eq. 2 with the poisoning guard: profile(device, state) <- EMA with rate eta, only if the window is
    confidently benign, max |Δ| < delta, and the device has had fewer than `budget` updates in the last hour.
    Updates the profiles in place: give it a copy.deepcopy of the fitted TwinProfiles to keep a static twin."""

    def __init__(self, twin, eta, delta, budget_per_hour):
        self.twin, self.eta, self.delta, self.budget = twin, eta, delta, budget_per_hour
        self.history = collections.defaultdict(collections.deque)
        self.counts = collections.Counter()

    def maybe_update(self, device, state, x_window, delta_vec, confident_benign, t):
        profile, level = self.twin.lookup(device, state)
        recent = self.history[device]
        while recent and t - recent[0] >= 3600:
            recent.popleft()
        reason = ("not_twinned" if level == "fleet" else "not_confident" if not confident_benign
                  else "drift_bound" if np.max(delta_vec) >= self.delta else "budget" if len(recent) >= self.budget
                  else None)
        if reason:
            self.counts[f"rejected_{reason}"] += 1
            return False
        x_bar = x_window.mean(axis=0)
        new_mu = (1 - self.eta) * profile.mu + self.eta * x_bar
        profile.sigma = np.sqrt((1 - self.eta) * profile.sigma ** 2 + self.eta * (x_bar - new_mu) ** 2)
        profile.mu = new_mu
        recent.append(t)
        self.counts["updated"] += 1
        return True


def summary_features(x_rows):
    """State-estimator input: mean and standard deviation of each feature over a device's last W rows."""
    return np.concatenate([x_rows.mean(axis=0), x_rows.std(axis=0)])


def state_samples(rows, w=16, stride=8):
    """(X, y, groups, start, length) from profiling rows: windows of w consecutive rows of one device in one
    capture, in time order. groups names the (file, device); start is the window's first row and length the
    number of rows of that (file, device), used for a time split without overlap."""
    xs, ys, groups, starts, lengths = [], [], [], [], []
    for (f, d), g in rows.sort_values("t_start").groupby(["file", "device"], sort=False):
        state = STATE_ALIASES.get(g["state"].iat[0], g["state"].iat[0])
        if state not in STATES or len(g) < w:
            continue
        x = feature_matrix(g)
        for i in range(0, len(g) - w + 1, stride):
            xs.append(summary_features(x[i:i + w]))
            ys.append(state)
            groups.append(f"{f}|{d}")
            starts.append(i)
            lengths.append(len(g))
    return np.array(xs), np.array(ys), np.array(groups), np.array(starts), np.array(lengths)


class StateEstimator:
    def __init__(self, w=16, seed=0):
        self.w = w
        self.model = HistGradientBoostingClassifier(random_state=seed, class_weight="balanced")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_rows(self, x_rows):
        """State of a device from its last w rows (already on the twin's scale)."""
        return self.model.predict(summary_features(x_rows[-self.w:])[None, :])[0]


def holdout_by_time(starts, lengths, w=16, train_share=0.7):
    """Time split inside every (file, device): training windows end before the cut at 70% of its rows, test
    windows start after it, so no row is in both."""
    cut = np.floor(train_share * lengths)
    return starts + w <= cut, starts >= cut
