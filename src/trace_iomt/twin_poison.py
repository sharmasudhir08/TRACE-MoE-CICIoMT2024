"""Poisoning test of the synchronised twin (TWIN_PROTOCOL.md §9, paper §V.E; added 2026-11-05).

A slow attacker who knows the profile sends benign-looking windows that move a device's profile toward that
device's attack profile. Every window sits just inside the drift bound (max |Δ| = margin x δ) and the attacker
uses the whole hourly budget. The detector is assumed to call every such window confident benign (worst case),
so only the poisoning guard limits the drift. The updates go through twin.Synchroniser, the same code as V4.
Measured: profile drift after each horizon (mean |μ_t - μ_0| / σ_0 over features, and the share of the way to
the attack profile), and the detection rate of a trained model on the device's attack test windows scored with
the drifted profile against the static one.
"""
import copy

import numpy as np
import torch

from . import twin
from .pcapfeat import FEATURES
from .train import twin_tensors
from .twin_data import window_inputs_at

NF = len(FEATURES)


def poisoned_key(profiles, device):
    """The profile the attacker targets: the device's best-supported state profile, else its device profile.
    Returns (key, state to pass to lookup)."""
    states = [(p.n, k) for k, p in profiles.profiles.items()
              if k[0] == device and k[1] is not None and p.n >= profiles.min_support]
    if states:
        key = max(states)[1]
        return key, key[1]
    key, _ = profiles.lookup_key(device, None)
    return key, None


def attack_profile(rows, device, min_rows=16):
    """(attack class, mean feature vector on the twin's scale, rows) of the attack the device receives most in
    the training captures: rows with packets to or from an attacker, in attack captures. None if too few rows."""
    label = rows["capture_label"].astype(str)
    m = ((rows["device"] == device) & (rows["split"] == "train") & (label != "Benign")
         & (rows["n_attacker_packets"] > 0))
    if m.sum() < min_rows:
        return None
    cls = label[m].value_counts().idxmax()
    sel = rows[m & (label == cls)]
    return cls, twin.feature_matrix(sel).mean(axis=0), int(len(sel))


def drift(p0, p, mu_attack):
    """Mean drift over features in σ_0 units, and the share of the distance to the attack profile covered."""
    d = (p.mu - p0.mu) / (p0.sigma + twin.EPSILON)
    full = (mu_attack - p0.mu) / (p0.sigma + twin.EPSILON)
    share = float(np.dot(d, full) / np.dot(full, full)) if np.dot(full, full) > 0 else 0.0
    return {"mean_sigma": float(np.abs(d).mean()), "share_to_attack": share}


def simulate(profiles, device, state, mu_attack, eta, delta, budget, hours=(1, 24), T=16, margin=0.99):
    """Run the attacker for max(hours) hours; returns {hours: (TwinProfiles snapshot, drift)} and the counts."""
    sync = twin.Synchroniser(copy.deepcopy(profiles), eta, delta, budget)
    key, _ = sync.twin.lookup_key(device, state)
    p0 = copy.deepcopy(sync.twin.profiles[key])
    step_s = 3600.0 / budget
    out = {}
    for i in range(int(max(hours) * budget)):
        p = sync.twin.profiles[key]
        u = mu_attack - p.mu
        ratio = np.abs(u) / (p.sigma + twin.EPSILON)
        s = min(1.0, margin * delta / ratio.max()) if ratio.max() > 0 else 0.0
        x = p.mu + s * u
        sync.maybe_update(device, state, np.tile(x, (T, 1)), np.abs(x - p.mu) / (p.sigma + twin.EPSILON), True,
                          i * step_s)
        h = (i + 1) / budget
        if h in hours:
            out[h] = (copy.deepcopy(sync.twin), drift(p0, sync.twin.profiles[key], mu_attack))
    return out, dict(sync.counts)


@torch.no_grad()
def attack_detection(model, wd, starts, profiles, device, batch_size=1024, amp=True):
    """Share of the windows predicted Attack (binary head) when their rows are compared with `profiles`."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    idx = np.asarray(starts, dtype=np.int64)[:, None] + np.arange(wd.T)
    rows = np.unique(idx)
    keys = wd.twin_keys
    mu = np.stack([profiles.profiles[k].mu for k in keys]).astype(np.float32)
    sigma = np.stack([profiles.profiles[k].sigma for k in keys]).astype(np.float32)
    k = wd.key_id[rows]
    ctx = wd.ctx[rows].copy()
    ctx[:, :NF] = np.abs(wd.xlog[rows] - mu[k]) / (sigma[k] + twin.EPSILON)
    state = wd.state_code[rows]
    attack = wd.hierarchy.index("label2")["Attack"]
    hits = 0
    for i in range(0, len(starts), batch_size):
        st = starts[i:i + batch_size]
        local = np.searchsorted(rows, idx[i:i + batch_size])
        x, proto, _ = wd.gather(st)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, _ = model(torch.from_numpy(np.ascontiguousarray(x)).to(device),
                              torch.from_numpy(np.ascontiguousarray(proto)).to(device),
                              twin=twin_tensors(*window_inputs_at(ctx, state, local), device))
        hits += int((model.heads.probabilities(logits)["label2"].argmax(1) == attack).sum())
    return hits / max(len(starts), 1)
