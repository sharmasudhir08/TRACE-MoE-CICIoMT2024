"""Synchronised twin at test time (variant V4, paper §III.B, TWIN_PROTOCOL.md §6-7; added 2026-11-05).

The windows of one split are scored in time order (by the start time of their last row), in chunks of
`batch_size` windows, as a gateway would see them. Each row is compared with its profile the first time a
window needs it, using the profile as it is at that moment. After each chunk, the windows the detector calls
Benign with a calibrated confidence at or above the acceptance threshold are offered to the Synchroniser,
which applies the drift bound and the hourly budget (the poisoning guard) and updates the profile (Eq. 2).
Chunking means a profile update reaches the next chunk, not the rest of its own chunk. Peers and topics are
not synchronised, so the novelty flags stay as in the static twin.
"""
import collections
import copy

import numpy as np
import torch

from . import twin
from .abstain import _log_softmax, fit_temperature_logits
from .models.heads import LEVELS, energy_score
from .pcapfeat import FEATURES
from .train import twin_tensors
from .twin_data import window_inputs

NF = len(FEATURES)


def acceptance_rule(logits_val, y_val, target_coverage):
    """(temperature, log-confidence threshold) from the static validation pass: the same calibrated reject rule
    as the abstain layer, so "confident" means "would not abstain"."""
    T = fit_temperature_logits(logits_val, y_val)
    conf = _log_softmax(np.asarray(logits_val, dtype=np.float64) / T).max(1)
    return T, float(np.quantile(conf, 1 - target_coverage))


def window_delta_summary(ctx, starts, T, chunk=65536):
    """Per window: mean and max over features of Δ (Eq. 1), in chunks to bound memory."""
    mean, mx = np.zeros(len(starts), np.float32), np.zeros(len(starts), np.float32)
    for i in range(0, len(starts), chunk):
        idx = np.asarray(starts[i:i + chunk], dtype=np.int64)[:, None] + np.arange(T)
        d = ctx[idx, :NF].mean(axis=1)
        mean[i:i + chunk], mx[i:i + chunk] = d.mean(axis=1), d.max(axis=1)
    return mean, mx


def _profile_arrays(profiles, keys):
    mu = np.stack([profiles.profiles[k].mu for k in keys]).astype(np.float32)
    sigma = np.stack([profiles.profiles[k].sigma for k in keys]).astype(np.float32)
    return mu, sigma


@torch.no_grad()
def stream(model, wd, split, profiles, eta, delta, budget, rule, benign_index, device, batch_size=1024, amp=True):
    """Score wd.starts[split] with a synchronised copy of `profiles` (the fitted profiles are not changed).
    Returns the outputs of train.predict (with logits19) in the order of wd.starts[split], plus Δ per window
    (delta_mean, delta_max) and the Synchroniser's counts."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    temperature, threshold = rule
    sync = twin.Synchroniser(copy.deepcopy(profiles), eta, delta, budget)
    mu, sigma = _profile_arrays(sync.twin, wd.twin_keys)
    ctx = wd.ctx.copy()
    scored = np.zeros(len(ctx), dtype=bool)
    starts = wd.starts[split]
    order = np.argsort(wd.t[starts + wd.T - 1], kind="stable")
    out = collections.defaultdict(list)
    counts = collections.Counter()
    for i in range(0, len(order), batch_size):
        st = starts[order[i:i + batch_size]]
        idx = st[:, None] + np.arange(wd.T)
        new = np.unique(idx[~scored[idx]])
        if len(new):
            k = wd.key_id[new]
            ctx[new, :NF] = np.abs(wd.xlog[new] - mu[k]) / (sigma[k] + twin.EPSILON)
            scored[new] = True
        x, proto, _ = wd.gather(st)
        extra = twin_tensors(*window_inputs(ctx, wd.state_code, st, wd.T), device)
        xt = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        pt = torch.from_numpy(np.ascontiguousarray(proto)).to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, _ = model(xt, pt, twin=extra)
        pr = model.heads.probabilities(logits)
        for lvl in LEVELS:
            out[lvl].append(pr[lvl].cpu().numpy())
        z = logits["label19"].float().cpu().numpy()
        out["logits19"].append(z)
        out["energy"].append(energy_score(logits["label19"]).cpu().numpy())
        if getattr(model, "last_route", None) is not None:
            out["route"].append(model.last_route.cpu().numpy())
        d = ctx[idx, :NF].mean(axis=1)
        out["delta_mean"].append(d.mean(axis=1))
        out["delta_max"].append(d.max(axis=1))

        lp = _log_softmax(z.astype(np.float64) / temperature)
        ok = (lp.argmax(1) == benign_index) & (lp.max(1) >= threshold)
        counts["not_offered"] += int((~ok).sum())
        updated = False
        for j in np.flatnonzero(ok):
            last = idx[j, -1]
            updated |= sync.maybe_update(wd.devices[wd.device_id[last]], wd.state[last], wd.xlog[idx[j]],
                                         d[j], True, wd.t[last])
        if updated:
            mu, sigma = _profile_arrays(sync.twin, wd.twin_keys)
    back = np.empty_like(order)
    back[order] = np.arange(len(order))
    cat = {k: np.concatenate(v)[back] for k, v in out.items()}
    res = {"proba": {lvl: cat[lvl] for lvl in LEVELS}, "energy": cat["energy"], "logits19": cat["logits19"],
           "delta_mean": cat["delta_mean"].astype(np.float32), "delta_max": cat["delta_max"].astype(np.float32),
           "sync_counts": {**counts, **sync.counts}}
    if "route" in cat:
        res["route"] = cat["route"]
    return res
