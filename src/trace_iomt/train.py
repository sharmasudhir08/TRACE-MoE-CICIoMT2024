"""Training and prediction loops for the neural models (mixed precision, early stopping, MoE balance loss)."""
import copy
import math
import time

import numpy as np
import pandas as pd
import torch

from .data.windows import class_balanced_sample, class_counts
from .metrics import evaluate_levels
from .models.heads import LEVELS, HierarchicalLoss, energy_score
from .twin_data import window_inputs


def twin_tensors(vec, state, device):
    return (torch.from_numpy(vec).to(device, non_blocking=True),
            torch.from_numpy(np.ascontiguousarray(state)).to(device, non_blocking=True))


def _to_tensors(wd, starts, device, with_twin=False):
    x, proto, y = wd.gather(starts)
    xt = torch.from_numpy(np.ascontiguousarray(x)).to(device, non_blocking=True)
    pt = torch.from_numpy(np.ascontiguousarray(proto)).to(device, non_blocking=True)
    yt = {lvl: torch.from_numpy(v).to(device) for lvl, v in y.items()}
    extra = {"twin": twin_tensors(*window_inputs(wd.ctx, wd.state_code, starts, wd.T), device)} if with_twin else {}
    return xt, pt, yt, extra


def uses_twin(model):
    """True for a TRACE-MoE built with model.twin.inputs (the twin study); it then needs wd.ctx."""
    return bool(getattr(model, "twin_inputs", None))


def iterate(wd, starts, batch_size, device, with_twin=False):
    for i in range(0, len(starts), batch_size):
        yield _to_tensors(wd, starts[i:i + batch_size], device, with_twin)


@torch.no_grad()
def predict(model, wd, starts, device, batch_size=1024, amp=True, return_embeddings=False, return_logits=False):
    """Probabilities for every level, the energy score, the expert chosen per window (TRACE-MoE only)
    and (optionally) embeddings and float32 19-class logits (for confidence scores that do not saturate)."""
    model.eval()
    probs = {lvl: [] for lvl in LEVELS}
    energies, embs, routes, logits19 = [], [], [], []
    use_amp = amp and device.type == "cuda"
    for x, p, _, extra in iterate(wd, starts, batch_size, device, uses_twin(model)):
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, h = model(x, p, **extra)
        pr = model.heads.probabilities(logits)
        for lvl in LEVELS:
            probs[lvl].append(pr[lvl].cpu().numpy())
        energies.append(energy_score(logits["label19"]).cpu().numpy())
        if getattr(model, "last_route", None) is not None:
            routes.append(model.last_route.cpu().numpy())
        if return_embeddings:
            embs.append(h.float().cpu().numpy())
        if return_logits:
            logits19.append(logits["label19"].float().cpu().numpy())
    out = {"proba": {lvl: np.concatenate(v) if v else np.zeros((0, model.heads.n[lvl]))
                     for lvl, v in probs.items()},
           "energy": np.concatenate(energies) if energies else np.zeros(0)}
    if routes:
        out["route"] = np.concatenate(routes)
    if return_embeddings:
        out["embeddings"] = np.concatenate(embs)
    if return_logits:
        out["logits19"] = np.concatenate(logits19) if logits19 else np.zeros((0, model.heads.n["label19"]))
    return out


def evaluate(model, wd, split, device, cfg, starts=None, return_logits=False):
    starts = wd.starts[split] if starts is None else starts
    pred = predict(model, wd, starts, device, amp=cfg["train"]["amp"], return_logits=return_logits)
    last = starts + wd.T - 1
    y_true = {lvl: wd.y[lvl][last] for lvl in LEVELS}
    metrics = evaluate_levels(y_true, pred["proba"], wd.hierarchy, cfg["evaluation"]["recall_at_fpr"])
    return metrics, pred, y_true


def _priors(wd):
    lab = {lvl: wd.labels("train", lvl) for lvl in LEVELS}
    return {lvl: class_counts(lab[lvl], wd.hierarchy.n(lvl)) for lvl in LEVELS}


def train_model(model, wd, cfg, device, logger):
    """Train with early stopping on validation macro-F1 of the finest level.
    Models with experts add model.aux_loss (load balancing) times model.moe.balance_weight.
    Returns (best_model, history DataFrame).
    """
    tc = cfg["train"]
    model.to(device)
    criterion = HierarchicalLoss(model.heads, _priors(wd), cfg["loss"]).to(device)
    params = list(model.parameters())
    balance_w = float((cfg["model"].get("moe") or {}).get("balance_weight", 0.01))
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(tc["epochs"], 1))
    use_amp = tc["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_starts = wd.starts["train"]
    train_lab = wd.labels("train", "label19")
    oversample = bool(cfg.get("data", {}).get("oversample_small_classes", False))
    best, best_score, bad, hist = None, -1.0, 0, []
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        t0 = time.time()
        idx = class_balanced_sample(train_lab, cfg["windows"]["max_windows_per_class_train"],
                                     tc["seed"] + epoch, oversample=oversample)
        starts = train_starts[idx]
        n_steps = math.ceil(len(starts) / tc["batch_size"])
        tot, parts_sum, aux_sum = 0.0, {}, 0.0
        for step, (x, p, y, extra) in enumerate(iterate(wd, starts, tc["batch_size"], device, uses_twin(model))):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, h = model(x, p, **extra)
                loss, parts = criterion(logits, y)
                aux = getattr(model, "aux_loss", None)
                if aux is not None and balance_w > 0:
                    loss = loss + balance_w * aux
                    aux_sum += aux.detach().item()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tc.get("grad_clip"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            tot += loss.detach().item()
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + (v.detach().item() if hasattr(v, "detach") else float(v))
        sched.step()

        val_metrics, _, _ = evaluate(model, wd, "val", device, cfg) if len(wd.starts.get("val", [])) else ({}, None, None)
        score = val_metrics.get("label19", {}).get("macro_f1", -tot)
        row = {"epoch": epoch, "train_loss": tot / max(n_steps, 1), "lr": opt.param_groups[0]["lr"],
               "seconds": round(time.time() - t0, 1),
               **{f"loss_{k}": v / max(n_steps, 1) for k, v in parts_sum.items()},
               **{f"val_macro_f1_{lvl}": val_metrics.get(lvl, {}).get("macro_f1") for lvl in LEVELS}}
        if getattr(model, "aux_loss", None) is not None or aux_sum:
            row["balance_loss"] = aux_sum / max(n_steps, 1)
        hist.append(row)
        logger.info(f"epoch {epoch:3d} | loss {row['train_loss']:.4f} | val macro-F1 "
                    f"2:{row['val_macro_f1_label2'] or 0:.4f} 6:{row['val_macro_f1_label6'] or 0:.4f} "
                    f"19:{row['val_macro_f1_label19'] or 0:.4f} | {row['seconds']}s")
        if score > best_score + 1e-4:
            best_score, best, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tc["patience"]:
                logger.info(f"Early stopping at epoch {epoch} (best val macro-F1 {best_score:.4f})")
                break
    if best is not None:
        model.load_state_dict(best)
    return model, pd.DataFrame(hist)
