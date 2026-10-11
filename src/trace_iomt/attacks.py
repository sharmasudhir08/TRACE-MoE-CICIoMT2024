"""Feature-constrained adversarial evasion (contribution C2: robustness at 19 classes).

The attacker controls only timing and size statistics (robustness.mutable_features). Protocol
indicators, flags and counts stay fixed. Perturbations live in the scaled feature space with an
L-infinity budget eps, and every perturbed value must still map to a non-negative raw value
(times, sizes and rates cannot be negative). The attack is white-box PGD (Madry et al., 2018) on the
19-class head, applied to attack windows only (the attacker wants malicious traffic to be missed).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .data.windows import class_balanced_sample
from .models.heads import HierarchicalLoss
from .train import _priors, _to_tensors, iterate


def constraint_tensors(wd, mutable_names, device):
    """Mask of mutable features and the lower bound of each feature in scaled units."""
    norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())
    wanted = {norm(n) for n in mutable_names}
    mask = np.array([norm(f) in wanted for f in wd.features], dtype=np.float32)
    lower = (0.0 - wd.scaler.center_) / wd.scaler.scale_            # raw value 0 in scaled units
    lower = np.maximum(lower, -wd.scaler.clip).astype(np.float32)
    upper = np.full_like(lower, wd.scaler.clip)
    t = lambda a: torch.as_tensor(a, device=device)
    return t(mask), t(lower), t(upper)


def pgd(model, x, proto, y19, eps, steps, step_frac, mask, lower, upper, rand_init=True, momentum=0.0):
    """Return adversarial x (same shape). Only features with mask=1 move.
    momentum > 0 gives MI-FGSM (Dong et al., 2018), which usually transfers better to other models."""
    if eps <= 0:
        return x
    alpha = eps * step_frac if steps > 1 else eps
    x0 = x.detach()
    delta = (torch.empty_like(x0).uniform_(-eps, eps) if rand_init else torch.zeros_like(x0)) * mask
    acc = torch.zeros_like(x0)
    with torch.backends.cudnn.flags(enabled=False):     # RNN backward in eval mode needs cuDNN off
        for _ in range(steps):
            delta.requires_grad_(True)
            xa = torch.max(torch.min(x0 + delta, upper), lower)
            logits, _ = model(xa, proto)
            loss = F.cross_entropy(logits["label19"].float(), y19) if "label19" in logits else 0
            if "label2" in logits:      # also push the binary head towards "Benign"
                loss = loss + F.cross_entropy(logits["label2"].float(), torch.zeros_like(y19)) * -1.0
            g, = torch.autograd.grad(loss, delta)
            if momentum > 0:
                g = g / g.abs().mean(dim=tuple(range(1, g.dim())), keepdim=True).clamp_min(1e-12)
                acc = momentum * acc + g
                g = acc
            delta = (delta.detach() + alpha * g.sign()) * mask
            delta = delta.clamp(-eps, eps)
    return torch.max(torch.min(x0 + delta.detach(), upper), lower).detach()


def attack_eval_starts(wd, max_windows, seed, benign_index):
    """Balanced sample of test windows: attack windows (to perturb) and benign windows (kept clean)."""
    lab = wd.labels("test", "label19")
    idx = class_balanced_sample(lab, max(1, max_windows // max(len(np.unique(lab)), 1)), seed)
    return wd.starts["test"][np.sort(idx)]


@torch.no_grad()
def _predict_batch(model, x, p):
    logits, _ = model(x, p)
    return model.heads.probabilities(logits)


def _asr_per_class(ys, p2, hierarchy):
    """Attack success rate per attack class: share of its (perturbed) windows the binary head calls Benign."""
    out = {}
    names = hierarchy.classes["label19"]
    for k in np.unique(ys["label19"][ys["label2"] == 1]):
        m = ys["label19"] == k
        out[names[int(k)]] = float((p2[m].argmax(1) == 0).mean())
    return out


def robustness_curve(model, wd, cfg, device, starts, logger, extra=None):
    """Metrics on the same windows for every eps. Benign windows are never perturbed.

    extra: optional {name: fn(X_last_rows) -> 19-class probabilities} for models that cannot be
    attacked with gradients (trees). They are evaluated on the adversarial windows crafted against
    `model` (a transfer attack)."""
    from .metrics import evaluate_levels
    rc = cfg["robustness"]
    mask, lower, upper = constraint_tensors(wd, rc["mutable_features"], device)
    benign19 = {i for i, c in enumerate(wd.hierarchy.classes["label19"])
                if wd.hierarchy.parent6[wd.hierarchy.parent19[i]] == 0}
    model.eval()
    out = {}
    for eps in [0.0] + list(rc["epsilons"]):
        probs = {k: [] for k in ("label2", "label6", "label19")}
        ys = {k: [] for k in probs}
        last_rows = []
        for x, p, y in iterate(wd, starts, 512, device):
            is_attack = torch.tensor([int(v) not in benign19 for v in y["label19"].tolist()], device=device)
            xa = x
            if eps > 0 and is_attack.any():
                adv = pgd(model, x[is_attack], p[is_attack], y["label19"][is_attack], eps, rc["pgd_steps"],
                          rc["pgd_step_frac"], mask, lower, upper, momentum=rc.get("momentum", 0.0))
                xa = x.clone()
                xa[is_attack] = adv
            pr = _predict_batch(model, xa, p)
            if extra:
                last_rows.append(xa[:, -1].cpu().numpy())
            for k in probs:
                probs[k].append(pr[k].cpu().numpy())
                ys[k].append(y[k].cpu().numpy())
        probs = {k: np.concatenate(v) for k, v in probs.items()}
        ys = {k: np.concatenate(v) for k, v in ys.items()}
        m = evaluate_levels(ys, probs, wd.hierarchy, cfg["evaluation"]["recall_at_fpr"])
        att = ys["label2"] == 1
        det = float((probs["label2"][att].argmax(1) == 1).mean()) if att.any() else float("nan")
        m["attack_detection_rate"] = det
        m["asr_per_class"] = _asr_per_class(ys, probs["label2"], wd.hierarchy)
        if extra:
            from .experiment import fine_to_coarse
            Xl = np.concatenate(last_rows)
            m["transfer"] = {}
            for name, fn in extra.items():
                pe = fine_to_coarse(fn(Xl), wd.hierarchy)
                me = evaluate_levels(ys, pe, wd.hierarchy, cfg["evaluation"]["recall_at_fpr"])
                me["attack_detection_rate"] = float((pe["label2"][att].argmax(1) == 1).mean()) if att.any() else float("nan")
                me["asr_per_class"] = _asr_per_class(ys, pe["label2"], wd.hierarchy)
                m["transfer"][name] = me
        out[str(eps)] = m
        logger.info(f"eps={eps:<5} macro-F1 19:{m['label19']['macro_f1']:.4f} 2:{m['label2']['macro_f1']:.4f} "
                    f"attack detection rate {det:.4f}")
    return out


def adversarial_finetune(model, wd, cfg, device, logger):
    """A few epochs of training on a 50/50 mix of clean and PGD windows (fewer PGD steps for speed)."""
    rc, tc = cfg["robustness"], cfg["train"]
    mask, lower, upper = constraint_tensors(wd, rc["mutable_features"], device)
    criterion = HierarchicalLoss(model.heads, _priors(wd), cfg["loss"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"] * 0.3, weight_decay=tc["weight_decay"])
    benign19 = torch.tensor([wd.hierarchy.parent6[wd.hierarchy.parent19[i]] == 0
                             for i in range(wd.hierarchy.n("label19"))], device=device)
    lab = wd.labels("train", "label19")
    steps = max(3, rc["pgd_steps"] // 4)
    oversample = bool(cfg.get("data", {}).get("oversample_small_classes", False))
    for ep in range(1, rc["adv_finetune_epochs"] + 1):
        idx = class_balanced_sample(lab, max(1, cfg["windows"]["max_windows_per_class_train"] // 4),
                                     tc["seed"] + 100 + ep, oversample=oversample)
        tot, n = 0.0, 0
        for x, p, y in iterate(wd, wd.starts["train"][idx], tc["batch_size"], device):
            att = ~benign19[y["label19"]]
            half = att & (torch.rand(len(x), device=device) < 0.5)
            if half.any():
                model.eval()
                eps = float(np.random.uniform(0, rc["adv_epsilon"]))
                x = x.clone()
                x[half] = pgd(model, x[half], p[half], y["label19"][half], eps, steps, rc["pgd_step_frac"] * 2,
                              mask, lower, upper)
            model.train()
            logits, _ = model(x, p)
            loss, _ = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip") or 1.0)
            opt.step()
            tot += loss.detach().item(); n += 1
        logger.info(f"adversarial fine-tune epoch {ep}: loss {tot / max(n, 1):.4f}")
    return model


def transfer_matrix(models, trees, wd, cfg, device, starts, logger):
    """Black-box transfer attacks: adversarial attack windows crafted on each surrogate model are given to every
    target model (neural and tree). Returns rows {surrogate, target, asr, white_box}. ASR = share of attack
    windows the target labels Benign (binary level)."""
    rc = cfg["robustness"]
    eps = rc.get("transfer_epsilon", max(rc["epsilons"]))
    mask, lower, upper = constraint_tensors(wd, rc["mutable_features"], device)
    benign19 = torch.tensor([wd.hierarchy.parent6[wd.hierarchy.parent19[i]] == 0
                             for i in range(wd.hierarchy.n("label19"))], device=device)
    from .experiment import fine_to_coarse
    rows = []
    for sname, smodel in models.items():
        smodel.eval()
        flagged = {t: 0 for t in list(models) + list(trees)}
        n_att = 0
        for x, p, y in iterate(wd, starts, 512, device):
            att = ~benign19[y["label19"]]
            if not att.any():
                continue
            xa = pgd(smodel, x[att], p[att], y["label19"][att], eps, rc["pgd_steps"], rc["pgd_step_frac"],
                     mask, lower, upper, momentum=rc.get("momentum", 0.0))
            pa = p[att]
            n_att += int(att.sum())
            for tname, tmodel in models.items():
                tmodel.eval()
                pr = _predict_batch(tmodel, xa, pa)
                flagged[tname] += int((pr["label2"].argmax(1) == 1).sum())
            if trees:
                last = xa[:, -1].cpu().numpy()
                for tname, fn in trees.items():
                    p2 = fine_to_coarse(fn(last), wd.hierarchy)["label2"]
                    flagged[tname] += int((p2.argmax(1) == 1).sum())
        for tname, k in flagged.items():
            rows.append({"surrogate": sname, "target": tname, "eps": eps, "n_attack_windows": n_att,
                         "asr": 1 - k / max(n_att, 1), "white_box": tname == sname})
        logger.info(f"transfer from {sname}: " + ", ".join(f"{r['target']} ASR {r['asr']:.3f}"
                                                            for r in rows if r["surrogate"] == sname))
    return rows
