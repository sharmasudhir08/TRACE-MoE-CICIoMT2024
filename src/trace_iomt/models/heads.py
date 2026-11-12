"""Hierarchical output heads and the losses that train them.

Levels: label2 (binary) <- label6 (category) <- label19 (attack type).
- Logit-adjusted loss (Menon et al., ICLR 2021) adds tau * log(prior) to the logits during training,
  which shifts decisions towards rare classes.
- Focal loss (Lin et al., 2017) down-weights easy examples.
- The consistency term asks each coarse head to agree with the summed probabilities of its children.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

LEVELS = ["label2", "label6", "label19"]


class HierarchicalHeads(nn.Module):
    def __init__(self, d_in, hierarchy, hierarchical=True, dropout=0.1):
        super().__init__()
        self.hierarchical = hierarchical
        self.n = {lvl: hierarchy.n(lvl) for lvl in LEVELS}
        self.register_buffer("parent19", torch.tensor(hierarchy.parent19, dtype=torch.long))
        self.register_buffer("parent6", torch.tensor(hierarchy.parent6, dtype=torch.long))
        self.drop = nn.Dropout(dropout)
        if hierarchical:
            self.heads = nn.ModuleDict({lvl: nn.Linear(d_in, self.n[lvl]) for lvl in LEVELS})
        else:
            self.heads = nn.ModuleDict({"label19": nn.Linear(d_in, self.n["label19"])})

    def forward(self, h):
        h = self.drop(h)
        return {lvl: head(h) for lvl, head in self.heads.items()}

    def aggregate(self, probs_fine, parent, n_parent):
        out = probs_fine.new_zeros(probs_fine.shape[0], n_parent)
        return out.index_add_(1, parent, probs_fine)

    def probabilities(self, logits):
        """Probabilities for all three levels. A flat model gets its coarse levels by summing children."""
        p = {lvl: F.softmax(v.float(), dim=-1) for lvl, v in logits.items()}
        if "label6" not in p:
            p["label6"] = self.aggregate(p["label19"], self.parent19, self.n["label6"])
        if "label2" not in p:
            p["label2"] = self.aggregate(p["label6"], self.parent6, self.n["label2"])
        return p


class HierarchicalLoss(nn.Module):
    def __init__(self, heads, priors, cfg_loss):
        """priors: dict level -> tensor of class frequencies in the training set."""
        super().__init__()
        self.heads = heads
        self.kind = cfg_loss["type"]
        self.gamma = cfg_loss.get("focal_gamma", 2.0)
        self.tau = cfg_loss.get("tau", 1.0)
        self.w = cfg_loss.get("head_weights", {"label2": 0.5, "label6": 0.5, "label19": 1.0})
        self.cons_w = cfg_loss.get("consistency_weight", 0.1)
        for lvl, pr in priors.items():
            pr = torch.as_tensor(pr, dtype=torch.float32).clamp_min(1.0)
            self.register_buffer(f"logprior_{lvl}", torch.log(pr / pr.sum()))

    def _single(self, logits, target, lvl):
        logits = logits.float()
        if "logit_adjusted" in self.kind:
            logits = logits + self.tau * getattr(self, f"logprior_{lvl}")
        logp = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(logp, target, reduction="none")
        if "focal" in self.kind:
            pt = torch.exp(-nll)
            nll = (1 - pt) ** self.gamma * nll
        return nll.mean()

    def forward(self, logits, targets):
        total, parts = 0.0, {}
        for lvl, lg in logits.items():
            l = self._single(lg, targets[lvl], lvl)
            parts[lvl] = l.detach()
            total = total + self.w.get(lvl, 1.0) * l
        if self.cons_w > 0 and len(logits) == 3:
            p = {lvl: F.softmax(v.float(), dim=-1) for lvl, v in logits.items()}
            agg6 = self.heads.aggregate(p["label19"], self.heads.parent19, self.heads.n["label6"])
            agg2 = self.heads.aggregate(p["label6"], self.heads.parent6, self.heads.n["label2"])
            cons = F.kl_div(torch.log(p["label6"] + 1e-8), agg6.detach(), reduction="batchmean") + \
                F.kl_div(torch.log(p["label2"] + 1e-8), agg2.detach(), reduction="batchmean")
            parts["consistency"] = cons.detach()
            total = total + self.cons_w * cons
        return total, parts


def energy_score(logits_fine, T=1.0):
    """Negative free energy: higher = more in-distribution. Used to flag unknown (zero-day) attacks."""
    return T * torch.logsumexp(logits_fine.float() / T, dim=-1)
