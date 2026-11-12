"""Neural models. All take x: (B, T, F) scaled features and proto: (B, T) protocol tokens,
and return (logits_by_level, embedding).

Baselines (one per model family, each a winner in published CICIoMT2024 / special-issue work):
  bilstm_dnn   recurrent, row-level, as in Benahmed et al. (2025)
  transformer  attention over the flow window, the family of Jodayree et al. (2026)
  tfsage_lite  trust-filtered graph model, simplified from Shieh et al. (CMC special issue)
  (rf, the tree-ensemble winner, lives in models/classical.py)
Ours:
  trace_moe    TRACE encoder + protocol-routed mixture of experts + hierarchical heads (+ abstain layer,
               applied after training in abstain.py)
  trace        the same encoder without the experts (used by the ablation)
TRACE-MoE also takes the Digital Twin's context when model.twin.inputs is set (paper §III.E-F, twin study).
"""
import torch
import torch.nn as nn

from ..data.preprocess import N_PROTO_TOKENS
from .heads import HierarchicalHeads

N_TWIN_STATES = 5          # 0 = no state, then twin.STATES (Power, Idle, Active, Interactions)


class _Base(nn.Module):
    row_level = False

    def __init__(self, n_features, hierarchy, cfg_model):
        super().__init__()
        self.F = n_features
        self.d = cfg_model["d_model"]
        self.heads = HierarchicalHeads(self.d, hierarchy, cfg_model.get("hierarchical", True),
                                       cfg_model.get("dropout", 0.1))

    def encode(self, x, proto):
        raise NotImplementedError

    def forward(self, x, proto=None):
        if self.row_level:
            x = x[:, -1:, :]
            proto = proto[:, -1:] if proto is not None else None
        h = self.encode(x, proto)
        return self.heads(h), h


# ----------------------------------------------------------------- row-level baseline
class BiLSTMDNN(_Base):
    """Literature-style BiLSTM-DNN: the features of ONE flow are read as a sequence (no time axis)."""
    row_level = True

    def __init__(self, n_features, hierarchy, cfg_model):
        super().__init__(n_features, hierarchy, cfg_model)
        self.lstm = nn.LSTM(1, 64, batch_first=True, bidirectional=True)
        p = cfg_model.get("dropout", 0.1)
        self.dense = nn.Sequential(nn.Dropout(p), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p),
                                   nn.Linear(128, self.d), nn.ReLU())

    def encode(self, x, proto):
        out, _ = self.lstm(x[:, -1].unsqueeze(-1))           # (B, F, 128)
        return self.dense(out[:, -1])


class FeatureTokenizer(nn.Module):
    """Each numeric feature j becomes a token W_j * x_j + b_j (as in FT-Transformer)."""

    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x):                                    # (..., F) -> (..., F, d)
        return x.unsqueeze(-1) * self.weight + self.bias


# ----------------------------------------------------------------- sequence models
class StepEmbedding(nn.Module):
    """Embed each time step: linear projection of the features (+ protocol token)."""

    def __init__(self, n_features, d, use_proto=True):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_features, d), nn.LayerNorm(d), nn.GELU())
        self.proto = nn.Embedding(N_PROTO_TOKENS, d) if use_proto else None

    def forward(self, x, proto):
        h = self.proj(x)
        if self.proto is not None and proto is not None:
            h = h + self.proto(proto)
        return h


class TransformerSeq(_Base):
    def __init__(self, n_features, hierarchy, cfg_model, max_len=256):
        super().__init__(n_features, hierarchy, cfg_model)
        self.emb = StepEmbedding(n_features, self.d, cfg_model.get("use_protocol_token", True))
        self.pos = nn.Parameter(torch.zeros(1, max_len, self.d))
        layer = nn.TransformerEncoderLayer(self.d, cfg_model["n_heads"], cfg_model["ff_dim"],
                                           cfg_model.get("dropout", 0.1), batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg_model["n_layers"], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(self.d)

    def encode(self, x, proto):
        h = self.emb(x, proto) + self.pos[:, :x.shape[1]]
        return self.norm(self.enc(h).mean(dim=1))


class AttentionPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, h):                                    # (B, L, d) -> (B, d)
        w = torch.softmax(self.score(h).squeeze(-1), dim=-1)
        return (w.unsqueeze(-1) * h).sum(dim=1)


class TRACE(_Base):
    """TRACE encoder (also the "no experts" ablation of TRACE-MoE).

    1. Feature tokenizer per time step, pooled with attention over features -> step embedding
       (+ protocol token from the 'Protocol Type' feature).
    2. BiGRU over the T steps for local dynamics.
    3. Transformer encoder (learned positions) for longer-range context.
    4. Attention pooling over time, concatenated with the last step -> hierarchical heads.
    """

    def __init__(self, n_features, hierarchy, cfg_model, max_len=256):
        super().__init__(n_features, hierarchy, cfg_model)
        d, dt, p = self.d, cfg_model.get("token_dim", 16), cfg_model.get("dropout", 0.1)
        self.tok = FeatureTokenizer(n_features, dt)
        self.feat_pool = AttentionPool(dt)
        self.step = nn.Sequential(nn.Linear(dt + n_features, d), nn.LayerNorm(d), nn.GELU())
        self.proto = nn.Embedding(N_PROTO_TOKENS, d) if cfg_model.get("use_protocol_token", True) else None
        self.gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        layer = nn.TransformerEncoderLayer(d, cfg_model["n_heads"], cfg_model["ff_dim"], p,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg_model["n_layers"], enable_nested_tensor=False)
        self.time_pool = AttentionPool(d)
        self.out = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU())

    def encode(self, x, proto):
        B, T, F = x.shape
        tokens = self.tok(x.reshape(B * T, F))               # (B*T, F, dt)
        pooled = self.feat_pool(tokens).reshape(B, T, -1)    # (B, T, dt)
        h = self.step(torch.cat([pooled, x], dim=-1))        # raw features kept as a skip path
        if self.proto is not None and proto is not None:
            h = h + self.proto(proto)
        h, _ = self.gru(h)
        h = self.enc(h + self.pos[:, :T])
        return self.out(torch.cat([self.time_pool(h), h[:, -1]], dim=-1))


class Expert(nn.Module):
    """Residual feed-forward expert: d -> hidden -> d."""

    def __init__(self, d, hidden, p):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(p), nn.Linear(hidden, d))

    def forward(self, h):
        return self.net(h)


class TRACEMoE(TRACE):
    """TRACE-MoE (ours): the TRACE window encoder followed by a protocol-routed mixture of experts.

    The router reads the window summary h and an embedding of the last flow's protocol token and picks
    ONE of n_experts small experts (top-1, Switch-style: the expert output is scaled by its gate
    probability so the router gets gradients). Only the chosen expert runs for each window. A load-balancing
    loss (Fedus et al., 2022) keeps every expert in use; train.py adds it through `aux_loss`.

    model.moe.routing:
      learned   router on [h, protocol embedding] (default)
      protocol  fixed routing by protocol token (ablation: no learned router)
      dense     one expert with n_experts x the hidden size, so the parameter count matches (ablation)
    The expert chosen for every window is kept in `last_route` (used for the specialisation analysis, H5).

    model.twin (twin study, added 2026-11-05; off by default):
      inputs    None (No-Twin), "delta" (Δ, novelty flags and profile level) or "delta_state" (also the state)
      n_inputs  length of the twin vector (twin_data.TWIN_INPUT_DIM)
      hidden    width of the two-layer MLP φ that maps the twin context to d (paper: h = h0 + φ(Δ, flags, s_d))
    With "delta_state" the router also reads the state embedding. forward() then needs twin=(vector, state).
    """

    def __init__(self, n_features, hierarchy, cfg_model, max_len=256):
        super().__init__(n_features, hierarchy, cfg_model, max_len)
        mc = {"n_experts": 4, "routing": "learned", "expert_hidden": 2, **(cfg_model.get("moe") or {})}
        d, p = self.d, cfg_model.get("dropout", 0.1)
        self.n_experts = int(mc["n_experts"])
        self.routing = mc["routing"]
        hidden = int(mc["expert_hidden"] * d)
        if self.routing == "dense":
            self.experts = nn.ModuleList([Expert(d, hidden * self.n_experts, p)])
        else:
            self.experts = nn.ModuleList([Expert(d, hidden, p) for _ in range(self.n_experts)])
        tc = cfg_model.get("twin") or {}
        self.twin_inputs = tc.get("inputs")
        if self.twin_inputs not in (None, "delta", "delta_state"):
            raise ValueError(f"model.twin.inputs must be null, delta or delta_state, got '{self.twin_inputs}'")
        uses_state = self.twin_inputs == "delta_state"
        self.state_emb = nn.Embedding(N_TWIN_STATES, d // 4) if uses_state else None
        if self.twin_inputs:
            n_in = int(tc["n_inputs"]) + (d // 4 if uses_state else 0)
            hidden = int(tc.get("hidden", 64))
            self.phi = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.router_proto = nn.Embedding(N_PROTO_TOKENS, d // 4)
        self.router = nn.Linear(d + d // 4 + (d // 4 if uses_state else 0), self.n_experts)
        self.aux_loss = None
        self.last_route = None

    def route(self, h, proto, s=None):
        last = proto[:, -1] if proto is not None else torch.zeros(len(h), dtype=torch.long, device=h.device)
        if self.routing == "protocol":
            idx = last % self.n_experts
            return idx, torch.ones(len(h), device=h.device, dtype=h.dtype), None
        parts = [h, self.router_proto(last)] + ([s] if s is not None else [])
        logits = self.router(torch.cat(parts, dim=-1)).float()
        probs = torch.softmax(logits, dim=-1)
        gate, idx = probs.max(dim=-1)
        return idx, gate.to(h.dtype), probs

    def add_twin(self, h, twin):
        """h = h0 + φ(Δ, flags, level[, state embedding]); returns (h, state embedding or None)."""
        if not self.twin_inputs:
            return h, None
        if twin is None:
            raise ValueError("this TRACE-MoE was built with model.twin.inputs set and needs the twin context")
        vec, state = twin
        s = self.state_emb(state) if self.state_emb is not None else None
        z = torch.cat([vec, s], dim=-1) if s is not None else vec
        return h + self.phi(z.to(h.dtype)), s

    def forward(self, x, proto=None, twin=None):
        h, s = self.add_twin(self.encode(x, proto), twin)
        if self.routing == "dense":
            out = h + self.experts[0](h)
            self.aux_loss, self.last_route = None, torch.zeros(len(h), dtype=torch.long, device=h.device)
        else:
            idx, gate, probs = self.route(h, proto, s)
            out = h.clone()
            for e, expert in enumerate(self.experts):
                sel = idx == e
                if sel.any():
                    out[sel] = h[sel] + gate[sel].unsqueeze(-1) * expert(h[sel])
            self.last_route = idx.detach()
            if probs is not None and self.training:
                frac = torch.bincount(idx, minlength=self.n_experts).float() / max(len(idx), 1)
                self.aux_loss = self.n_experts * (frac * probs.mean(dim=0)).sum()
            else:
                self.aux_loss = None
        return self.heads(out), out


class WindowGraphSAGE(_Base):
    """Graph baseline in the spirit of graph-based IoT IDS (e.g. GraphSAGE / TF-SAGE in the same special issue).

    The T flows of a window are the nodes. Each node is linked to its k most similar flows (cosine similarity
    of the scaled features). Two GraphSAGE layers (self + mean of neighbours) are applied, and the last flow's
    node and the mean over nodes are read out. With trust_threshold set (TFSAGELite), edges whose similarity is
    below the threshold are dropped before message passing, keeping at least k_min edges per node, which is
    the trust-filtering idea of TF-SAGE. This is a simplified re-implementation, not the authors' code
    (no stochastic uncertainty inference or robust-training term).
    """
    trust_threshold = None

    def __init__(self, n_features, hierarchy, cfg_model):
        super().__init__(n_features, hierarchy, cfg_model)
        d, p = self.d, cfg_model.get("dropout", 0.1)
        self.k = cfg_model.get("graph_k", 4)
        self.k_min = cfg_model.get("graph_k_min", 1)
        self.emb = StepEmbedding(n_features, d, cfg_model.get("use_protocol_token", True))
        self.layers = nn.ModuleList([nn.ModuleDict({"self": nn.Linear(d, d), "neigh": nn.Linear(d, d)}) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.drop = nn.Dropout(p)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())

    def adjacency(self, x):
        B, T, _ = x.shape
        if T == 1:
            return x.new_zeros(B, 1, 1)
        z = torch.nn.functional.normalize(x.float(), dim=-1)
        sim = z @ z.transpose(1, 2)
        eye = torch.eye(T, device=x.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, -2.0)
        k = min(self.k, T - 1)
        top = sim.topk(k, dim=-1)
        A = torch.zeros_like(sim).scatter_(-1, top.indices, 1.0)
        if self.trust_threshold is not None:
            keep = (sim >= self.trust_threshold).float()
            kmin = min(self.k_min, k)
            must = torch.zeros_like(sim).scatter_(-1, top.indices[..., :kmin], 1.0)
            A = A * torch.maximum(keep, must)
        return A

    def encode(self, x, proto):
        A = self.adjacency(x).to(x.dtype)
        deg = A.sum(-1, keepdim=True).clamp_min(1.0)
        h = self.emb(x, proto)
        for layer, norm in zip(self.layers, self.norms):
            neigh = (A @ h) / deg
            h = norm(h + self.drop(torch.relu(layer["self"](h) + layer["neigh"](neigh))))
        return self.out(torch.cat([h[:, -1], h.mean(dim=1)], dim=-1))


class TFSAGELite(WindowGraphSAGE):
    """WindowGraphSAGE with trust filtering of edges (simplified TF-SAGE)."""

    def __init__(self, n_features, hierarchy, cfg_model):
        super().__init__(n_features, hierarchy, cfg_model)
        self.trust_threshold = cfg_model.get("trust_threshold", 0.5)


MODELS = {
    "bilstm_dnn": BiLSTMDNN, "transformer": TransformerSeq, "tfsage_lite": TFSAGELite,
    "graphsage": WindowGraphSAGE, "trace": TRACE, "trace_moe": TRACEMoE,
}
ROW_LEVEL = {k for k, v in MODELS.items() if v.row_level}
