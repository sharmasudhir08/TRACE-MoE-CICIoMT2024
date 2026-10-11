"""Fast checks (about a minute on CPU):  python -m pytest -q tests"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_iomt.config import load_config  # noqa: E402
from trace_iomt.data.labels import Hierarchy, ciciomt_category, ciciomt_label_from_filename  # noqa: E402
from trace_iomt.data.preprocess import protocol_tokens  # noqa: E402
from trace_iomt.data.windows import build_windowed  # noqa: E402
from trace_iomt.metrics import ood_metrics, recall_at_fpr  # noqa: E402
from trace_iomt.models.factory import build_model  # noqa: E402
from trace_iomt.models.networks import MODELS  # noqa: E402


def test_labels_from_filenames():
    assert ciciomt_label_from_filename("TCP_IP-DDoS-ICMP1_train.pcap.csv") == ("TCP_IP-DDoS-ICMP", "train")
    assert ciciomt_label_from_filename("MQTT-Malformed_Data_test.pcap.csv") == ("MQTT-Malformed_Data", "test")
    assert ciciomt_category("MQTT-DDoS-Connect_Flood") == "MQTT"
    assert ciciomt_category("TCP_IP-DoS-SYN") == "DoS"
    assert ciciomt_category("TCP_IP-DDoS-SYN") == "DDoS"
    assert ciciomt_category("ARP_Spoofing") == "Spoofing"


def test_protocol_tokens():
    assert protocol_tokens(np.array([6, 17, 1, 0, 8.4])).tolist() == [2, 3, 1, 0, 4]


def test_metrics():
    y = np.r_[np.zeros(100), np.ones(100)]
    s = np.r_[np.zeros(100), np.ones(100)]
    assert recall_at_fpr(y, s, 0.01) == 1.0
    assert ood_metrics(np.ones(50), np.zeros(50))["auroc"] == 1.0


def _toy_frame():
    rng = np.random.default_rng(0)
    rows = []
    for lab, cat, split in [("Benign", "Benign", "train"), ("Benign", "Benign", "test"),
                            ("TCP_IP-DoS-SYN", "DoS", "train"), ("TCP_IP-DoS-SYN", "DoS", "test")]:
        n = 120
        d = pd.DataFrame(rng.normal(cat == "DoS", 1, (n, 4)), columns=["a", "b", "c", "Protocol Type"])
        d["label19"], d["label6"], d["label2"] = lab, cat, "Benign" if cat == "Benign" else "Attack"
        d["split"], d["source_file"], d["row_in_file"], d["dataset"] = split, f"{lab}_{split}", np.arange(n), "toy"
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def test_windows_never_cross_splits_and_models_run():
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    df = _toy_frame()
    wd = build_windowed(df, cfg, regime="official", T=8)
    seg_of_row = df.loc[wd.row_index, "split"].to_numpy()
    for s in ("train", "test"):
        st = wd.starts[s]
        idx = st[:, None] + np.arange(8)
        assert (df.loc[wd.row_index[idx.ravel()], "source_file"].to_numpy().reshape(idx.shape)
                == df.loc[wd.row_index[idx[:, :1].ravel()], "source_file"].to_numpy()[:, None]).all()
    assert set(seg_of_row[wd.starts["test"]]) == {"test"}
    x, p, y = wd.gather(wd.starts["train"][:4])
    for name in MODELS:
        m = build_model(name, x.shape[-1], wd.hierarchy, cfg["model"])
        logits, h = m(torch.from_numpy(x), torch.from_numpy(p))
        assert logits["label19"].shape == (4, wd.hierarchy.n("label19"))


def test_synthetic_generator_runs(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "00_make_synthetic.py"), "--out", str(tmp_path),
                        "--train-rows", "60", "--test-rows", "20"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert len(list(tmp_path.rglob("*.csv"))) > 19


def test_moe_routing_modes_and_balance_loss():
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    wd = build_windowed(_toy_frame(), cfg, regime="official", T=8)
    x, p, _ = wd.gather(wd.starts["train"][:16])
    x, p = torch.from_numpy(x), torch.from_numpy(p)
    sizes = {}
    for routing in ("learned", "protocol", "dense"):
        mc = {**cfg["model"], "moe": {**cfg["model"]["moe"], "routing": routing}}
        m = build_model("trace_moe", x.shape[-1], wd.hierarchy, mc)
        sizes[routing] = sum(q.numel() for q in m.experts.parameters())
        m.train()
        logits, _ = m(x, p)
        assert logits["label19"].shape[0] == 16 and m.last_route.shape == (16,)
        assert int(m.last_route.max()) < m.n_experts
        if routing == "learned":
            assert m.aux_loss is not None and float(m.aux_loss.detach()) > 0
            logits["label19"].sum().backward()                               # router receives gradients
            assert m.router.weight.grad is not None
    # the dense ablation keeps the expert parameter count of the routed model (same size, no routing);
    # only the per-expert LayerNorm and output biases differ
    assert abs(sizes["dense"] - sizes["learned"]) / sizes["learned"] < 0.03


def test_abstain_layer():
    from trace_iomt.abstain import fit_temperature, risk_coverage, selective_report
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600)
    logits = rng.normal(0, 1, (600, 3)); logits[np.arange(600), y] += 2.0
    proba = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    T = fit_temperature(proba[:300], y[:300])
    assert 0.25 <= T <= 8
    aurc, curve = risk_coverage(proba.max(1), proba.argmax(1) == y)
    assert 0 <= aurc <= 1 and len(curve["coverage"]) == 20
    rep = selective_report(proba[:300], y[:300], proba[300:], y[300:], 0.9)
    assert 0.8 <= rep["coverage"] <= 1.0
    assert rep["selective_risk"] <= rep["risk_full"] + 1e-9     # abstaining on low confidence never hurts here
