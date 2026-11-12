"""Twin context in TRACE-MoE, the synchronised stream and step 16 end to end, on synthetic device rows."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from test_twin_data import cfg, device_rows, write
from trace_iomt import twin, twin_data, twin_sync
from trace_iomt.config import load_config
from trace_iomt.models.factory import build_model, count_parameters
from trace_iomt.pcapfeat import FEATURES
from trace_iomt.train import predict

ROOT = Path(__file__).resolve().parents[1]
NF = len(FEATURES)


@pytest.fixture
def tree(tmp_path):
    """Benign and one attack class in train and test, one twinned device plus an untwinned host, two states."""
    root = tmp_path / "device"
    write(root, "train", "Benign_train", [device_rows(300, "cam", 5, "train", "Benign"),
                                           device_rows(200, "mac:20:7b", 20, "train", "Benign", seed=1)])
    write(root, "train", "TCP_IP-DDoS-ICMP1_train", [device_rows(300, "cam", 40, "train", "TCP_IP-DDoS-ICMP", attacker_every=2, seed=2)])
    write(root, "test", "Benign_test", [device_rows(120, "cam", 5, "test", "Benign", seed=5)])
    write(root, "test", "TCP_IP-DDoS-ICMP1_test", [device_rows(120, "cam", 40, "test", "TCP_IP-DDoS-ICMP", attacker_every=2, seed=3,
                                                               peers="10.0.0.1;10.9.9.9", topics="?")])
    for state, base, seed in (("Idle", 5, 4), ("Active", 12, 6)):
        prof = device_rows(300, "cam", base, "profiling", "Benign", seed=seed)
        prof["state"] = state
        write(root, "profiling", state, [prof])
    return root


def _twin_wd(tree, mode="device_state"):
    rows = twin_data.apply_labels(twin_data.load_attack_rows(tree, cap=None), "file")
    prof, est, _ = twin_data.fit_twin(twin_data.load_profiling_rows(tree), rows, {"cam"}, 0.15, min_support=50)
    return twin_data.build_twin_windowed(cfg(), rows, prof, est, mode, seed=0), rows, prof, est


def _model_cfg(inputs):
    m = load_config(ROOT / "configs" / "smoke.yaml")["model"]
    return {**m, "twin": {"inputs": inputs, "n_inputs": twin_data.TWIN_INPUT_DIM, "hidden": 16}}


def test_window_inputs_are_log_delta_flags_level_shares_and_last_state():
    ctx = np.zeros((6, len(twin_data.CTX_NAMES)), np.float32)
    ctx[:, :NF] = np.arange(6)[:, None]                  # z of row i is i
    ctx[[1, 2], NF] = 1.0                                # new peer on rows 1 and 2
    ctx[:, -1] = [0, 0, 1, 2, 2, 2]
    state = np.array([1, 2, 3, 4, 1, 2])
    vec, s = twin_data.window_inputs(ctx, state, np.array([0, 2]), T=4)
    assert vec.shape == (2, twin_data.TWIN_INPUT_DIM)
    np.testing.assert_allclose(vec[:, 0], np.log1p([1.5, 3.5]), rtol=1e-6)
    np.testing.assert_allclose(vec[:, NF], [0.5, 0.25])
    np.testing.assert_allclose(vec[:, NF + 2:], [[0.5, 0.25, 0.25], [0.0, 0.25, 0.75]])
    assert s.tolist() == [4, 2]


def test_twin_context_enters_encoder_and_router():
    from trace_iomt.data.labels import Hierarchy
    h = Hierarchy(["Benign", "TCP_IP-DDoS-ICMP", "ARP_Spoofing"])
    plain = build_model("trace_moe", NF, h, _model_cfg(None))
    delta = build_model("trace_moe", NF, h, _model_cfg("delta"))
    full = build_model("trace_moe", NF, h, _model_cfg("delta_state"))
    d = plain.d
    assert plain.router.in_features == d + d // 4 and full.router.in_features == d + d // 2
    assert count_parameters(plain) < count_parameters(delta) < count_parameters(full)
    x, p = torch.randn(3, 8, NF), torch.zeros(3, 8, dtype=torch.long)
    tw = (torch.randn(3, twin_data.TWIN_INPUT_DIM), torch.tensor([0, 2, 4]))
    full.eval()
    logits, _ = full(x, p, twin=tw)
    assert logits["label19"].shape == (3, h.n("label19"))
    other, _ = full(x, p, twin=(tw[0], torch.tensor([1, 1, 1])))
    assert not torch.allclose(logits["label19"], other["label19"])     # the state changes the output
    with pytest.raises(ValueError):
        full(x, p)


def test_attach_twin_switches_mode_without_rebuilding_windows(tree):
    wd, rows, prof, est = _twin_wd(tree)
    starts = {k: v.copy() for k, v in wd.starts.items()}
    assert set(wd.state_code) <= {1, 2, 3, 4} and len(wd.twin_keys) == wd.key_id.max() + 1
    twin_data.attach_twin(wd, rows, prof, est, "fleet", 4)
    assert set(wd.ctx[:, -1]) == {2.0} and set(wd.state_code) == {0}
    assert all(np.array_equal(starts[k], wd.starts[k]) for k in starts)
    np.testing.assert_allclose(wd.xlog, twin.feature_matrix(rows.reset_index(drop=True).iloc[wd.row_index]), rtol=1e-6)


def test_single_packet_rows_do_not_put_nan_into_the_model_input(tree):
    rows = twin_data.apply_labels(twin_data.load_attack_rows(tree, cap=None), "file")
    rows.loc[rows.index[::7], ["Std", "Variance"]] = np.nan          # pcapfeat writes NaN for one-packet rows
    prof, est, _ = twin_data.fit_twin(twin_data.load_profiling_rows(tree), rows, {"cam"}, 0.15, min_support=50)
    wd = twin_data.build_twin_windowed(cfg(), rows, prof, est, "device_state", seed=0)
    assert np.isfinite(wd.X).all() and np.isfinite(wd.ctx).all()


def test_stream_without_updates_matches_static_prediction(tree):
    wd, _, prof, _ = _twin_wd(tree)
    model = build_model("trace_moe", len(wd.features), wd.hierarchy, _model_cfg("delta_state"))
    dev = torch.device("cpu")
    static = predict(model, wd, wd.starts["test"], dev, amp=False, return_logits=True)
    res = twin_sync.stream(model, wd, "test", prof, eta=0.5, delta=1e9, budget=0, rule=(1.0, -np.inf),
                           benign_index=0, device=dev, batch_size=16, amp=False)
    np.testing.assert_allclose(res["logits19"], static["logits19"], rtol=1e-4, atol=1e-5)
    assert res["sync_counts"].get("updated", 0) == 0


def test_stream_updates_a_copy_of_the_profiles(tree):
    wd, _, prof, _ = _twin_wd(tree)
    before = copy.deepcopy(prof)
    model = build_model("trace_moe", len(wd.features), wd.hierarchy, _model_cfg("delta_state"))
    dev = torch.device("cpu")
    top = np.bincount(predict(model, wd, wd.starts["test"], dev, amp=False)["proba"]["label19"].argmax(1)).argmax()
    res = twin_sync.stream(model, wd, "test", prof, eta=0.5, delta=1e9, budget=10_000, rule=(1.0, -np.inf),
                           benign_index=int(top), device=dev, batch_size=16, amp=False)
    assert res["sync_counts"]["updated"] > 0
    assert len(res["delta_mean"]) == len(wd.starts["test"])
    for k, p in prof.profiles.items():
        np.testing.assert_array_equal(p.mu, before.profiles[k].mu)          # the fitted twin is unchanged


def test_poisoning_respects_the_budget_and_moves_toward_the_attack(tree):
    from trace_iomt import twin_poison
    wd, rows, prof, _ = _twin_wd(tree)
    cls, mu_attack, n = twin_poison.attack_profile(rows, "cam")
    assert cls == "TCP_IP-DDoS-ICMP" and n > 0
    key, state = twin_poison.poisoned_key(prof, "cam")
    snaps, counts = twin_poison.simulate(prof, "cam", state, mu_attack, eta=0.05, delta=5, budget=10, T=4)
    assert counts["updated"] == 240 and set(snaps) == {1, 24}                # 10 per hour for 24 h
    assert 0 < snaps[1][1]["share_to_attack"] < snaps[24][1]["share_to_attack"] <= 1
    np.testing.assert_array_equal(prof.profiles[key].mu, copy.deepcopy(prof).profiles[key].mu)
    model = build_model("trace_moe", len(wd.features), wd.hierarchy, _model_cfg("delta_state"))
    dev, starts = torch.device("cpu"), wd.starts["test"]
    attack = wd.hierarchy.index("label2")["Attack"]
    static = predict(model, wd, starts, dev, amp=False)["proba"]["label2"].argmax(1) == attack
    assert np.isclose(twin_poison.attack_detection(model, wd, starts, prof, dev, 16, False), static.mean())


def _step16(tmp_path, tree, stage, *extra):
    devices = tmp_path / "devices.json"
    devices.write_text(json.dumps({"devices": {"cam": "00:00:00:00:00:01"}}))
    cmd = [sys.executable, str(ROOT / "scripts" / "16_twin_train.py"), "--config", str(ROOT / "configs" / "twin_smoke.yaml"),
           "--stage", stage, "--set", f"twin.rows_dir={tree}", "--set", f"twin.devices={devices}",
           "--set", f"paths.results_dir={tmp_path / 'results'}", "--set", f"paths.logs_dir={tmp_path / 'logs'}",
           "--set", "train.device=cpu", "--set", "windows.length=4", *extra]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stderr + out.stdout


def test_step16_end_to_end_and_restartable(tmp_path, tree):
    res = tmp_path / "results" / "twin"
    with pytest.raises(AssertionError):
        _step16(tmp_path, tree, "sync")                                    # no V3 model and no frozen values yet
    _step16(tmp_path, tree, "train")
    for run in ("trace_moe_official_T4_V0_seed42", "trace_moe_official_T4_V1_seed42", "trace_moe_official_T4_V2_seed42",
                "trace_moe_official_T4_V3_seed42", "rf_official_T1_seed42"):
        assert (res / run / "posthoc_v2.json").exists(), run
    z = np.load(res / "trace_moe_official_T4_V3_seed42" / "predictions.npz")
    assert len(z["twin_delta_mean"]) == len(z["y_true_label19"]) and "val_t_last" in z
    assert "already done" in _step16(tmp_path, tree, "train")
    _step16(tmp_path, tree, "tune")
    params = json.loads((res / "sync_params.json").read_text())
    assert len(params["grid"]) == 2 and params["eta"] in (0.01, 0.05)
    _step16(tmp_path, tree, "sync")
    m = json.loads((res / "trace_moe_official_T4_V4_seed42" / "metrics.json").read_text())
    assert m["sync"]["delta"] == 5 and "label19" in m["test"] and m["model_from"].endswith("V3_seed42")
    assert (res / "trace_moe_official_T4_V4_seed42" / "posthoc_v2.json").exists()
    for script in ("18_twin_poison.py", "19_twin_cost.py"):
        cmd = [sys.executable, str(ROOT / "scripts" / script), "--config", str(ROOT / "configs" / "twin_smoke.yaml"),
               "--set", f"twin.rows_dir={tree}", "--set", f"twin.devices={tmp_path / 'devices.json'}",
               "--set", f"paths.results_dir={tmp_path / 'results'}", "--set", f"paths.logs_dir={tmp_path / 'logs'}",
               "--set", "windows.length=4", "--set", "train.device=cpu"] + (["--repeats", "40"] if "cost" in script else [])
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        assert out.returncode == 0, (script, out.stderr[-3000:])
    poison = json.loads((res / "poison_seed42.json").read_text())
    assert poison["devices"][0]["device"] == "cam" and "detection_24h" in poison["devices"][0]
    cost = json.loads((res / "cost.json").read_text())
    assert cost["memory"]["n_twinned_devices"] == 1 and cost["time_ms_per_window"]["end_to_end_ms"] > 0
    cmd = [sys.executable, str(ROOT / "scripts" / "17_twin_compare.py"), "--config", str(ROOT / "configs" / "twin_smoke.yaml"),
           "--set", f"twin.rows_dir={tree}", "--set", f"twin.devices={tmp_path / 'devices.json'}",
           "--set", f"paths.results_dir={tmp_path / 'results'}", "--set", f"paths.logs_dir={tmp_path / 'logs'}",
           "--set", "windows.length=4", "--resamples", "50"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    h11 = json.loads((tmp_path / "results" / "twin_h11.json").read_text())
    assert h11["verdict"] in ("supported", "partly supported", "not supported") and h11["file_bootstrap"]["n_files"] == 2
    tables = tmp_path / "results" / "tables"
    for name in ("twin_ablation", "twin_differences", "twin_wilcoxon", "twin_per_class", "twin_delta_auroc", "twin_sync",
                 "twin_time_to_detect", "twin_poison", "twin_cost"):
        assert (tables / f"{name}.csv").exists(), name
    assert len(pd.read_csv(tables / "twin_ablation.csv")) == 6
