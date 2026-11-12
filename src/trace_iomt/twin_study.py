"""Shared pieces of the twin-study steps 16-19 (added 2026-11-05): variants, run names, one seed's data, the V3
model. Paths in the twin block of the config are relative to the project folder."""
from pathlib import Path

import torch

from . import twin_data
from .experiment import latest_run_dir, run_name
from .models import build_model
from .utils import set_seed

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "twin"
VARIANTS = {"V0": (None, "device_state"), "V1": ("delta", "fleet"), "V2": ("delta", "device"),
            "V3": ("delta_state", "device_state")}           # variant -> (model.twin.inputs, twin mode)


def project_path(p):
    return Path(p) if Path(p).is_absolute() else ROOT / p


def deep_name(cfg, variant, seed):
    return run_name("trace_moe", "official", cfg["windows"]["length"], variant, seed)


def model_cfg(cfg, variant):
    inputs = VARIANTS["V3" if variant == "V4" else variant][0]
    tw = {"inputs": inputs, "n_inputs": twin_data.TWIN_INPUT_DIM, "hidden": cfg["twin"]["mlp_hidden"]}
    return {**cfg, "model": {**cfg["model"], "name": "trace_moe", "twin": tw}}


class SeedData:
    """Device rows, fitted twin and windows of one seed (the cap picks a seed-dependent block per segment)."""

    def __init__(self, cfg, seed, log):
        tc = cfg["twin"]
        self.cfg = {**cfg, "train": {**cfg["train"], "seed": seed}}
        rows_dir, devices = project_path(tc["rows_dir"]), project_path(tc["devices"])
        set_seed(seed)
        self.rows = twin_data.apply_labels(twin_data.load_attack_rows(rows_dir, tc["cap"], seed, log), tc["labels"])
        profiling = twin_data.load_profiling_rows(rows_dir)
        val_tail = float(cfg["data"].get("val_tail_fraction", 0.15))
        self.profiles, self.estimator, self.fit_report = twin_data.fit_twin(
            profiling, self.rows, twin_data.twinned_devices(devices), val_tail, tc["min_support"],
            tc["shrinkage"], seed, log)
        self.wd = twin_data.build_twin_windowed(self.cfg, self.rows, self.profiles, self.estimator,
                                                "device_state", seed, log)
        self.mode = "device_state"

    def windows(self, mode):
        if mode != self.mode:
            twin_data.attach_twin(self.wd, self.rows, self.profiles, self.estimator, mode, self.cfg["windows"]["length"])
            self.mode = mode
        self.wd.report["twin"]["fit"] = self.fit_report
        return self.wd


def load_v3(cfg, seed, wd, device):
    d = latest_run_dir(cfg, EXPERIMENT, deep_name(cfg, "V3", seed))
    if d is None or not (d / "model.pt").exists():
        raise SystemExit(f"[TWIN_NO_V3_MODEL] V3 seed {seed} has no model.pt in results/{EXPERIMENT}/. Steps 16 (V4) "
                         f"and 18 use the V3 model: run step 16 --stage train --variants V3 --seeds {seed} first.")
    model = build_model("trace_moe", len(wd.features), wd.hierarchy, model_cfg(cfg, "V3")["model"])
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    return model.to(device), d
