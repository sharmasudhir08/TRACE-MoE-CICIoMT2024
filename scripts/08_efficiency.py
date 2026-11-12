"""Step 8. Model size and inference latency (hypothesis H10).

Every neural model is profiled at batch 1 and 1024 on the CPU (one thread by default, as reported in the
paper) and on the GPU when one is available. The models are freshly initialised: latency does not depend on
the trained weights.

Results: results/efficiency/efficiency.csv (+ .tex)
Re-written on 2026-10-29: the original script is not in the public repository.
"""
from pathlib import Path

import pandas as pd
import torch

from _common import setup
from trace_iomt.compare import write_table
from trace_iomt.data.labels import Hierarchy
from trace_iomt.data.loaders import feature_cols
from trace_iomt.efficiency import profile_model
from trace_iomt.experiment import audit_drop_features, load_prepared


def add(p):
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--threads", type=int, default=1, help="CPU threads for the CPU timings")


def main():
    args, cfg, log = setup("Efficiency", "08_efficiency", add)
    torch.set_num_threads(args.threads)
    models = args.models or cfg["main"]["deep_models"]
    df = load_prepared(cfg)
    n_features = len([c for c in feature_cols(df) if c not in set(audit_drop_features(cfg))])
    hierarchy = Hierarchy(df["label19"].astype(str).unique().tolist())
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + \
        (["mps"] if torch.backends.mps.is_available() else [])
    T = cfg["windows"]["length"]
    rows = []
    for m in models:
        row = profile_model(m, n_features, T, hierarchy, {**cfg["model"], "name": m}, devices)
        row["cpu_threads"] = args.threads
        log.info(row)
        rows.append(row)
    out = Path(cfg["paths"]["results_dir"]) / "efficiency"
    write_table(pd.DataFrame(rows), out, "efficiency", "Model size and inference latency per window")
    log.info(f"Efficiency written to {out}")


if __name__ == "__main__":
    main()
