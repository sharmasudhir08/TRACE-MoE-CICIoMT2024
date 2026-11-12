"""Step 19 (twin study; added 2026-11-05). Twin cost on one CPU thread (TWIN_PROTOCOL.md §10, paper C6).

Measures, with the fitted twin of one seed:
  memory    serialised size of each twinned device's profiles (per added device), the fleet profile and the
            state estimator (shared by all devices)
  time      per window at batch 1: Δ (Eq. 1), the novelty flags, the state estimate, one synchronisation update,
            and the detector without and with the twin input (V0 and V3 architectures; weights do not change the
            time, so untrained models are used); end to end = Δ + flags + state + V3 detector
Uses results/twin/twin_model.pkl of step 15 when it exists, else fits the twin (about 2 minutes). Run it when
nothing else is running on the machine, so the timings are not disturbed.

Writes results/twin/cost.json and results/tables/twin_cost.csv.

  python scripts/19_twin_cost.py --config configs/twin.yaml
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")            # one CPU thread, set before numpy and torch load
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import copy  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from _common import setup  # noqa: E402
from trace_iomt import twin, twin_data  # noqa: E402
from trace_iomt.compare import write_table  # noqa: E402
from trace_iomt.data.labels import Hierarchy  # noqa: E402
from trace_iomt.models import build_model  # noqa: E402
from trace_iomt.pcapfeat import FEATURES  # noqa: E402
from trace_iomt.twin_study import EXPERIMENT, model_cfg, project_path  # noqa: E402
from trace_iomt.utils import save_json  # noqa: E402

NF = len(FEATURES)


def add(p):
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=2000)


def median_ms(fn, repeats, warmup=50):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))


def load_twin(cfg, seed, log):
    pkl = Path(cfg["paths"]["results_dir"]) / EXPERIMENT / "twin_model.pkl"
    if pkl.exists():
        with open(pkl, "rb") as fh:
            m = pickle.load(fh)
        log.info(f"twin from {pkl}")
        return m["profiles"], m["estimator"]
    t = cfg["twin"]
    rows_dir = project_path(t["rows_dir"])
    rows = twin_data.apply_labels(twin_data.load_attack_rows(rows_dir, t["cap"], seed, log), t["labels"])
    profiles, estimator, _ = twin_data.fit_twin(
        twin_data.load_profiling_rows(rows_dir), rows, twin_data.twinned_devices(project_path(t["devices"])),
        float(cfg["data"].get("val_tail_fraction", 0.15)), t["min_support"], t["shrinkage"], seed, log)
    return profiles, estimator


def memory(profiles, estimator):
    size = lambda obj: len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    per_device = {d: size({k: v for k, v in profiles.profiles.items() if k[0] == d}) for d in sorted(profiles.twinned)
                  if any(k[0] == d for k in profiles.profiles)}
    fleet = size(profiles.profiles[(twin.FLEET, None)])
    est = size(estimator)
    return {"bytes_per_device": per_device, "mean_bytes_per_added_device": float(np.mean(list(per_device.values()))),
            "fleet_profile_bytes": fleet, "state_estimator_bytes": est,
            "total_bytes": int(sum(per_device.values()) + fleet + est), "n_twinned_devices": len(per_device)}


def detector_ms(cfg, variant, repeats, T):
    h = Hierarchy(["Benign"])
    model = build_model("trace_moe", NF, h, model_cfg(cfg, variant)["model"]).eval()
    x, p = torch.randn(1, T, NF), torch.zeros(1, T, dtype=torch.long)
    tw = (torch.randn(1, twin_data.TWIN_INPUT_DIM), torch.tensor([2])) if variant != "V0" else None
    with torch.no_grad():
        return median_ms(lambda: model(x, p, twin=tw) if tw else model(x, p), repeats // 4)


def timings(cfg, profiles, estimator, repeats):
    T = cfg["windows"]["length"]
    device = sorted(profiles.twinned)[0]
    key, _ = profiles.lookup_key(device, "Idle")
    prof = profiles.profiles[key]
    x = np.log1p(np.abs(np.random.default_rng(0).standard_normal((T, NF)))).astype(np.float32)
    peers, topics = ["10.0.0.1;10.0.0.2"] * T, ["sensor/spo2"] * T
    sync = twin.Synchroniser(copy.deepcopy(profiles), eta=0.01, delta=1e9, budget_per_hour=10 ** 9)
    clock = iter(range(10 ** 9))
    out = {"delta_ms": median_ms(lambda: twin.discrepancy(x, prof), repeats),
           "novelty_ms": median_ms(lambda: twin.novelty(peers, topics, prof), repeats),
           "state_estimate_ms": median_ms(lambda: estimator.predict_rows(x), repeats // 4),
           "sync_update_ms": median_ms(lambda: sync.maybe_update(device, key[1], x, np.zeros(NF), True, next(clock)),
                                       repeats),
           "detector_no_twin_ms": detector_ms(cfg, "V0", repeats, T),
           "detector_twin_ms": detector_ms(cfg, "V3", repeats, T)}
    out["end_to_end_ms"] = out["delta_ms"] + out["novelty_ms"] + out["state_estimate_ms"] + out["detector_twin_ms"]
    return out


def main():
    args, cfg, log = setup("Twin cost (memory, time per window)", "19_twin_cost", add)
    torch.set_num_threads(1)
    profiles, estimator = load_twin(cfg, args.seed, log)
    mem = memory(profiles, estimator)
    t = timings(cfg, profiles, estimator, args.repeats)
    res = Path(cfg["paths"]["results_dir"])
    save_json({"memory": mem, "time_ms_per_window": t, "threads": 1, "torch_threads": torch.get_num_threads()},
              res / EXPERIMENT / "cost.json")
    rows = [{"item": f"memory: {k}", "value": v} for k, v in mem.items() if k != "bytes_per_device"]
    rows += [{"item": f"time: {k}", "value": round(v, 4)} for k, v in t.items()]
    write_table(pd.DataFrame(rows), res / "tables", "twin_cost", "Twin cost on one CPU thread")
    log.info(f"memory per added device {mem['mean_bytes_per_added_device']:.0f} B, total {mem['total_bytes']:,} B; "
             f"end to end {t['end_to_end_ms']:.3f} ms per window ({t})")


if __name__ == "__main__":
    main()
