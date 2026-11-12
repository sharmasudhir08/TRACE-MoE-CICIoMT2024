"""Step 18 (twin study; added 2026-11-05). Poisoning test of the synchronised twin (TWIN_PROTOCOL.md §9).

For every twinned device that receives an attack in the training captures, a slow attacker drifts the device's
profile toward that attack at the fastest rate the guard allows (frozen η, δ and budget of step 16), for 1 h and
24 h of simulated time. The V3 model of the seed then scores the device's attack test windows with the drifted
profile. Needs results/twin/sync_params.json and the V3 model of --seed.

Writes results/twin/poison_seed<k>.json and results/tables/twin_poison.csv.

  python scripts/18_twin_poison.py --config configs/twin.yaml --seed 42 --set train.device=mps
"""
from pathlib import Path

import numpy as np
import pandas as pd

from _common import setup
from trace_iomt import twin_poison
from trace_iomt.compare import write_table
from trace_iomt.twin_study import EXPERIMENT, SeedData, load_v3
from trace_iomt.utils import load_json, resolve_device, save_json

HOURS = (1, 24)
GUARD_LIMIT_POINTS = 5.0        # TWIN_PROTOCOL.md §9: the guard holds if detection drops less than this at 24 h


def add(p):
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-windows", type=int, default=20000, help="attack test windows scored per device")


def device_windows(wd, device, attack_class, max_windows, seed):
    """Attack test windows of the device: its windows of the attack class, else of any attack class."""
    starts = wd.starts["test"]
    last = starts + wd.T - 1
    dev = wd.devices[wd.device_id[last]] == device
    y19, y2 = wd.y["label19"][last], wd.y["label2"][last]
    chosen = dev & (y19 == wd.hierarchy.index("label19")[attack_class])
    scope = "same class"
    if not chosen.any():
        chosen, scope = dev & (y2 == wd.hierarchy.index("label2")["Attack"]), "any attack"
    st = starts[chosen]
    if len(st) > max_windows:
        st = np.sort(np.random.default_rng(seed).choice(st, max_windows, replace=False))
    return st, scope


def test_device(data, model, device_name, params, args, cfg, torch_device):
    target = twin_poison.attack_profile(data.rows, device_name)
    if target is None:
        return None
    cls, mu_attack, n_rows = target
    key, state = twin_poison.poisoned_key(data.profiles, device_name)
    snaps, counts = twin_poison.simulate(data.profiles, device_name, state, mu_attack, params["eta"],
                                         params["delta"], params["budget"], HOURS, data.wd.T)
    starts, scope = device_windows(data.wd, device_name, cls, args.max_windows, args.seed)
    row = {"device": device_name, "profile": f"{key[0]}|{key[1]}", "attack_class": cls, "attack_rows": n_rows,
           "test_windows": int(len(starts)), "test_scope": scope, **params, "updates": counts.get("updated", 0)}
    if not len(starts):
        return row
    det = lambda prof: twin_poison.attack_detection(model, data.wd, starts, prof, torch_device,
                                                    cfg["twin"]["stream_batch"], cfg["train"]["amp"])
    row["detection_static"] = det(data.profiles)
    for h, (prof, d) in snaps.items():
        row[f"drift_sigma_{h:g}h"] = d["mean_sigma"]
        row[f"share_to_attack_{h:g}h"] = d["share_to_attack"]
        row[f"detection_{h:g}h"] = det(prof)
        row[f"drop_points_{h:g}h"] = 100 * (row["detection_static"] - row[f"detection_{h:g}h"])
    return row


def main():
    args, cfg, log = setup("Twin poisoning test", "18_twin_poison", add)
    res = Path(cfg["paths"]["results_dir"])
    path = res / EXPERIMENT / "sync_params.json"
    if not path.exists():
        raise SystemExit(f"[TWIN_NO_SYNC_PARAMS] {path} is missing: run step 16 --stage tune first (the test uses "
                         f"the frozen η, δ and budget).")
    params = {k: load_json(path)[k] for k in ("eta", "delta", "budget")}
    torch_device = resolve_device(cfg["train"]["device"])
    data = SeedData(cfg, args.seed, log)
    model, v3_dir = load_v3(cfg, args.seed, data.wd, torch_device)
    rows = []
    for name in sorted(data.profiles.twinned):
        row = test_device(data, model, name, params, args, cfg, torch_device)
        if row is None:
            log.info(f"{name}: no attack rows with attacker packets in training, skipped")
            continue
        rows.append(row)
        log.info(f"{name}: {row['attack_class']}, detection static {row.get('detection_static', float('nan')):.4f} "
                 f"-> 24 h {row.get('detection_24h', float('nan')):.4f}, drift {row.get('drift_sigma_24h', 0):.2f}σ "
                 f"({row['updates']} updates)")
    df = pd.DataFrame(rows)
    drops = df.get("drop_points_24h", pd.Series(dtype=float)).dropna()
    summary = {"seed": args.seed, "model": str(v3_dir), "sync": params, "hours": list(HOURS),
               "mean_drop_points_24h": float(drops.mean()) if len(drops) else None,
               "max_drop_points_24h": float(drops.max()) if len(drops) else None,
               "guard_holds": bool(len(drops) and drops.max() < GUARD_LIMIT_POINTS), "devices": rows}
    save_json(summary, res / EXPERIMENT / f"poison_seed{args.seed}.json")
    write_table(df.round(4), res / "tables", "twin_poison", "Poisoning test of the synchronised twin")
    log.info(f"mean drop after 24 h {summary['mean_drop_points_24h']} points, max {summary['max_drop_points_24h']}; "
             f"guard holds (< {GUARD_LIMIT_POINTS} points): {summary['guard_holds']}")


if __name__ == "__main__":
    main()
