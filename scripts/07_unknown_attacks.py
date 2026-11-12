"""Step 7. Unknown-attack detection (hypothesis H8): leave one attack class out of training.

For each class in zero_day.classes and each model in zero_day.models, the model is trained on the official
split without that class (regime loao). On the test set, the held-out windows must be told apart from the
known ones by the energy (abstain) score, compared with plain softmax confidence.

Results: results/zero_day/<model>_loao_T<T>_<class>_seed<k>/  with an "ood" block in metrics.json
Re-written on 2026-10-29: the original script is not in the public repository.
"""
import json

import numpy as np
from sklearn.metrics import f1_score

from _common import setup
from trace_iomt.data.windows import build_windowed
from trace_iomt.experiment import audit_drop_features, load_prepared, run_deep, run_done, run_name
from trace_iomt.metrics import ood_metrics
from trace_iomt.train import predict
from trace_iomt.utils import resolve_device, save_json


def add(p):
    p.add_argument("--classes", nargs="+", default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)


def ood_report(model, wd, held_out, device, target_coverage):
    k = wd.hierarchy.index("label19")[held_out]
    attack = wd.hierarchy.index("label2")["Attack"]
    test = predict(model, wd, wd.starts["test"], device)
    y19 = wd.labels("test", "label19")
    unknown = y19 == k
    msp = test["proba"]["label19"].max(1)
    energy = test["energy"]
    val_energy = predict(model, wd, wd.starts["val"], device)["energy"] if len(wd.starts["val"]) else energy[~unknown]
    thr = float(np.quantile(val_energy, 1 - target_coverage))      # keeps target_coverage of known windows
    flagged = energy < thr
    as_attack = test["proba"]["label2"].argmax(1) == attack
    known_labels = [i for i in np.unique(y19[~unknown])]
    pred19 = test["proba"]["label19"].argmax(1)
    return {"held_out_class": held_out, "threshold_energy": thr, "n_unknown": int(unknown.sum()),
            "energy": ood_metrics(energy[~unknown], energy[unknown]),
            "msp": ood_metrics(msp[~unknown], msp[unknown]),
            "held_out_detected_as_attack": float(as_attack[unknown].mean()) if unknown.any() else float("nan"),
            "held_out_flagged_unknown": float(flagged[unknown].mean()) if unknown.any() else float("nan"),
            "held_out_caught_any": float((flagged | as_attack)[unknown].mean()) if unknown.any() else float("nan"),
            "known_false_alarm_rate": float(flagged[~unknown].mean()),
            "known_classes_macro_f1": float(f1_score(y19[~unknown], pred19[~unknown], labels=known_labels,
                                                     average="macro", zero_division=0))}


def main():
    args, cfg, log = setup("Unknown-attack detection", "07_unknown_attacks", add)
    zd = cfg["zero_day"]
    classes, models = args.classes or zd["classes"], args.models or zd["models"]
    seeds = args.seeds or cfg["evaluation"]["seeds"]
    T = cfg["windows"]["length"]
    device = resolve_device(cfg["train"]["device"])
    df = load_prepared(cfg)
    for held_out in classes:
        cl = {**cfg, "split": {**cfg["split"], "regime": "loao", "loao_class": held_out}}
        for seed in seeds:
            cs = {**cl, "train": {**cl["train"], "seed": seed}}
            todo = [m for m in models
                    if not run_done(cfg, "zero_day", run_name(m, "loao", T, held_out, seed), required_key="ood")]
            if not todo:
                log.info(f"skip {held_out} seed {seed}: {models} already done")
                continue
            wd = build_windowed(df, cs, regime="loao", T=T, seed=seed, logger=log,
                                drop_features=audit_drop_features(cfg))
            for m in todo:
                d, _, model, _ = run_deep(df, cl, m, "loao", seed, "zero_day", log, T=T, tag=held_out, wd=wd)
                ood = ood_report(model, wd, held_out, device, cfg["abstain"]["target_coverage"])
                metrics = json.loads((d / "metrics.json").read_text())
                metrics["ood"] = ood
                save_json(metrics, d / "metrics.json")
                log.info(f"[zero-day] {m} without {held_out} seed {seed}: energy AUROC {ood['energy']['auroc']:.4f} "
                         f"vs softmax {ood['msp']['auroc']:.4f}; flagged unknown {ood['held_out_flagged_unknown']:.3f}, "
                         f"caught at all {ood['held_out_caught_any']:.3f}")


if __name__ == "__main__":
    main()
