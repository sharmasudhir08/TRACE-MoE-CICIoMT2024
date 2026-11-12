"""Step 6. Robustness (hypotheses H6, H7): feature-constrained PGD, black-box transfer, adversarial fine-tuning.

For every seed, the neural models trained by step 04 are reloaded (or trained if missing) and a random forest
is fitted on the last flows of the same windows, as a transfer target. On a class-balanced sample of test
windows (robustness.max_eval_windows):
  1. transfer matrix: attack windows crafted on each surrogate are given to every model, incl. the forest;
  2. robustness curve of each clean model over robustness.epsilons (the forest is scored on the same windows);
  3. adversarial fine-tuning, then the curve again (H6 compares per-class ASR at robustness.main_epsilon);
  4. the fine-tuned model is scored on the full clean test set (H7: cost in clean macro-F1).

Results: results/robustness/<model>_official_T<T>_seed<k>/   (metrics.json "test" = fine-tuned model, clean)
         results/robustness/transfer_matrix_seed<k>/transfer_matrix.json
Re-written on 2026-10-29: the original script is not in the public repository.
"""
import copy
from pathlib import Path

import torch

from _common import setup
from trace_iomt.attacks import adversarial_finetune, attack_eval_starts, robustness_curve, transfer_matrix
from trace_iomt.data.windows import build_windowed
from trace_iomt.experiment import (audit_drop_features, classical_proba, fit_classical, load_or_train_deep,
                                   load_prepared, run_dir, run_done, run_name, save_run)
from trace_iomt.train import evaluate
from trace_iomt.utils import now_iso, resolve_device, save_json, set_seed


def add(p):
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--skip-transfer", action="store_true")


def _tree_predictors(wd, cfg, seed, names):
    n19 = wd.hierarchy.n("label19")
    out = {}
    for name in names:
        clf, _, _ = fit_classical(name, wd, cfg, seed)
        out[name] = (lambda c: (lambda X: classical_proba(c, X, n19)))(clf)
    return out


def main():
    args, cfg, log = setup("Robustness", "06_robustness", add)
    rc = cfg["robustness"]
    models = args.models or rc["models"]
    seeds = args.seeds or cfg["evaluation"]["seeds"]
    T = cfg["windows"]["length"]
    device = resolve_device(cfg["train"]["device"])
    df = load_prepared(cfg)
    for seed in seeds:
        set_seed(seed)
        cs = {**cfg, "train": {**cfg["train"], "seed": seed}}
        tm_path = Path(cfg["paths"]["results_dir"]) / "robustness" / f"transfer_matrix_seed{seed}" / "transfer_matrix.json"
        todo = [m for m in models if not run_done(cfg, "robustness", run_name(m, "official", T, "", seed))]
        need_transfer = rc.get("transfer_matrix", True) and not args.skip_transfer and not tm_path.exists()
        if not todo and not need_transfer:
            log.info(f"skip seed {seed}: robustness already done")
            continue
        wd = build_windowed(df, cs, regime="official", T=T, seed=seed, logger=log,
                            drop_features=audit_drop_features(cfg))
        nets = {m: load_or_train_deep(df, cs, m, "official", seed, "main", log, T=T, wd=wd)[0] for m in models}
        trees = _tree_predictors(wd, cs, seed, rc.get("tree_models") or [])
        benign = wd.hierarchy.index("label19")["Benign"]
        starts = attack_eval_starts(wd, rc["max_eval_windows"], seed, benign)
        if need_transfer:
            rows = transfer_matrix(nets, trees, wd, cs, device, starts, log)
            tm_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(rows, tm_path)
        for name, model in ((n, nets[n]) for n in todo):
            started_at = now_iso()
            log.info(f"[{name} seed {seed}] PGD on {len(starts):,} test windows")
            clean_curve = robustness_curve(model, wd, cs, device, starts, log, extra=trees)
            tuned = adversarial_finetune(copy.deepcopy(model), wd, cs, device, log)
            log.info(f"[{name} seed {seed}] after adversarial fine-tuning")
            tuned_curve = robustness_curve(tuned, wd, cs, device, starts, log, extra=trees)
            test_m, _, _ = evaluate(tuned, wd, "test", device, cs)
            run = run_name(name, "official", T, "", seed)
            meta = dict(experiment="robustness", run=run, model=name, family="deep", regime="official", seed=seed,
                        T=T, tag="", started_at=started_at, n_eval_windows=int(len(starts)),
                        main_epsilon=rc["main_epsilon"], curves={"clean": clean_curve, "adv_finetuned": tuned_curve},
                        hierarchy=wd.hierarchy.to_dict(), data_report=wd.report)
            d = run_dir(cfg, "robustness", run)
            save_run(d, cs, meta, test_m)
            torch.save(tuned.state_dict(), d / "model.pt")
            log.info(f"[{name} seed {seed}] clean test macro-F1 after fine-tuning "
                     f"19:{test_m['label19']['macro_f1']:.4f} -> {d}")


if __name__ == "__main__":
    main()
