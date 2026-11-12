"""Step 4. TRACE-MoE and the neural baselines (BiLSTM-DNN, Transformer, TF-SAGE-lite) under each split regime
and seed (main comparison table). Each run also stores the abstain report and, for TRACE-MoE, the expert routing.

Results: results/main/<model>_<regime>_T<T>_seed<k>/  (metrics.json, history.csv, predictions.npz, model.pt)
All models with the same regime/seed are scored on the same test windows.
"""
from _common import setup
from trace_iomt.data.windows import build_windowed
from trace_iomt.experiment import audit_drop_features, load_prepared, run_deep, run_done, run_name


def add(p):
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--regimes", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--T", type=int, default=None, help="window length (default windows.length)")


def main():
    args, cfg, log = setup("Neural models", "04_train_deep", add)
    models = args.models or cfg["main"]["deep_models"]
    regimes = args.regimes or cfg["main"]["regimes"]
    seeds = args.seeds or cfg["evaluation"]["seeds"]
    T = args.T or cfg["windows"]["length"]
    df = load_prepared(cfg)
    for regime in regimes:
        for seed in seeds:
            todo = [m for m in models if not run_done(cfg, "main", run_name(m, regime, T, "", seed))]
            if not todo:
                log.info(f"skip {regime} seed {seed}: {models} already done")
                continue
            wd = build_windowed(df, {**cfg, "train": {**cfg["train"], "seed": seed}}, regime=regime, T=T,
                                seed=seed, logger=log, drop_features=audit_drop_features(cfg))
            for m in todo:
                run_deep(df, cfg, m, regime, seed, "main", log, T=T, wd=wd)


if __name__ == "__main__":
    main()
