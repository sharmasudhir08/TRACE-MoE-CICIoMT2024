"""Step 3. Random Forest baseline (the tree-ensemble winner in published CICIoMT2024 papers) on single flows,
under each split regime and seed. Its abstain report uses the same calibrated reject rule as the neural models.

Results: results/main/<model>_<regime>_T1_seed<k>/
"""
from _common import setup
from trace_iomt.data.windows import build_windowed
from trace_iomt.experiment import audit_drop_features, load_prepared, run_classical, run_done, run_name


def add(p):
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--regimes", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)


def main():
    args, cfg, log = setup("Random Forest baseline", "03_train_rf", add)
    models = args.models or cfg["classical"]["models"]
    regimes = args.regimes or cfg["main"]["regimes"]
    seeds = args.seeds or cfg["evaluation"]["seeds"]
    df = load_prepared(cfg)
    for regime in regimes:
        for seed in seeds:
            todo = [m for m in models if not run_done(cfg, "main", run_name(m, regime, 1, "", seed))]
            if not todo:
                log.info(f"skip {regime} seed {seed}: {models} already done")
                continue
            wd = build_windowed(df, cfg, regime=regime, T=1, seed=seed, logger=log,
                                drop_features=audit_drop_features(cfg))
            for m in todo:
                try:
                    run_classical(df, cfg, m, regime, seed, "main", log, wd=wd)
                except ImportError as e:
                    log.warning(f"Skipping {m}: {e}")


if __name__ == "__main__":
    main()
