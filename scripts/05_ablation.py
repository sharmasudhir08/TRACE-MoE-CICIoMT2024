"""Step 5. TRACE-MoE ablation (hypothesis H4) and window length, official split.

Each variant in ablation.variants applies its config overrides to TRACE-MoE (e.g. dense_expert: one expert
of the same size, no routing). Window lengths in ablation.window_lengths retrain the full model with another T.

Results: results/ablation/<model>_official_T<T>_<variant>_seed<k>/   (window runs use the tag "window")
Re-written on 2026-10-29: the original script is not in the public repository.
"""
import copy

from _common import setup
from trace_iomt.config import apply_overrides
from trace_iomt.data.windows import build_windowed
from trace_iomt.experiment import audit_drop_features, load_prepared, run_deep, run_done, run_name


def add(p):
    p.add_argument("--variants", nargs="+", default=None, help="subset of ablation.variants (default: all)")
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--window-lengths", nargs="+", type=int, default=None)
    p.add_argument("--skip-windows", action="store_true", help="only the variants, no window-length runs")


def main():
    args, cfg, log = setup("TRACE-MoE ablation", "05_ablation", add)
    ab = cfg["ablation"]
    variants = args.variants or list(ab["variants"])
    unknown = [v for v in variants if v not in ab["variants"]]
    if unknown:
        raise SystemExit(f"Unknown ablation variants {unknown}; defined: {list(ab['variants'])}")
    seeds = args.seeds or cfg["evaluation"]["seeds"]
    T0 = cfg["windows"]["length"]
    lengths = [] if args.skip_windows else [T for T in (args.window_lengths or ab["window_lengths"]) if T != T0]
    base = {**cfg, "model": {**cfg["model"], "name": ab["model"]}}
    df = load_prepared(cfg)
    for seed in seeds:
        cs = {**base, "train": {**base["train"], "seed": seed}}
        todo = []
        for v in variants:
            cv = apply_overrides(copy.deepcopy(base), ab["variants"][v])
            if not run_done(cfg, "ablation", run_name(cv["model"]["name"], "official", T0, v, seed)):
                todo.append(v)
        if todo:
            wd = build_windowed(df, cs, regime="official", T=T0, seed=seed, logger=log,
                                drop_features=audit_drop_features(cfg))
        for v in todo:
            cv = apply_overrides(copy.deepcopy(base), ab["variants"][v])
            log.info(f"[ablation] variant {v} seed {seed}: overrides {ab['variants'][v]}")
            run_deep(df, cv, cv["model"]["name"], "official", seed, "ablation", log, T=T0, tag=v, wd=wd)
        for T in lengths:
            if run_done(cfg, "ablation", run_name(ab["model"], "official", T, "window", seed)):
                continue
            wdT = build_windowed(df, cs, regime="official", T=T, seed=seed, logger=log,
                                 drop_features=audit_drop_features(cfg))
            log.info(f"[ablation] window length T={T} seed {seed}")
            run_deep(df, base, ab["model"], "official", seed, "ablation", log, T=T, tag="window", wd=wdT)


if __name__ == "__main__":
    main()
