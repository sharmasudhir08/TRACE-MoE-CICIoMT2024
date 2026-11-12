# Re-running TRACE-MoE (reconstructed pipeline)

The public repository was missing code that every step needs. On 2026-10-29 the following were rebuilt from
the callers, `METHODOLOGY.md` and the counts recorded in `logs_v2/`:

| File | Status |
|---|---|
| `src/trace_iomt/data/` (`loaders`, `labels`, `preprocess`, `windows`) | re-implemented: `.gitignore`'s `data/` rule had excluded it |
| `scripts/05_ablation.py`, `06_robustness.py`, `07_unknown_attacks.py`, `08_efficiency.py`, `09_compare.py` | re-written |
| `scripts/10_hypothesis_verdicts.py` | replaces `10_technical_report.py` and `11_write_paper.py`: H1–H10 verdicts only, no Word report |
| `run_all.py`, `configs/rerun.yaml`, `tests/test_windows.py` | new |

Because the data package is a re-implementation, numbers will be close to, not identical with, the paper.
Step 2 below shows how close.

## 1. Set up (once)

From the repository folder:

```bash
python3.12 -m venv .venv && source .venv/bin/activate        # Windows: py -3.12 -m venv .venv; .venv\Scripts\activate
# PyTorch first, same version as tested: NVIDIA GPU -> CUDA build (see requirements-tested.txt); Mac -> plain wheel
pip install torch==2.15.0
pip install -r requirements-tested.txt                        # exact versions tested (Python 3.12.5)
python -m pytest -q tests                                     # expect 16 passed
python run_all.py --smoke --set train.device=cpu              # synthetic data, every step, a few minutes
```

The real CSVs must be at `data/raw/CICIoMT2024/WiFi_and_MQTT/attacks/CSV/{train,test}/`. If they are
elsewhere, add `--set paths.ciciomt_csv_dir=<path>` to every command below.

## 2. Prepare the data and check the reconstruction

```bash
python scripts/01_prepare.py --config configs/rerun.yaml
python scripts/02_audit.py   --config configs/rerun.yaml
python scripts/03_train_rf.py --config configs/rerun.yaml --seeds 42 --regimes official
```

Compare with what the original code logged (`logs_v2/`):

| Check | Original | Where to look |
|---|---|---|
| Rows / classes | 8,775,013 rows, 19 classes, 51 train + 21 test files | `logs_rerun/01_prepare.log` |
| Windows, official, T = 1 (seed 42) | train 3,298,594 · val 727,862 · test 1,548,977 | `logs_rerun/03_train_rf.log` |
| Duplicates removed, T = 1 | 2,788,114 · 346,261 · 65,205 | same |
| Windows, official, T = 16 (seed 42) | train 616,087 · val 59,092 · test 1,584,389 | `logs_rerun/04_train_deep.log` (the timing run in step 3) |
| Duplicates removed, T = 16 | 144,682 · 8,014 · 29,478 | same |

Windows before de-duplication (kept + removed) must match **exactly**: train 6,086,708 · val 1,074,123 ·
test 1,614,182 at T = 1. This was confirmed on 2026-10-30 for train and validation. If the totals differ,
stop and report them. The test total needs all 21 test files.

**Known deviation (duplicate removal).** The original duplicate rule is not in the repository. The closest
rule found compares scaled values rounded to 6 decimals. It removes 2,768,690 training rows (original
2,788,114, −0.7 %) and 313,345 validation rows (original 346,261, −9.5 %). Exact values, float16, other
roundings, other scaler sample sizes and counting repeats inside validation were all further off. So the
de-duplicated counts will be slightly higher than the original's, by about 0.3 % of training rows and
3 % of validation rows.

## 3. The full re-run

Two lanes run in parallel on one GPU, and every step skips runs that are already finished, so stopping and
restarting a lane costs only the run in progress:

```bash
./rerun_all.sh mps A        # terminal 1: RF + neural models (5 seeds), robustness, ablation seeds 43-44
./rerun_all.sh mps B        # terminal 2: unknown attacks, ablation seed 42
./rerun_all.sh mps final    # when both lanes are done: efficiency, tables, H1-H10 verdicts
```

| Experiment | Seeds |
|---|---|
| Main comparison (steps 03–04) | 42–46, both splits |
| H4 ablation: full vs dense expert | 42–44 |
| Other ablation variants, window lengths | 42 only (exploratory) |
| H6/H7 robustness | 42–44 |
| H8 unknown attacks: TRACE-MoE | 42–44 |
| H8 baselines (Transformer, BiLSTM-DNN) | 42 only (context) |

No hypothesis has fewer than 3 seeds. The single-seed runs are reported as exploratory.

**Time on the Mac Studio (M1 Ultra, MPS):** about 4–5 days if the two lanes share the GPU well, and if the
Mac does not stall. Exclude the repo from Spotlight and Time Machine, and turn on System Settings → Energy →
"Prevent automatic sleeping". The script calls `caffeinate` for every step.

## 4. What to send back

`results_rerun/tables/`, `results_rerun/figures/`, `results_rerun/hypotheses.json`, `results_rerun/efficiency/`
and `logs_rerun/` (zipped). The per-run folders (`results_rerun/<experiment>/<run>/`, with `predictions.npz`)
are large; keep them on that machine, but they are needed for the post-hoc analyses.

## 5. Post-hoc corrections, step 11 (added 2026-11-03; no training)

`scripts/11_posthoc.py` re-reads every finished main run. It **does not train anything**.
- **Neural models:** reloads `model.pt` and re-scores validation and test windows, keeping the float32 logits. Expect about 20–40 minutes for all 50 main runs on the Mac Studio, mostly rebuilding windows.
- **Random forest** (model not saved): uses the probabilities saved in `predictions.npz`.

It writes `<run>/posthoc_v2.json` with:
- **Corrected AURC.** Tied confidences are handled in expectation, and confidences are computed in log space (`log1p`), so they don't saturate at 1.0. Before this, 80–99% of neural-model windows were tied, and AURC depended on file order.
- **AURC and selective risk at 90% / 95% coverage** for five scores: MSP raw, MSP calibrated, margin, negative entropy, energy.
- **Hierarchy consistency** of predicted labels (19→6, 6→2, joint).
- **Binary macro-F1** at 0.5 and at the validation-tuned threshold.
- **Agreement** of the re-scored predictions with `predictions.npz`. It should be ≥ 99.9%; a lower value is logged as `[POSTHOC_PRED_MISMATCH]`.

It also writes `results_rerun/tables/class_window_counts_<regime>.csv`.

Steps 09 and 10 then use `posthoc_v2.json` automatically:
- the `abstain_*` tables and the H9 verdict use the corrected AURC;
- the original values are kept as `abstain_original` in memory, and `metrics.json` is not changed;
- new tables: `selective_scores_*`, `hierarchy_consistency_*`, `binary_tuned_*`, `noninferiority_binary_tuned`.

```bash
./rerun_all.sh mps posthoc      # step 11, then 09 and 10
```

`posthoc_v2.json` is never overwritten without `--force`, so the step can be stopped and restarted.

## 6. Twin study, step 16 (added 2026-11-05)

Needs the device rows of step 14 in `data/processed/twin/device` (already there from 2026-11-04). Settings
are in `configs/twin.yaml`; results go to `results_rerun/twin/`. Every stage skips finished runs.

| Lane | What | Runs | Time (estimate) |
|---|---|---|---|
| `check` | 1 epoch of V3 seed 42, then tune and sync, into `results/_twin_check` | 1 | 30–45 min |
| `A` | V3 then V0, seeds 42–46 | 10 | about 17 h |
| `B` | V1 and V2 on seeds 42–44, random forest on 42–46 | 6 + 5 | about 11 h |
| `final` | choose η, δ, budget on validation of V3 seed 42 (frozen in `sync_params.json`), then V4 on 5 seeds | 5 | about 1 h |

```bash
./twin_all.sh mps check          # first; delete results/_twin_check and logs/_twin_check afterwards
./twin_all.sh mps A              # terminal 1
./twin_all.sh mps B              # terminal 2, at the same time
./twin_all.sh mps final          # after lane A has finished
```

After lane `final`, make the tables and the H11 verdict (no training; the file bootstrap rebuilds the windows of
each seed once, about 15 minutes in total):

```bash
python scripts/17_twin_compare.py --config configs/twin.yaml
```

It writes `results_rerun/tables/twin_*.csv` and `results_rerun/twin_h11.json`.

Then the poisoning test and the cost (no training). Run step 19 when nothing else is running, so the
timings are clean:

```bash
python scripts/18_twin_poison.py --config configs/twin.yaml --seed 42 --set train.device=mps   # 10-20 min
python scripts/19_twin_cost.py --config configs/twin.yaml                                     # 2-5 min
```

Each run folder has `posthoc_v2.json` written by the run itself (corrected AURC, consistency, tuned binary
threshold), so step 11 is not needed for the twin runs. Send back `results_rerun/twin/` without `model.pt`
files, plus `logs_rerun/` (zipped).
