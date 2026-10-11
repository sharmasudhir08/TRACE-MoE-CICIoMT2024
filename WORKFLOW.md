# Workflow

## Pipeline

```
raw CSVs ──01_prepare──> data/processed/ciciomt2024.parquet
                │
                ├─02_audit──────────> results/audit/   (sanity checks; optional data.drop_features: auto)
                ├─03_train_rf───────> results/main/rf_<regime>_T1_seed*/
                ├─04_train_deep─────> results/main/<model>_<regime>_T16_seed*/   (+ model.pt, abstain, routing)
                ├─05_ablation───────> results/ablation/
                ├─06_robustness─────> results/robustness/   (reuses model.pt from step 04)
                ├─07_unknown_attacks> results/zero_day/
                └─08_efficiency─────> results/efficiency/
                                      │
                          09_compare ─┴─> results/tables/*.csv|.tex, results/figures/*.png, results/summary.md
                                      │      (+ published_comparison from configs/published_results.yaml)
                 10_technical_report ─┼─> documents/report/technical_report.docx + manifest.json (IDs, sources, checksums)
                     11_write_paper ──┴─> documents/paper/paper_plan.docx → paper.docx, paper.tex (same IDs; stops if results changed)
```

Every script reads `configs/default.yaml` (or `--config`), takes `--set key=value` overrides, and writes two
logs: `logs/<script>.log` (current invocation, overwritten) and `logs/sessions/<session>_<script>_<host>.log`
(append-only history, never overwritten). When `run_all.py` drives the pipeline, every step shares the same
`<session>` id via the `TRACE_SESSION` env var.

**Append-only runs:** by default (`run.append_only: true` in `configs/default.yaml`), repeating a run never
destroys the previous result — the second attempt lands in `<run>__v2`, the third in `<run>__v3`, and so on.
Every run also appends one JSON line to `results/registry.jsonl`, which is the single ledger of every run ever done.

**Audit any time:** `python scripts/verify_traceability.py` confirms every number in the report/paper traces
back to a source file whose SHA-256 still matches, and every `metrics.json` is listed in the registry.

## One-week plan (target: CMC special issue, deadline 15 Dec 2026)

| Day | Work | Commands |
|---|---|---|
| 1 | Install, smoke test, prepare data, sanity checks; RF (2 splits × 5 seeds) | `pytest -q tests`, `python run_all.py --smoke`, `01_prepare.py`, `02_audit.py`, `03_train_rf.py` |
| 2 | One-seed run of all neural models to check timings | `04_train_deep.py --seeds 42` |
| 3 | All neural models, 5 seeds, official then random split | `04_train_deep.py --regimes official`, then `--regimes random` |
| 4 | Ablation and window length | `05_ablation.py` (use `--seeds 42 43 44` if time is short) |
| 5 | Robustness | `06_robustness.py` |
| 6 | Unknown attacks, efficiency; build all tables | `07_unknown_attacks.py`, `08_efficiency.py`, `09_compare.py` |
| 7 | Technical report, read it, then paper plan and draft | `10_technical_report.py`, `11_write_paper.py` |

## Practical tips

* **Do one seed first.** Run every step with `--seeds 42`, then the other seeds.
* **Out of memory (GPU):** `--set train.batch_size=128`. **Out of memory (RAM):** re-run step 01 with `--set data.max_rows_per_file=200000`.
* **Slow RF:** `--set classical.max_rows_per_class=50000` or `--set classical.n_estimators=100`.
* **Shortcut features:** if the sanity check suggests features to drop, decide once and use `data.drop_features: auto` everywhere.
* **Step 06** reuses trained models when `results/main/<model>_official_T16_seed<k>/model.pt` exists, so run step 04 first.
* **Changed something?** Run `09_compare.py` again, then 10 and 11. Step 11 refuses to run if results changed after step 10.
* **Published values:** `configs/published_results.yaml` was checked against the publisher pages on 2 Oct 2026; re-check if you add a paper.

## Mapping to the research questions

| RQ | Question | Evidence |
|---|---|---|
| RQ1 | Does protocol-routed MoE improve 19-class detection over the published winners? | `tables/main_official_*`, `wilcoxon_*`, `noninferiority_binary`, `published_comparison` |
| RQ2 | Do experts specialise, and does routing help? | `tables/ablation*`, `tables/expert_routing`, `figures/expert_by_category_*` |
| RQ3 | How robust are the models at 19 classes, and does adversarial fine-tuning help? | `tables/robustness*`, `transfer_attack_matrix` |
| RQ4 | Can calibrated abstention flag unseen attacks and lower the error on accepted traffic? | `tables/abstain_*`, `tables/zero_day*`, `figures/risk_coverage` |
| RQ5 | Is TRACE-MoE light enough for an edge gateway? | `efficiency/efficiency.*` |
