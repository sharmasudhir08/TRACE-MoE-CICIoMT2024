# Results guide

## Layout

```
results/                          simulation record (no .docx is written here)
├── registry.jsonl                append-only ledger: one JSON line per run ever done
├── data/                         class counts per split (step 01)
├── audit/                        data sanity checks (step 02); audit_report.md is the readable summary
├── main/<run>/                   steps 03 and 04 (RF and the neural models). Append-only: <run>__v2, __v3, ...
├── ablation/<run>/               step 05
├── robustness/<run>/             step 06 (+ transfer_matrix_*/transfer_matrix.json)
├── zero_day/<run>/               step 07 (unknown attacks)
├── efficiency/efficiency.csv     step 08
├── tables/                       step 09: CSV + LaTeX for the paper, incl. all_runs_long.csv (one row per run x level)
│                                 + <table>.provenance.json sidecars listing exact runs/git_commits per row
├── figures/                      step 09: PNG figures (figures.dpi, default 600)
└── summary.md                    step 09: all key tables in one file

documents/                        final deliverables, written only at the very end of the pipeline
├── report/                       step 10: technical_report.docx, manifest.json (ID -> source + SHA-256 + git_commit),
│                                 hypotheses.json (frozen H1-H10 verdicts), environment.json, tables/
└── paper/                        step 11: paper_plan.docx, paper.docx, paper.tex, references.bib,
                                  traceability.json (paper table/figure -> report ID -> source + SHA-256 + git)
```

**One-command audit after every rerun:** `python scripts/verify_traceability.py` reports whether every number in
`documents/paper/paper.docx` and `documents/report/technical_report.docx` resolves to a source file whose SHA-256
still matches, and whether every `results/*/*/metrics.json` is listed in `registry.jsonl`.

Run folder name: `<model>_<regime>_T<window length>[_<tag>]_seed<seed>`. Each holds:

| File | Content |
|---|---|
| `config.json` | the exact configuration of the run |
| `metrics.json` | `test` and `val` metrics per level (label2/label6/label19) with summary metrics, `per_class` and `confusion_matrix`; `abstain` (temperature, threshold, coverage, selective risk, AURC, ECE before/after, risk-coverage curve); `routing` for TRACE-MoE (NMI with category/class, expert shares, counts per category); run metadata **including `train_seconds`, `test_seconds`, `n_params`, `epochs_run`, `n_test_windows`** |
| `history.csv` | per-epoch loss (incl. `balance_loss` for TRACE-MoE), per-epoch wall time (`seconds`) and validation macro-F1 |
| `predictions.npz` | `y_true_<level>`, `y_pred_<level>`, `route` (TRACE-MoE expert per window), and `proba_<level>` (class probabilities, float16) when `evaluation.save_proba: true` (default) |
| `model.pt` | trained weights (neural models) |

Aggregate view: `results/tables/all_runs_long.csv` has one row per (run × level) with model, family, regime, seed, T,
tag, `n_params`, `train_seconds`, `test_seconds` and the summary metrics — so every number in the main comparison
tables can be mapped back to a specific `results/<experiment>/<run>/metrics.json`.

## Tables (results/tables)

| File | Use in the paper | Hypothesis |
|---|---|---|
| `main_official_label19.*` | **main comparison table** (19 classes, official split) | H1, H2 |
| `main_official_label6.*`, `main_official_label2.*` | category and binary results | H3 |
| `main_random_*` | the same under the random split (comparable with most published papers) | |
| `split_inflation.*` | macro-F1 per model under each split | |
| `wilcoxon_trace_moe_vs_baselines_official.*` | TRACE-MoE vs each baseline, per-class F1, Holm | H1 |
| `friedman_nemenyi_official.json` | Friedman test, average ranks, critical difference | |
| `noninferiority_binary.*` | binary macro-F1 TRACE-MoE minus RF with 95% CI | H3 |
| `published_comparison.*` | TRACE-MoE next to results **reported** by the 5 closest papers | positioning |
| `ablation.*`, `ablation_tests.*` | TRACE-MoE variants and window length; paired tests vs the full model | H4 |
| `expert_routing.*`, `expert_by_category_official.csv` | expert specialisation (NMI) and load | H5 |
| `robustness.*`, `robustness_curves.csv` | attack detection rate vs ε, before/after adversarial fine-tuning | |
| `robustness_asr.*` | clean vs attacked macro-F1, drop and ASR | H7 |
| `robustness_asr_tests.*` | per-class ASR tests, TRACE-MoE vs each baseline | H6 |
| `transfer_attack_matrix.*` | black-box transfer ASR (rows: surrogate, columns: target, incl. RF) | |
| `abstain_official.*`, `abstain_random.*` | coverage, error with/without abstention, AURC, ECE | H9 |
| `zero_day_overall.*`, `zero_day.*` | unknown attacks: abstain-score vs softmax AUROC, flagged-unknown, false alarms, caught at all | H8 |
| `all_runs_long.csv` | every run × level in one table | |

Efficiency (H10) is in `results/efficiency/efficiency.*`. The LaTeX tables use `booktabs` and bold the best
value per column. Values are percentages, mean ± std over seeds.

## Figures (results/figures)

`per_class_f1_official`, `confusion_trace_moe_official`, `critical_difference_official`, `val_curves_official`,
`split_inflation`, `window_length`, `expert_by_category_official`, `robustness_curve_clean_model`,
`robustness_curve_adv_finetuned`, `risk_coverage`.

## Writing results honestly

* The official split is the main result; the random split is shown only for comparison with published numbers.
* Published values are reported, not reproduced; say so wherever they appear.
* If TRACE-MoE does not win on clean F1, keep the table as it is and lead with robustness and abstention
  (the paper plan does this automatically from the hypothesis verdicts).
