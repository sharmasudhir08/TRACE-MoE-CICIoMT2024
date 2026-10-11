# What is in the trace-iomt folder (TRACE-MoE)

## Documents (Markdown)
| File | What it is |
|---|---|
| `README.md` | Overview, folder layout, install, list of steps |
| `HOW_TO_RUN.md` | Step-by-step instructions to run everything on your own computer |
| `METHODOLOGY.md` | Research questions, objectives, hypotheses H1-H10 (Section 0) and the method in detail |
| `WORKFLOW.md` | Pipeline diagram, one-week plan, mapping of RQs to output files |
| `DATASETS.md` | Where to put the CICIoMT2024 files and the expected file names |
| `RESULTS_GUIDE.md` | What every output file contains, where it goes in the paper, which hypothesis it tests |
| `CONTENTS.md` | This list |
| `requirements.txt` | Python packages to install |

## Run everything
| File | What it does |
|---|---|
| `run_all.py` | Runs steps 01-11 in order (`--smoke` runs 00-11 on dummy data) |
| `run_all.sh` | The same for Linux/macOS |
| `configs/default.yaml` | All settings for the real experiments (5 seeds), every key commented, incl. hypothesis thresholds |
| `configs/smoke.yaml` | Tiny settings for the dummy-data test |
| `configs/published_results.yaml` | Results reported by the 5 closest CICIoMT2024 papers, with a `verified` flag per value |

## Scripts (`scripts/`, run in order)
| File | What it does |
|---|---|
| `00_make_synthetic.py` | Makes dummy CSVs with the exact CICIoMT2024 layout (testing only) |
| `01_prepare.py` | **Preprocessing:** reads the CSVs, labels from file names (19/6/2 classes), saves Parquet, class counts |
| `02_audit.py` | **Sanity checks:** duplicates, train/test overlap, single-feature shortcuts |
| `03_train_rf.py` | Random Forest baseline (tree-ensemble winner in published papers) |
| `04_train_deep.py` | TRACE-MoE (proposed), BiLSTM-DNN, Transformer, TF-SAGE-lite; abstain and routing reports |
| `05_ablation.py` | TRACE-MoE variants (dense expert, protocol routing, no experts, no balance loss, ...) and window lengths |
| `06_robustness.py` | Feature-constrained PGD, per-class ASR, adversarial fine-tuning, black-box transfer matrix |
| `07_unknown_attacks.py` | Leave-one-attack-out unknown-attack detection with the abstain score and reject threshold |
| `08_efficiency.py` | Parameters, model size, CPU/GPU latency |
| `09_compare.py` | **Comparison metrics:** mean ± std, Wilcoxon, Friedman/Nemenyi, non-inferiority, published-results table, LaTeX + CSV tables, figures. **Also writes `<table>.provenance.json` sidecars listing the exact runs that produced each row.** |
| `10_technical_report.py` | Technical report (Word .docx) from `results/` and `logs/` with the H1-H10 verdicts; every table/figure has an ID, source file, script, log and git commit; **also freezes `hypotheses.json`, `environment.json` and `manifest.json`**. |
| `11_write_paper.py` | Paper plan (.docx) from the report, then the paper draft (.docx + .tex + references.bib); **also writes `traceability.json` mapping every paper table/figure to a report item + source file + git commit**; stops if results changed after the report |
| `verify_traceability.py` | **Audit tool** — checks that every table/figure in the report + paper traces to a source file whose SHA-256 still matches, and that every `metrics.json` is listed in `registry.jsonl`. Run after step 11 (or any time) |
| `_common.py` | Shared setup (config, logging) used by every script; attaches an append-only session log at `logs/sessions/<session>_<step>_<host>.log` |

## Source code (`src/trace_iomt/`)
| File | What it does |
|---|---|
| `config.py` | Reads YAML configs and `--set key=value` overrides |
| `utils.py` | Seeds, logging to `logs/`, JSON helpers |
| `data/loaders.py` | Loads the raw CSVs into one table |
| `data/labels.py` | Class names → 19-class / 6-category / binary labels and the hierarchy |
| `data/preprocess.py` | Robust scaling (fitted on training only), protocol token, duplicate-window removal |
| `data/windows.py` | Split regimes (official/random/LOAO) and windows of consecutive flows |
| `models/networks.py` | **TRACE-MoE** (router, experts, balance loss) and the baselines BiLSTM-DNN, Transformer, TF-SAGE-lite |
| `models/heads.py` | Hierarchical heads, logit-adjusted focal loss, consistency loss, energy score |
| `models/classical.py` | Random Forest |
| `models/factory.py` | Builds a model by name, counts parameters |
| `abstain.py` | **Abstain layer:** temperature calibration, reject threshold, risk-coverage/AURC, expert-routing report |
| `train.py` | Training loop: mixed precision (fits a 4 GB GPU), early stopping, MoE balance loss |
| `metrics.py` | All metrics: macro-F1, balanced accuracy, PR-AUC, recall at 1% FPR, ECE, per-class, confusion matrix, OOD AUROC |
| `audit.py` | Duplicate and shortcut checks |
| `attacks.py` | Constrained PGD, per-class ASR, adversarial fine-tuning, transfer matrix |
| `efficiency.py` | Latency measurement |
| `compare.py` | Aggregation, statistical tests, LaTeX tables, figures |
| `experiment.py` | Runs one model and saves its run folder |
| `reporting.py` | Collects results + logs for the report and the paper; hypothesis verdicts H1-H10 |

## Tests
| File | What it does |
|---|---|
| `tests/test_smoke.py` | 7 quick checks: labels, protocol token, metrics, windows never cross splits and all models run, synthetic generator, MoE routing modes and balance loss, abstain layer |

## Data, results, logs, documents
| Folder | Content |
|---|---|
| `data/raw/` | Put the CICIoMT2024 CSVs here |
| `data/processed/` | Parquet cache written by step 01 |
| `results/` | **Simulation outputs only** — one folder per experiment and run, plus `tables/` (incl. `all_runs_long.csv` + `<table>.provenance.json` sidecars), `figures/`, `summary.md`, and the append-only `registry.jsonl` ledger. Every per-run folder has `config.json`, `metrics.json` (with git_commit + code_sha256 + data_sha256 + started_at/finished_at + host/session fingerprint), `history.csv`, `predictions.npz` (test + val + class probabilities) and `model.pt`. Append-only: a repeated run goes to `<name>__v2`, `<name>__v3`, …; nothing is destroyed. No `.docx` is ever written here. |
| `logs/` | `logs/<step>.log` (current invocation, overwritten) **and** `logs/sessions/<session>_<step>_<host>.log` (append-only history, one timestamped file per invocation × step). `TRACE_SESSION` is set once by `run_all.py` so every step of one pipeline run shares the session id. |
| `documents/` | **Final deliverables only** — `report/technical_report.docx` + `manifest.json` + `hypotheses.json` + `environment.json` (step 10) and `paper/paper.docx` + `paper.tex` + `paper_plan.docx` + `references.bib` + `traceability.json` (step 11). Each table/figure maps back to a source file via `manifest.json` (ID, source path, SHA-256, producing script, log, git commit, provenance sidecar). |
| `results/_smoke/`, `logs/_smoke/`, `documents/_smoke/` | Example outputs from the dummy-data test run (the numbers mean nothing) |
