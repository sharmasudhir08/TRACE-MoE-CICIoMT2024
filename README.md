# TRACE-MoE on CICIoMT2024

Companion repository for the paper:

> **TRACE-MoE: A Protocol-Routed Mixture of Experts with Hierarchical Heads and
> Calibrated Abstention for Multi-Class Intrusion Detection in the Internet of
> Medical Things.**
> 
> *Computers, Materials & Continua*, Special Issue "Deep Learning for
> Next-Generation Cybersecurity: Architectures, Robustness and Applications,"
> 2026 (under review).

## Scope of this repository

This repository is provided for **inspection and verification** of every
result, table, and figure reported in the manuscript — it is not intended for
end-to-end re-execution by reviewers. It contains:

- the core experimental pipeline code (data preparation through model
  training), for transparency on how results were produced;
- every aggregated result file cited by a table or figure number in the
  paper;
- the finalized manuscript bundle with full traceability; and
- a SHA-256 integrity manifest (`paper/CHECKSUMS.json`) tying every uploaded
  table/figure, and all 50 underlying experimental runs, back to a single
  fixed codebase and dataset snapshot.

Scripts for ablation studies, robustness analysis, result aggregation, and
manuscript generation are intentionally not included (see below) — their
outputs are already present here as data.

## Repository structure

```
TRACE-MoE-CICIoMT2024/
├── README.md
├── LICENSE
├── requirements.txt
├── METHODOLOGY.md, WORKFLOW.md, HOW_TO_RUN.md, DATASETS.md, RESULTS_GUIDE.md, CONTENTS.md
│
├── src/trace_iomt/        TRACE-MoE library: models, routing, hierarchical
│                           heads, calibrated abstention, data loaders
├── scripts/                00_make_synthetic.py, 01_prepare.py, 02_audit.py,
│                           03_train_rf.py, 04_train_deep.py, _common.py
├── configs/                 frozen YAML configs (default_v2.yaml = the 5-seed
│                           settings used for every reported result)
├── tests/                   smoke-test suite
│
├── paper/                   the finalized, submitted manuscript bundle
│   ├── paper.docx, paper.pdf, paper.tex
│   ├── references.bib
│   ├── traceability.json    maps every table/figure number to its source file
│   ├── CHECKSUMS.json       SHA-256 of every table/figure + the shared
│   │                       code/data fingerprint across all 50 runs
│   ├── figures/              (7 files, matching results_v2/figures/)
│   └── tables/                (13 files, matching results_v2/tables/)
│
├── report/
│   ├── environment.json     exact Python/CUDA/library versions used
│   └── hypotheses.csv       H1-H10 verdicts (Table 1 source)
│
├── results_v2/
│   ├── tables/               11 CSVs cited by Tables 2-12
│   ├── efficiency/efficiency.csv   Table 13 source
│   └── figures/               7 PNGs cited by Figures 1-7
│
└── logs_v2/                  training session logs
```

## What is NOT in this repository, and why

| Not included | Why |
|:---|:---|
| CICIoMT2024 raw dataset (13 GB) | Public dataset, not redistributable here. Download directly from the Canadian Institute for Cybersecurity: <https://www.unb.ca/cic/datasets/iomt-dataset-2024.html>. |
| `scripts/05_ablation.py`, `06_robustness.py` | The H4 ablation and H6/H7 adversarial-robustness experiments were not run on real data for this paper; both are named in the manuscript's Future Work section. |
| `scripts/07`–`12` (unknown-attack probe, efficiency, aggregation, technical report, paper-writer, review analyses) and all `build_paper*.py` | These produce the paper and the aggregated tables/figures, which are already included here as data. The scripts themselves are internal tooling, not needed to inspect or verify a reported number. |
| Per-run artefacts under `results_v2/main/` (~318 MB: `metrics.json`, `config.json`, `history.csv`, `model.pt`, `predictions.npz` for 50 runs) | Superseded by `paper/CHECKSUMS.json`, which records that all 50 runs share one `code_sha256` and one `data_sha256` fingerprint — the provenance evidence without the bulk. |
| `results_v2/verification/` | Internal audit notes from an intermediate project stage, superseded by the current, finalized results and `CHECKSUMS.json`. |
| Comparator-paper PDFs | Copyrighted. All comparators are cited by DOI in `references.bib`. |

## Verifying a reported number

1. Find the table/figure number in `paper/traceability.json` to get its
   `source_file` (the canonical CSV/PNG under `results_v2/`).
2. Look up the same entry in `paper/CHECKSUMS.json` for its `source_sha256`
   and `local_sha256` — hash the file yourself (`sha256sum <file>` or
   PowerShell `Get-FileHash <file> -Algorithm SHA256`) and confirm it
   matches.
3. `paper/CHECKSUMS.json → provenance_fingerprint` records a single
   `code_sha256`/`data_sha256` pair, confirmed identical across all 50
   experimental run folders — i.e. every number in the paper comes from one
   fixed codebase and dataset snapshot, not a mix of runs from different code
   versions.

## Core pipeline scripts included (for transparency)

| Script | What it produces |
|:---|:---|
| `00_make_synthetic.py` | Synthetic smoke-test data (pipeline validation only) |
| `01_prepare.py` | Parquet table, class-count table from the raw CICIoMT2024 CSVs |
| `02_audit.py` | Data sanity checks: duplicates, overlap, shortcut features |
| `03_train_rf.py` | Random Forest, official and random splits, 5 seeds |
| `04_train_deep.py` | TRACE-MoE, BiLSTM-DNN, Transformer, TF-SAGE-lite; abstain report and expert-routing report per run |

These five scripts plus `configs/default_v2.yaml` are sufficient to
regenerate the raw per-seed run artefacts if you independently obtain the
CICIoMT2024 dataset. Aggregation into the tables/figures published here
(steps 05–12 of the authors' private pipeline) is not part of this
repository.

## Hardware and software environment used

- Windows 11, Intel Core i7-12700H CPU, 16 GB RAM
- NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM, CUDA 12.1)
- Python 3.12.4, PyTorch 2.5.1+cu121, scikit-learn 1.9.1
- Full version list: `report/environment.json`

Five seeds (42-46) are frozen in `configs/default_v2.yaml`.

## Citing this work

```bibtex
@article{sharma2026tracemoe,
  title   = {{TRACE-MoE}: A Protocol-Routed Mixture of Experts with
             Hierarchical Heads and Calibrated Abstention for Multi-Class
             Intrusion Detection in the Internet of Medical Things},
  author  = {},
  journal = {Computers, Materials \& Continua},
  year    = {2026},
  note    = {Special Issue on Deep Learning for Next-Generation Cybersecurity,
             under review}
}
```

## License

- Code and documentation: MIT — see `LICENSE`.
- The CICIoMT2024 dataset is governed by CIC's own dataset terms and is not
  redistributed here.

## Access

This is a private repository. Reviewer access is granted individually via
GitHub's Collaborator mechanism (read-only) — it is not publicly listed.

## Contact

For questions about this repository, open a GitHub issue, or contact the
corresponding author at the address listed in the manuscript.
