# Release V2_1_2026_11_11

Code and results behind "A Behavioural Digital Twin for Device-Aware Edge Intrusion Detection in the Internet of
Medical Things" (IEEE Internet of Things Journal submission).

Same code and results as V2_2026_11_07, plus `HYPOTHESES.md`, which maps the hypothesis labels of the
three papers from this project to the protocol labels (P-H1 to P-H11) that the code and result files use.

## What is included

- `src/`, `scripts/`, `tests/`, `configs/`: the rebuilt TRACE-MoE pipeline (steps 00–11), the per-device packet-capture
  extractor (steps 12–14) and the Digital Twin study (steps 15–19). `HOW_TO_RUN.md`, `RERUN.md` and `WORKFLOW.md`
  explain how to run them; `rerun_all.sh` and `twin_all.sh` run the full studies.
- `rerun_2026-11-02/`: the re-run of all 103 experiments on the official CICIoMT2024 CSVs (tag V1_2026_11_02).
- `rerun_2026-11-03/`: the post-hoc step (tie-aware AURC, consistency, tuned thresholds; tag V1_1_2026_11_03).
- `twin_2026-11-04/`, `twin_2026-11-05/`: extraction, validation and dry run of the twin study.
- `twin_2026-11-07/`: the twin study (26 runs), its tables, the H11 verdict, the poisoning test and the cost
  (tag V2_2026_11_07).
- `audit_2026-11-07/`: the data audit on the full data.
- `results_v2/`, `logs_v2/`, `paper/`, `report/`: the original authors' results and paper, kept for comparison.

Every run folder has `metrics.json` (with the SHA-256 of its code and data) and, where applicable,
`posthoc_v2.json`. Trained models and per-window predictions are not included because of their size.

## Changes for publication

Paths on the training machine are replaced by `<repo>`, `<data>` and `<home>`, and its host name by `mac-studio`,
in logs and run records; log file names are renamed the same way. `twin_model.pkl` is not included: loading a
Python pickle runs code, and step 15 (`scripts/15_twin_build.py`) regenerates it from the data. Code, tests,
configurations and documentation are unchanged from the tagged commit.

## Data

CICIoMT2024 is available from the Canadian Institute for Cybersecurity:
https://www.unb.ca/cic/datasets/iomt-dataset-2024.html
