# How to run TRACE-MoE on your own computer (step by step)

Type the commands in a terminal: **Anaconda Prompt** or **PowerShell** on Windows, Terminal on Linux/macOS.
Run every command from inside the `trace-iomt` folder.

## Experimental Seed Protocol

### Excluded seed
`Seed 42 v1` is **permanently excluded** from future comparison, aggregation, tables, figures, technical reports and the research paper.

### Included seeds (the final five-seed experimental set)

```
42 v2
43
44
45
46
```

All five seeds use exactly the same experimental configuration (`configs/default_v2.yaml`). The random seed is the only intended experimental difference.

### Current execution status

```
42 v2 — completed (RF folders on disk; 8 deep-model folders deleted — see verification report, pending rerun before Step 09 if 3-seed deep aggregation is required)
43    — completed (all 10 run folders on disk)
44    — completed (all 10 run folders on disk)
45    — pending execution
46    — pending execution
```

### Configuration policy

Seeds 45 and 46 **must** be executed using the same experimental configuration as Seed 42 v2, Seed 43 and Seed 44. The random seed is the only intended experimental difference. Verification `results_v2/verification/configuration_audit.md` confirms that 50 of 51 scientifically-meaningful keys are identical across seeds 42 v2, 43 and 44; the one mismatch is `train.seed` itself.

### Approved command for Seeds 45 and 46 (PowerShell, Windows)

Paste this single command into a PowerShell window started in `D:\Sharda University\CICIoMT2024\trace-iomt`. It runs seed 45 (step 03 then step 04); stops on failure; only continues to seed 46 if seed 45 succeeds. Each seed gets its own `TRACE_SESSION`. Append-only run folders ensure nothing overwrites existing seed 43 / seed 44 artefacts.

```powershell
cd "D:\Sharda University\CICIoMT2024\trace-iomt"; $env:OMP_NUM_THREADS="20"; $env:MKL_NUM_THREADS="20"; $env:OPENBLAS_NUM_THREADS="20"; $env:NUMEXPR_NUM_THREADS="20"; $py=".\.venv\Scripts\python.exe"; $cfg="configs\default_v2.yaml"; foreach ($s in 45,46) { $env:TRACE_SESSION = "v2_seed" + $s + "_" + (Get-Date -Format "yyyy-MM-dd_HHmmss"); Write-Host ("`n=== SEED {0} | TRACE_SESSION={1} ===" -f $s, $env:TRACE_SESSION); & $py scripts\03_train_rf.py --config $cfg --seeds $s --set "train.device=cuda"; if ($LASTEXITCODE -ne 0) { Write-Error "Seed $s step 03 (RF) failed with exit $LASTEXITCODE"; exit $LASTEXITCODE }; & $py scripts\04_train_deep.py --config $cfg --seeds $s --set "train.device=cuda"; if ($LASTEXITCODE -ne 0) { Write-Error "Seed $s step 04 (deep) failed with exit $LASTEXITCODE"; exit $LASTEXITCODE } }
```

### Downstream analysis

Steps 08 (`08_efficiency.py`), 09 (`09_compare.py`), 10 (`10_technical_report.py`) and 11 (`11_write_paper.py`) are **ON HOLD** and will not be executed until explicitly approved after seeds 45 and 46 complete.


## Step 0. Unzip and open the folder
```
cd path\to\trace-iomt
```

## Step 1. Create a Python environment (Python 3.10, 3.11 or 3.12)
With Anaconda:
```
conda create -n trace python=3.11 -y
conda activate trace
```
Or with plain Python:
```
python -m venv .venv
.venv\Scripts\activate            (Windows)
source .venv/bin/activate         (Linux/macOS)
```

## Step 2. Install PyTorch with GPU support, then the rest
Check your CUDA version with `nvidia-smi` (top right, "CUDA Version"). Then run, for example:
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The last line must print `True` and your GPU name. If it prints `False`, reinstall torch with the index URL for
your CUDA version from https://pytorch.org/get-started/locally/.

## Step 3. Check the code on dummy data (about 6 minutes)
```
python -m pytest -q tests
python run_all.py --smoke
```
The tests should end with `7 passed`, and every step should finish with `exit code 0`.
The dummy outputs go to `results/_smoke/` and `logs/_smoke/`. They only show that the code works and what the
outputs look like.

## Step 4. Put the real CSVs in place
Copy the CICIoMT2024 Wi-Fi/MQTT CSV folder so that you have
```
trace-iomt/data/raw/CICIoMT2024/WiFi_and_MQTT/attacks/CSV/train/*.csv
trace-iomt/data/raw/CICIoMT2024/WiFi_and_MQTT/attacks/CSV/test/*.csv
```
If your data is somewhere else, add `--set paths.ciciomt_csv_dir="D:/data/CICIoMT2024/.../CSV"` to step 5.

## Step 5. Prepare the data and run the sanity checks
```
python scripts/01_prepare.py
python scripts/02_audit.py
```
Check `results/data/ciciomt2024_class_counts.csv`: it should list **19 classes**.
Then read `results/audit/audit_report.md`.
If the machine has less than 16 GB RAM, run step 01 like this instead:
`python scripts/01_prepare.py --set data.max_rows_per_file=200000`

## Step 6. Do a quick trial with one seed and see the timings
```
python scripts/03_train_rf.py --seeds 42 --regimes official
python scripts/04_train_deep.py --seeds 42 --regimes official --models trace_moe transformer
```
The logs show the time of each epoch. If you get "CUDA out of memory", add `--set train.batch_size=128`.

## Step 7. The full experiments (run them one after another)
```
python scripts/03_train_rf.py
python scripts/04_train_deep.py
python scripts/05_ablation.py
python scripts/06_robustness.py
python scripts/07_unknown_attacks.py
python scripts/08_efficiency.py
python scripts/09_compare.py
python scripts/10_technical_report.py
python scripts/11_write_paper.py
```
Or run everything at once with `python run_all.py`. To continue from a step, use for example `python run_all.py --from 5`.
Every step runs 5 seeds (42-46) by default. If time is short, add `--seeds 42 43 44` to steps 05, 06 and 07.

## Step 8. Collect the results for the paper

The simulation record lives in `results/` and `logs/` (CSV, JSON, PNG, LaTeX, `.npz`, `.log`). Every per-run
`metrics.json` carries a reproducibility fingerprint (git commit, code SHA-256, data SHA-256, started/finished
timestamps, host, session) and every aggregated table in `results/tables/` has a sibling `<table>.provenance.json`
listing the exact run folders and git commits that fed each row. `results/registry.jsonl` is a single
append-only ledger of every run ever done. The final Word/LaTeX deliverables are written at the END of the
pipeline by steps 10 and 11, into a separate `documents/` folder.

After step 11, audit the chain any time with:
```
python scripts/verify_traceability.py              # pass/fail summary
python scripts/verify_traceability.py --strict     # non-zero exit on any failure
```

* `documents/report/technical_report.docx` is the full technical report (Word). Every table and figure has an ID
  (T1…, F1…) and names its source file in `results/`, the script and the log. It covers the research-question and
  hypothesis verdicts (H1-H10), environment, dataset, sanity checks, every table and figure, run times, and all warnings/errors from `logs/`.
  `documents/report/manifest.json` holds the SHA-256 of every source file so the paper can refuse to run if anything
  changed afterwards.
* `documents/paper/paper_plan.docx` is the **paper plan** made from the technical report. It covers the claims the
  results allow, the storyline, which report table/figure goes in which section, and the CMC checklist. Read it first.
* `documents/paper/paper.docx` and `paper.tex` are the **paper draft**. It uses exactly the report's tables and figures:
  each caption gives the report ID and the result file, and the appendix maps every paper table/figure to the report,
  the result file and the log.
* If you re-run any experiment, run steps 09, 10 and 11 again. Step 11 refuses to run when results changed after the report.
  * Compile it with `pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper`, or upload the
    `documents/paper` folder to Overleaf.
  * Lines marked `[EDIT]` need your input. Introduction and Related Work must be merged from our paper draft.
  * Read every sentence before submitting. It is a draft.
* `results/summary.md` has all the main tables in one place.
* `results/tables/*.tex` are LaTeX tables, ready to paste. `results/tables/*.csv` have the same numbers.
* `results/figures/*.png` are the figures, at 600 dpi.
* `logs/*.log` hold the full log of every script.

Send me `documents/report/technical_report.docx` and the `results/tables` folder (zipped), and I will help write the
Results and Discussion sections from them.

## Useful options
| Need | Add to the command |
|---|---|
| fewer epochs | `--set train.epochs=15` |
| smaller batch (GPU memory) | `--set train.batch_size=128` |
| CPU only | `--set train.device=cpu` |
| drop the shortcut features found by the audit | `--set data.drop_features=auto` |
| other window length | `--T 8` (step 04) or `--set windows.length=8` |
