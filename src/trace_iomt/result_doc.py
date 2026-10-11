"""Living simulation ledger: documents/result.docx, rebuilt from results/registry.jsonl on every run.

Each new run adds one line to results/registry.jsonl (see experiment._append_registry). This module
reads that ledger and renders a Word document that is always current — the user can watch the file
grow as experiments progress. It is independent from the end-of-pipeline technical_report.docx and
paper.docx, which are frozen snapshots built by steps 10 and 11.

Call sites:
  - experiment.save_run()        auto-refreshes when run.live_result_doc is true (default)
  - scripts/update_result_doc.py standalone refresh on demand
"""
from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _read_registry(path: Path):
    """Load registry.jsonl into a DataFrame; return an empty frame if the ledger is missing."""
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return pd.DataFrame(rows)


def _fmt_pct(x, d=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or x is None)):
        return "n/a"
    return f"{100 * float(x):.{d}f}%"


def _mean_pm(vals, d=2):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return "n/a"
    v = np.array(v, dtype=float)
    if len(v) == 1:
        return f"{100 * v[0]:.{d}f}%"
    return f"{100 * v.mean():.{d}f}% ± {100 * v.std(ddof=1):.{d}f}%"


def _add_table(doc, df, header_bold=True, font_pt=8):
    from docx.shared import Pt
    if len(df) == 0:
        p = doc.add_paragraph("(no runs yet)")
        p.runs[0].italic = True
        return
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    for j, c in enumerate(df.columns):
        cell = t.rows[0].cells[j]; cell.text = ""
        r = cell.paragraphs[0].add_run(str(c))
        if header_bold:
            r.bold = True
        r.font.size = Pt(font_pt)
    for row in df.itertuples(index=False):
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = ""
            s = f"{v:.4f}" if isinstance(v, float) else str(v)
            cells[j].paragraphs[0].add_run(s).font.size = Pt(font_pt)


def _summary_by_experiment(reg: pd.DataFrame):
    """One row per experiment with # runs, total train time, seeds, last-run timestamp."""
    if not len(reg):
        return pd.DataFrame()
    g = reg.groupby("experiment", dropna=False)
    out = pd.DataFrame({
        "runs": g.size(),
        "models": g["model"].agg(lambda s: ", ".join(sorted(set(x for x in s if isinstance(x, str))))),
        "seeds": g["seed"].agg(lambda s: ", ".join(sorted(set(str(int(x)) for x in s if pd.notna(x))))),
        "train_minutes": (g["train_seconds"].sum() / 60).round(1),
        "last_run": g["ts"].max(),
    })
    return out.reset_index()


def _per_model_table(reg: pd.DataFrame, experiment="main"):
    """Mean ± std of macro-F1 at each level, grouped by (experiment, model, regime)."""
    d = reg[reg.get("experiment") == experiment]
    if not len(d):
        return pd.DataFrame()
    rows = []
    for (model, regime), g in d.groupby(["model", "regime"], dropna=False):
        rows.append({
            "model": model, "regime": regime, "n_seeds": int(g["seed"].nunique()),
            "n_runs": int(len(g)),
            "label19": _mean_pm(g["macro_f1_label19"]),
            "label6":  _mean_pm(g["macro_f1_label6"]),
            "label2":  _mean_pm(g["macro_f1_label2"]),
            "train_s_mean": round(g["train_seconds"].mean(), 1) if g["train_seconds"].notna().any() else None,
        })
    return pd.DataFrame(rows).sort_values(["regime", "model"]).reset_index(drop=True)


def _recent_runs(reg: pd.DataFrame, limit=25):
    cols = ["ts", "experiment", "run", "model", "regime", "seed", "T", "tag",
            "train_seconds", "macro_f1_label19", "git_commit", "run_id"]
    cols = [c for c in cols if c in reg.columns]
    d = reg[cols].copy()
    if "ts" in d.columns:
        d = d.sort_values("ts", ascending=False)
    return d.head(limit).reset_index(drop=True)


def _augmentation_meta(prev_path: Path):
    """If result.docx already exists, read its properties to tell the user how many times we've updated it."""
    if not prev_path.exists():
        return 1, None
    try:
        from docx import Document
        doc = Document(str(prev_path))
        prev = doc.core_properties.revision or 1
        return int(prev) + 1, doc.core_properties.modified
    except Exception:
        return 1, None


def render(registry_path: Path, out_path: Path, cfg: dict | None = None):
    """Rebuild <out_path> from <registry_path>. Idempotent and fast (reads only the ledger).
    Returns a short status string."""
    from docx import Document
    from docx.shared import Inches, Pt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reg = _read_registry(Path(registry_path))
    revision, prev_modified = _augmentation_meta(out_path)

    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Inches(0.8)

    doc.add_heading("Simulation results (running ledger)", 0)
    doc.add_paragraph(
        f"Augmentation #{revision} | updated {datetime.now():%Y-%m-%d %H:%M} | host {platform.node()} | "
        f"source: {registry_path}"
    )
    doc.add_paragraph(
        "This document is rebuilt from results/registry.jsonl on every run. It is the running record of "
        "every simulation done on this project. The frozen end-of-pipeline deliverables are "
        "documents/report/technical_report.docx (step 10) and documents/paper/paper.docx (step 11)."
    )
    if prev_modified is not None:
        doc.add_paragraph(f"Previous update: {prev_modified}").runs[0].italic = True

    if not len(reg):
        doc.add_paragraph("No runs recorded yet. Run scripts/03_train_rf.py or scripts/04_train_deep.py "
                          "to start accumulating runs.")
        doc.save(str(out_path))
        return f"Wrote {out_path} (empty ledger)"

    # Headline counts
    doc.add_heading("1. Headline counts", 1)
    headline = pd.DataFrame({
        "metric": ["total runs", "unique run_ids", "experiments", "models", "seeds", "sessions",
                   "git commits seen", "earliest run", "latest run"],
        "value": [len(reg),
                  reg["run_id"].nunique() if "run_id" in reg.columns else "n/a",
                  reg["experiment"].nunique(),
                  reg["model"].nunique(),
                  ", ".join(sorted(set(str(int(x)) for x in reg["seed"].dropna()))),
                  reg["run_id"].str.rsplit("_", n=2).str[0].nunique() if "run_id" in reg.columns else "n/a",
                  reg["git_commit"].nunique() if "git_commit" in reg.columns else "n/a",
                  reg["ts"].min() if "ts" in reg.columns else "n/a",
                  reg["ts"].max() if "ts" in reg.columns else "n/a"],
    })
    _add_table(doc, headline)

    # Summary by experiment
    doc.add_heading("2. Summary by experiment", 1)
    _add_table(doc, _summary_by_experiment(reg))

    # Per-model aggregates (main experiment)
    doc.add_heading("3. Main results (mean ± std over seeds)", 1)
    _add_table(doc, _per_model_table(reg, "main"))

    # Ablation
    abl = _per_model_table(reg, "ablation")
    if len(abl):
        doc.add_heading("4. Ablation (mean ± std over seeds)", 1)
        _add_table(doc, abl)

    # Robustness
    rob = _per_model_table(reg, "robustness")
    if len(rob):
        doc.add_heading("5. Robustness (per clean-model summary; detailed PGD/ASR tables in step 06 results)", 1)
        _add_table(doc, rob)

    # Zero-day
    zd = _per_model_table(reg, "zero_day")
    if len(zd):
        doc.add_heading("6. Zero-day (leave-one-attack-out, mean ± std over seeds)", 1)
        _add_table(doc, zd)

    # Recent 25 runs
    doc.add_heading("7. Last 25 runs", 1)
    _add_table(doc, _recent_runs(reg, 25))

    # Reproducibility fingerprints seen
    doc.add_heading("8. Reproducibility fingerprints", 1)
    rep = pd.DataFrame({
        "git_commit": sorted(set(reg["git_commit"].dropna().astype(str))) if "git_commit" in reg.columns else [],
    })
    _add_table(doc, rep)

    # Augmentation note
    doc.add_heading("9. How this file grows", 1)
    doc.add_paragraph(
        "Every call to experiment.save_run() appends one line to results/registry.jsonl AND refreshes this "
        "document. The frozen technical_report.docx and paper.docx are built once per step-10/11 invocation; "
        "this file instead reflects the current state of every run ever done. Call "
        "scripts/update_result_doc.py to refresh it manually. Set run.live_result_doc: false in the config to "
        "disable the auto-refresh."
    )

    doc.core_properties.revision = revision
    doc.core_properties.title = "Simulation results (running ledger)"
    doc.save(str(out_path))
    return f"Wrote {out_path} (augmentation #{revision}, {len(reg)} runs)"
