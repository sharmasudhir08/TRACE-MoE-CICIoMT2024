"""Step 1. Read the raw CSVs, attach labels from the file names, and cache one Parquet table.

Outputs
  data/processed/ciciomt2024.parquet
  results/data/ciciomt2024_files.csv         one line per CSV file
  results/data/ciciomt2024_class_counts.csv  rows per class in train / test (paper Table: dataset)
"""
from pathlib import Path

from _common import setup
from trace_iomt.compare import write_table
from trace_iomt.data.loaders import load_ciciomt2024, save_table


def main():
    args, cfg, log = setup("Prepare dataset", "01_prepare", None)
    out = Path(cfg["paths"]["results_dir"]) / "data"
    out.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading CICIoMT2024 from {cfg['paths']['ciciomt_csv_dir']}")
    df, manifest = load_ciciomt2024(cfg["paths"]["ciciomt_csv_dir"], cfg["data"]["max_rows_per_file"],
                                    cfg["train"]["seed"], log)
    p = save_table(df, cfg["paths"]["processed_dir"], "ciciomt2024")
    manifest.to_csv(out / "ciciomt2024_files.csv", index=False)
    counts = df.groupby(["label6", "label19", "split"], observed=True).size().unstack("split", fill_value=0)
    counts = counts.reset_index().sort_values(["label6", "label19"])
    counts["total"] = counts[[c for c in ("train", "test") if c in counts]].sum(axis=1)
    write_table(counts, out, "ciciomt2024_class_counts", "CICIoMT2024 Wi-Fi/MQTT flows per class and split")
    log.info(f"CICIoMT2024: {len(df):,} rows, {df['label19'].nunique()} classes, "
             f"{len([c for c in df.columns if c not in ('label19','label6','label2','split','source_file','row_in_file','dataset')])} features -> {p}")


if __name__ == "__main__":
    main()
