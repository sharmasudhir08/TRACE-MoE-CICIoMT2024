"""Load the raw CICIoMT2024 Wi-Fi/MQTT files into one table and cache it as Parquet."""
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import binary_label, ciciomt_category, ciciomt_label_from_filename

META_COLS = ("label19", "label6", "label2", "split", "source_file", "row_in_file", "dataset")


def feature_cols(df):
    return [c for c in df.columns if c not in META_COLS]


def _read(path):
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def _contiguous_block(n, max_rows, rng):
    """A contiguous block (not a random sample) keeps consecutive flows together, so windows stay meaningful."""
    if not max_rows or n <= max_rows:
        return 0, n
    start = int(rng.integers(0, n - max_rows + 1))
    return start, start + max_rows


def load_ciciomt2024(csv_dir, max_rows_per_file=None, seed=42, logger=None):
    """Read every <Class><n>_<train|test>.pcap.csv under csv_dir. Returns (table, per-file manifest)."""
    root = Path(csv_dir)
    if not root.exists():
        raise FileNotFoundError(f"CICIoMT2024 folder not found: {root}. Set paths.ciciomt_csv_dir "
                                f"(see DATASETS.md for the expected layout).")
    files = sorted(p for p in root.rglob("*") if p.is_file() and ciciomt_label_from_filename(p.name))
    if not files:
        raise FileNotFoundError(f"No files named <Class><n>_<train|test>.pcap.csv under {root}")
    rng = np.random.default_rng(seed)
    frames, manifest, features = [], [], None
    for p in files:
        label19, split = ciciomt_label_from_filename(p.name)
        d = _read(p)
        d.columns = [c.strip() for c in d.columns]
        if features is None:
            features = list(d.columns)
        missing = [c for c in features if c not in d.columns]
        if missing:
            raise ValueError(f"{p.name} lacks columns {missing}; all files must share the 45 CICIoMT2024 features")
        n = len(d)
        a, b = _contiguous_block(n, max_rows_per_file, rng)
        d = d.iloc[a:b][features].astype(np.float32).reset_index(drop=True)
        d["row_in_file"] = np.arange(a, b, dtype=np.int64)
        d["label19"], d["split"], d["source_file"] = label19, split, p.name
        frames.append(d)
        manifest.append({"file": p.name, "label19": label19, "split": split, "rows": n, "rows_kept": b - a})
        if logger:
            logger.info(f"{p.name}: {n:,} rows ({b - a:,} kept) -> {label19} / {split}")
    df = pd.concat(frames, ignore_index=True)
    df["label6"] = df["label19"].map(ciciomt_category)
    df["label2"] = df["label6"].map(binary_label)
    df["dataset"] = "CICIoMT2024"
    for c in ("label19", "label6", "label2", "split", "source_file", "dataset"):
        df[c] = df[c].astype("category")
    return df, pd.DataFrame(manifest)


def save_table(df, processed_dir, name):
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.parquet"
    df.to_parquet(p, index=False)
    return p


def load_table(processed_dir, name):
    p = Path(processed_dir) / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run scripts/01_prepare.py first")
    return pd.read_parquet(p)
