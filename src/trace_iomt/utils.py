"""Seeding, logging, timing, hashing, git/env info and small I/O helpers."""
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# One session identifier per Python process. run_all.py sets TRACE_SESSION once at the top
# so every subprocess (every step) shares the same stamp; stand-alone scripts get their own.
_SESSION = os.environ.get("TRACE_SESSION") or time.strftime("%Y-%m-%d_%H%M%S")


def session_id():
    return _SESSION


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def sha256_file(path, chunk=1 << 20):
    """SHA-256 of one file (returns None if missing or unreadable)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(chunk), b""):
                h.update(b)
        return h.hexdigest()
    except (OSError, FileNotFoundError):
        return None


def sha256_many(paths):
    """Combined, order-independent SHA-256 of several files."""
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        s = sha256_file(p)
        if s:
            h.update(s.encode())
    return h.hexdigest()


_GIT_INFO = None


def git_info(root=None):
    """(short commit, dirty_bool) for the project. Cached per process. ('no-git', False) if git unavailable."""
    global _GIT_INFO
    if _GIT_INFO is not None:
        return _GIT_INFO
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
        rc = subprocess.run(["git", "-C", str(root), "diff", "--quiet"], stderr=subprocess.DEVNULL)
        dirty = rc.returncode != 0
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        sha, dirty = "no-git", False
    _GIT_INFO = (sha, dirty)
    return _GIT_INFO


_CODE_HASH = None


def code_hash(root=None):
    """Combined SHA-256 of every src/trace_iomt/**/*.py file. Cached per process."""
    global _CODE_HASH
    if _CODE_HASH is not None:
        return _CODE_HASH
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    _CODE_HASH = sha256_many(sorted((root / "src" / "trace_iomt").rglob("*.py")))
    return _CODE_HASH


_DATA_HASH_CACHE = {}


def data_hash(path):
    """SHA-256 of a data file; cached per process so step 04 doesn't rehash parquet 20 times."""
    key = str(path)
    if key not in _DATA_HASH_CACHE:
        _DATA_HASH_CACHE[key] = sha256_file(path)
    return _DATA_HASH_CACHE[key]


def env_info():
    env = {"python": platform.python_version(), "os": f"{platform.system()} {platform.release()}",
           "cpu": platform.processor() or platform.machine(), "host": platform.node(),
           "session": session_id()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except ImportError:
        pass
    return env


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name, logs_dir=None, filename=None):
    """Logger that writes to the console and, if logs_dir is given, to logs/<filename>."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logs_dir:
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(logs_dir) / (filename or f"{name}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


@contextmanager
def timer(logger, label):
    t0 = time.perf_counter()
    yield
    logger.info(f"{label} took {time.perf_counter() - t0:.1f}s")


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=NumpyEncoder)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def resolve_device(name="auto"):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_name(model, task_tag, regime, seed, extra=None):
    parts = [model, task_tag, regime]
    if extra:
        parts.append(extra)
    parts.append(f"seed{seed}")
    return "_".join(str(p) for p in parts)
