"""Shared setup for the numbered scripts: puts src/ on the path, reads the config, opens log files.

Every invocation writes to TWO log files:
  - logs/<step>.log                       the current log (overwritten each invocation, for compatibility)
  - logs/sessions/<SESSION>_<step>_<host>.log   append-only session history (never overwritten)

SESSION is one timestamp per run_all.py invocation (via the TRACE_SESSION env var), so all 12 steps
of one pipeline run share the same session id. Stand-alone scripts get their own session id.
"""
import logging
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_iomt.config import base_parser, load_config  # noqa: E402
from trace_iomt.utils import get_logger, session_id  # noqa: E402


def _attach_session_file(logger, logs_dir, log_name):
    sess_dir = Path(logs_dir) / "sessions"
    sess_dir.mkdir(parents=True, exist_ok=True)
    host = platform.node().replace(" ", "_") or "unknown"
    path = sess_dir / f"{session_id()}_{log_name}_{host}.log"
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    return path


def setup(description, log_name, add_args=None):
    p = base_parser(description)
    if add_args:
        add_args(p)
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    logger = get_logger(log_name, cfg["paths"]["logs_dir"], f"{log_name}.log")
    sess_log = _attach_session_file(logger, cfg["paths"]["logs_dir"], log_name)
    logger.info(f"--- {Path(sys.argv[0]).name} | config {args.config} | overrides {args.set}")
    logger.info(f"Session {session_id()} | host {platform.node()} | session log: {sess_log}")
    return args, cfg, logger
