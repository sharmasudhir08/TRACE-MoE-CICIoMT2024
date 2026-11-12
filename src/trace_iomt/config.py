"""Load a YAML config and apply --set key=value overrides."""
import argparse
import copy
from pathlib import Path

import yaml


def _parse_value(text):
    # YAML parsing turns "5" into 5, "true" into True, "[a, b]" into a list.
    return yaml.safe_load(text)


def apply_overrides(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return cfg


ROOT = Path(__file__).resolve().parents[2]      # the trace-iomt project folder


def _deep_merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path, overrides=None):
    """Read a YAML config. A config may name a parent with `base: default.yaml` and change only some keys.
    Relative paths under `paths:` are resolved against the project folder."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if cfg.get("base"):
        parent = load_config(path.parent / cfg.pop("base"))
        cfg = _deep_merge(parent, cfg)
    cfg = apply_overrides(cfg, overrides)
    for k, v in cfg.get("paths", {}).items():
        if v and not Path(v).is_absolute():
            cfg["paths"][k] = str((ROOT / v).resolve())
    return cfg


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs" / "default.yaml"))
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a config value, e.g. --set train.epochs=5 (repeatable)")
    return p
