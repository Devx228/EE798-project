"""YAML configs with ``section.key=value`` command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("inherit", None)
    if base is not None:
        parent = load_config(Path(path).parent / base)
        cfg = merge(parent, cfg)
    return cfg


def merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, key: str, value) -> None:
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def parse_value(text: str):
    """YAML-parse a command-line value; also accept floats like ``3e-3`` that YAML 1.1 reads as strings."""
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def apply_overrides(cfg: dict, overrides: list[str] | dict) -> dict:
    cfg = copy.deepcopy(cfg)
    items = overrides.items() if isinstance(overrides, dict) else (o.split("=", 1) for o in overrides)
    for key, value in items:
        set_dotted(cfg, key, parse_value(value) if isinstance(value, str) else value)
    return cfg
