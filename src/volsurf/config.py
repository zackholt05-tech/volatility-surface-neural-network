"""Small helpers for YAML-driven dataclass configs."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively overlay ``override`` on ``base`` without mutating either."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def from_dict(cls: type, d: dict[str, Any]):
    """Build dataclass ``cls`` from a dict, recursing into nested dataclass fields.

    YAML lists become tuples, matching the tuple-typed fields used in configs.
    """
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise KeyError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    kwargs = {}
    for key, value in d.items():
        hint = hints[key]
        if is_dataclass(hint) and isinstance(value, dict):
            value = from_dict(hint, value)
        elif isinstance(value, list):
            value = tuple(value)
        kwargs[key] = value
    return cls(**kwargs)


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}
