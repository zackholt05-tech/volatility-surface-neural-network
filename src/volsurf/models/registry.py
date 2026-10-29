"""Build surface models from config specs: ``{"name": ..., "model": ..., "params": {...}}``."""

from __future__ import annotations

from volsurf.config import from_dict
from volsurf.models.base import SurfaceModel
from volsurf.models.essvi import ESSVISurface
from volsurf.models.mlp import MLPConfig, MLPSurface
from volsurf.models.neural_essvi import NeuralESSVIConfig, NeuralESSVISurface
from volsurf.models.spline import SplineSurface
from volsurf.models.ssvi import SSVISurface
from volsurf.models.svi import SVISurface


def build_model(spec: dict, seed: int) -> SurfaceModel:
    kind, params = spec["model"], dict(spec.get("params") or {})
    # spec-level keys such as name, warm_start and seed_offset are handled by the runner
    if kind == "ssvi":
        return SSVISurface(**params)
    if kind == "essvi":
        return ESSVISurface(**params)
    if kind == "svi":
        return SVISurface(**params)
    if kind == "spline":
        return SplineSurface(**params)
    if kind == "nessvi":
        return NeuralESSVISurface(from_dict(NeuralESSVIConfig, params), seed=seed)
    if kind == "mlp":
        return MLPSurface(from_dict(MLPConfig, params), seed=seed)
    raise ValueError(f"unknown model {kind!r}")
