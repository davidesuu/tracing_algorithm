"""
vtrace -- a documented Python port of VectorCraft's Black-and-White Image Trace.

Quick start::

    from vtrace import trace_image, preset, save_svg
    result = trace_image("logo.png", preset("Silhouettes"))
    save_svg(result, "logo.svg")
"""
from .params import PRESETS, TraceParams, preset, with_overrides
from .pipeline import TracedPath, TraceResult, trace
from .svg import save_svg, to_svg

import numpy as np


def load_rgba(path: str) -> np.ndarray:
    """Read any image file Pillow understands as an (h, w, 4) uint8 RGBA array."""
    from PIL import Image
    return np.array(Image.open(path).convert("RGBA"), dtype=np.uint8)


def trace_image(path: str, params: TraceParams = TraceParams()) -> TraceResult:
    """Convenience: load ``path`` and trace it."""
    return trace(load_rgba(path), params)


__all__ = ["TraceParams", "PRESETS", "preset", "with_overrides", "trace", "trace_image",
           "TraceResult", "TracedPath", "to_svg", "save_svg", "load_rgba"]
