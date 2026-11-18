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


def trace_image(
    path: str,
    params: TraceParams = TraceParams(),
    rectify: bool = False,
    remove_bg: bool = False,
) -> TraceResult:
    """Convenience: load ``path``, optionally preprocess (remove bg / rectify), and trace it."""
    rgba = load_rgba(path)
    if rectify or remove_bg:
        from .preprocess import preprocess_image
        rgba = preprocess_image(rgba, rectify=rectify, remove_bg=remove_bg)
    return trace(rgba, params)


__all__ = [
    "TraceParams", "PRESETS", "preset", "with_overrides", "trace", "trace_image",
    "TraceResult", "TracedPath", "to_svg", "save_svg", "load_rgba"
]
