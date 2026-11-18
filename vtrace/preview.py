"""
preview.py -- render a TraceResult to a PNG (so you can look at it without an SVG viewer).

Each subpath is sampled into a polygon, and the polygons of one path are combined with
XOR (even-odd), which for non-overlapping outlines + holes gives exactly the same picture as
an SVG viewer would.

Note: PIL fills polygon boundary pixels inclusively, so this preview looks a touch "fatter"
(up to ~1/scale px) than the real vector. Use the SVG for exact geometry.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .geom import sample_cubics
from .pipeline import TraceResult


def subpath_polyline(sp, spacing: float = 0.1) -> np.ndarray:
    """Sample a closed SubPath into a dense polyline of points."""
    n = len(sp.anchors)
    segs = [np.array([sp.anchors[i].p, sp.anchors[i].h_out, sp.anchors[(i + 1) % n].h_in, sp.anchors[(i + 1) % n].p])
            for i in range(n)]
    return sample_cubics(segs, spacing)


def render_png(result: TraceResult, scale: int = 4, background=(255, 255, 255)) -> Image.Image:
    W, H = result.width * scale, result.height * scale
    canvas = Image.new("RGB", (W, H), background)
    for tp in result.paths:
        mask = np.zeros((H, W), dtype=bool)
        for sp in tp.subpaths:
            layer = Image.new("1", (W, H), 0)
            ImageDraw.Draw(layer).polygon([tuple(q * scale) for q in subpath_polyline(sp)], fill=1)
            mask ^= np.array(layer, dtype=bool)                  # even-odd: holes cancel
        arr = np.array(canvas)
        arr[mask] = tp.color
        canvas = Image.fromarray(arr)
    return canvas
