"""
pipeline.py -- the whole Black-and-White Image Trace, start to finish.

Port of ``trace_paths`` / ``Layers::layer`` in ``lib.rs`` (Black-and-White branch).

    RGBA image
      |  1. quantize_bw          luma < threshold -> BLACK, else WHITE, alpha < 128 -> TRANSPARENT
      |  2. denoise              merge regions smaller than ``noise`` px into their neighbour
      |  3. choose layers        one layer per label (black, white); skip white if Ignore White
      |  for each layer:
      |     4. trace_mask        pixel-crack boundary loops (outer boundary + holes per component)
      |     5. subpixel.outline  move every crack to where the anti-aliased edge really is,
      |                          cut the loop at junctions into pieces
      |     6. fit.fit_loop      pieces -> corners restored, lines/arcs snapped, Beziers fitted
      |     7. assemble          outer loop + holes = one path
      v
    TraceResult (list of TracedPath, each a set of closed Bezier subpaths + a colour)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .contour import trace_mask
from .fit import SubPath, fit_loop
from .params import TraceParams
from .quantize import BLACK, TRANSPARENT, WHITE, bw_tones, denoise, luma_float, quantize_bw
from .subpixel import EdgeReader

PALETTE = {WHITE: (255, 255, 255), BLACK: (0, 0, 0)}


@dataclass
class TracedPath:
    """One traced shape: an outer contour plus its holes, filled with ``color``."""
    subpaths: list           # [SubPath, ...]; the first is the outer boundary, the rest are holes
    color: tuple             # (r, g, b)
    pixels: int              # pixel count of the component the path came from


@dataclass
class TraceResult:
    width: int
    height: int
    paths: list = field(default_factory=list)
    palette: list = field(default_factory=list)
    debug: dict = field(default_factory=dict)      # intermediate images, for learning / debugging

    def anchor_count(self) -> int:
        return sum(len(sp.anchors) for p in self.paths for sp in p.subpaths)


def trace(rgba: np.ndarray, params: TraceParams = TraceParams()) -> TraceResult:
    """Trace an ``(h, w, 4)`` uint8 RGBA array. Returns a TraceResult."""
    rgba = np.asarray(rgba)
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError("expected an (h, w, 4) uint8 RGBA array")
    h, w = rgba.shape[:2]
    out = TraceResult(width=w, height=h)
    if h == 0 or w == 0:
        return out

    # ---- STAGE 1: black / white / transparent labels ------------------------------------------
    raw = quantize_bw(rgba, params.threshold)
    # ---- STAGE 2: remove specks ------------------------------------------------------------
    labels = denoise(raw, params.noise)
    out.debug["quantized"] = raw
    out.debug["denoised"] = labels

    # ---- the two reference tones (lightest paper / darkest ink) ------------------------------
    tones = bw_tones(rgba, labels)
    out.debug["tones"] = tones

    # ---- STAGE 3: which layers do we trace, in which order? ---------------------------------
    counts = {lab: int((labels == lab).sum()) for lab in (WHITE, BLACK)}
    order = sorted((lab for lab in (WHITE, BLACK) if counts[lab] > 0), key=lambda lab: (-counts[lab], lab))
    layers = [lab for lab in order if not (params.ignore_white and lab == WHITE)]   # white = palette[0]

    reader = EdgeReader(labels, luma_float(rgba), tones, params.threshold)
    style = params.outline_style()
    across = lambda c: (reader.shift(c), reader.neighbour(c))
    min_hole = max(params.noise, 1)

    for lab in layers:
        ys, xs = np.nonzero(labels == lab)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1   # bounding box
        mask = (labels[y0:y1, x0:x1] == lab)
        # ---- STAGE 4: boundary loops ---------------------------------------------------------
        comps = trace_mask(mask)
        shift_pt = lambda loop: type(loop)([(px + x0, py + y0) for px, py in loop.pts], loop.area2)
        for comp in comps:
            outer = shift_pt(comp.outer)
            holes = [shift_pt(hl) for hl in comp.holes]
            # ---- STAGES 5+6: sub-pixel placement and curve fitting ---------------------------
            fitted = fit_loop(outer, w, h, across, reader.junction, style)
            if fitted is None:
                continue
            subs = [fitted[0]]
            for hole in holes:
                if abs(hole.area2) // 2 < min_hole and params.noise > 0:
                    continue                                   # a speck of paper inside the shape
                fh = fit_loop(hole, w, h, across, reader.junction, style)
                if fh is not None:
                    subs.append(fh[0])
            # ---- STAGE 7: outer + holes = one path ---------------------------------------
            out.paths.append(TracedPath(subs, PALETTE[lab], comp.pixels))
        if PALETTE[lab] not in out.palette:
            out.palette.append(PALETTE[lab])
    return out
