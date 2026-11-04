"""
params.py -- the Image Trace panel settings (Black-and-White subset) and the presets.

Port of ``TraceParams``, ``TraceParams::outline_style`` and ``preset()`` in ``lib.rs``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .restore import RestoreParams, Style


@dataclass(frozen=True)
class TraceParams:
    """Settings of a Black-and-White trace.

    threshold            0-255   pixels with luma BELOW this are black.
    paths                0-100   Fidelity. Higher follows the pixels more tightly (more anchors).
                                 Sets the Bezier fit tolerance: 0.78 px at 0 ... 0.08 px at 100.
    corners              0-100   Higher keeps more corners. Sets the turning angle that counts as a
                                 corner: 90 deg at 0 ... 35 deg at 100.
    noise                px      Regions (and holes) smaller than this many pixels are removed.
    ignore_white         bool    Don't output the white shapes (no background rectangle).
    snap_curves_to_lines bool    Near-straight curves become lines, parts of circles become true
                                 arcs, near-level/upright lines become exactly level/upright.
    """
    threshold: int = 128
    paths: float = 50.0
    corners: float = 75.0
    noise: int = 25
    ignore_white: bool = False
    snap_curves_to_lines: bool = False

    def outline_style(self) -> Style:
        """Translate the sliders into the numbers the curve fitter uses."""
        def unit(v: float, default: float) -> float:
            return min(max(v / 100.0, 0.0), 1.0) if math.isfinite(v) else default

        snap = self.snap_curves_to_lines
        return Style(
            step=0.25,
            sigma=0.5,
            corner_deg=90.0 - 55.0 * unit(self.corners, 0.75),
            corner_win=0.7,
            tol=0.08 + 0.7 * (1.0 - unit(self.paths, 0.5)) ** 1.5,
            snap=True,                          # straight edges are ALWAYS lines
            arcs=snap,                          # ...but arcs only with Snap Curves To Lines
            line_tol=0.5 if snap else 0.2,
            circle_tol=0.3 if snap else 0.15,
            restore=RestoreParams(),
            max_dev=1.4,
            check_step=0.15,
            upright=snap,
        )


def _bw(threshold: int, paths: float, corners: float, noise: int, snap: bool) -> TraceParams:
    return TraceParams(threshold=threshold, paths=paths, corners=corners, noise=noise,
                       ignore_white=True, snap_curves_to_lines=snap)


#: Built-in Black-and-White presets, in panel order. (Colour / grayscale / flat-logo presets of
#: the original are not part of this port. 'Outline' needs Create Strokes, also not ported.)
PRESETS = {
    "Default":              TraceParams(),
    "Black and White Logo": _bw(128, 95.0, 80.0, 8, True),
    "Sketched Art":         _bw(150, 50.0, 50.0, 100, False),
    "Silhouettes":          _bw(200, 40.0, 60.0, 30, False),
    "Line Art":             _bw(128, 80.0, 70.0, 10, False),
    "Technical Drawing":    _bw(128, 95.0, 90.0, 4, True),
}


def preset(name: str) -> TraceParams:
    """Look up a preset by name (case-insensitive, ``[Default]`` accepted)."""
    key = name.strip().strip("[]").lower()
    for k, v in PRESETS.items():
        if k.lower() == key:
            return v
    raise KeyError(f"unknown preset {name!r}; choose from {list(PRESETS)}")


def with_overrides(p: TraceParams, **kw) -> TraceParams:
    """A copy of ``p`` with the given fields replaced (ignores None values)."""
    return replace(p, **{k: v for k, v in kw.items() if v is not None})
