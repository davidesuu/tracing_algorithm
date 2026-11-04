"""
svg.py -- write a TraceResult as an SVG file.

Every TracedPath becomes one <path> whose ``d`` attribute holds all of its subpaths
(outer boundary first, then the holes). Holes run in the opposite direction to the outline,
so the default ``fill-rule="nonzero"`` cancels them out: a hole really is a hole.
"""
from __future__ import annotations

import numpy as np

from .pipeline import TraceResult


def _num(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _pt(p) -> str:
    return f"{_num(float(p[0]))} {_num(float(p[1]))}"


def subpath_d(sp) -> str:
    """SVG path data for one closed subpath: ``M`` first anchor, then ``L``/``C`` to each next anchor."""
    a = sp.anchors
    n = len(a)
    parts = [f"M{_pt(a[0].p)}"]
    for i in range(n if sp.closed else n - 1):
        p, q = a[i], a[(i + 1) % n]
        if np.array_equal(p.h_out, p.p) and np.array_equal(q.h_in, q.p):
            parts.append(f"L{_pt(q.p)}")                       # no handles -> straight line
        else:
            parts.append(f"C{_pt(p.h_out)} {_pt(q.h_in)} {_pt(q.p)}")
    if sp.closed:
        parts.append("Z")
    return "".join(parts)


def to_svg(result: TraceResult, background: str | None = None) -> str:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{result.width}" height="{result.height}" '
           f'viewBox="0 0 {result.width} {result.height}">']
    if background:
        out.append(f'<rect width="100%" height="100%" fill="{background}"/>')
    for tp in result.paths:
        d = "".join(subpath_d(sp) for sp in tp.subpaths)
        r, g, b = tp.color
        out.append(f'<path fill="#{r:02x}{g:02x}{b:02x}" d="{d}"/>')
    out.append("</svg>")
    return "\n".join(out)


def save_svg(result: TraceResult, path: str, background: str | None = None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_svg(result, background))
