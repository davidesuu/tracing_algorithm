"""
outline_fit.py -- turn ONE outline (a list of sub-pixel points) into Bezier curves.

Port of ``crates/trace/src/logo/outline.rs``.

For a CLOSED outline (``fit_outline``) the steps are:

    raw points
      -> resample evenly                     (geom.resample_closed)
      -> light smoothing                     (geom.smooth_closed)
      -> restore rounded corners             (restore.restore_corners)
      -> find corners                        (geom.find_corners)
      -> for every run BETWEEN corners:      (fit_run)
            * almost straight?   -> one exact line
            * part of a circle?  -> exact arc(s)      [only with Snap Curves To Lines]
            * otherwise          -> Schneider Bezier fit
      -> VERIFY the result stays within ``max_dev`` of the raw contour,
         both ways; if not, retry without the clever parts, then fall back
         to a plain fit.

An OPEN run (``fit_open``) is the same minus corner restoration; both of its
end points are kept exactly (they are junctions shared with other shapes).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .bezier import fit_cubics, tangent
from .geom import (
    _round_half_up, all_within, arc_cubics, circle_fit, cross2, dot2, find_corners,
    resample_closed, resample_open, sample_cubics, smooth_closed, smooth_open,
)
from .restore import Style, restore_corners


@dataclass
class Stats:
    """What fitting did (handy for debugging / tests)."""
    restored: int = 0
    lines: int = 0
    arcs: int = 0
    circles: int = 0
    reverted: int = 0


def line_cubic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A straight segment written as a cubic (handles at 1/3 and 2/3)."""
    d = b - a
    return np.array([a, a + d / 3.0, a + d * (2.0 / 3.0), b])


def _sweep(pts: np.ndarray, c: np.ndarray):
    """Angles (start, end) swept going from the first to the last point around centre ``c``.

    Continuous (may exceed pi, may be negative) so we know which way round the arc goes.
    """
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    d = np.diff(ang)
    d = (d + math.pi) % (2 * math.pi) - math.pi
    return float(ang[0]), float(ang[0] + d.sum())


def plain_fit(p: np.ndarray, tol: float) -> list:
    """Last-resort fit of the raw closed contour: no smoothing, corners or snapping."""
    n = len(p)
    if n < 3:
        return []
    t = p[1] - p[-1]
    ln = float(np.hypot(*t))
    if not ln > 1e-12:
        return []
    t = t / ln
    return fit_cubics(np.vstack([p, p[:1]]), t, -t, tol)


def fit_run(run: np.ndarray, sg: float, st: Style, stats: Stats, segs: list) -> None:
    """Fit ONE run (between two corners) and append its cubics to ``segs``.

    Decision order: line -> arc -> general Bezier.
    """
    if len(run) < 4:                                   # tiny run: a straight edge between its ends
        if len(run) >= 2 and np.hypot(*(run[0] - run[-1])) > 1e-9:
            segs.append(line_cubic(run[0], run[-1]))
        return
    sm = smooth_open(run, sg)
    a, b = sm[0], sm[-1]
    length = float(np.hypot(*(a - b)))

    # --- straight? every smoothed sample within line_tol of the chord a->b ----------------------
    if st.snap and length > 1.5:
        d = b - a
        nrm = np.array([-d[1], d[0]]) / length
        if np.abs((sm - a) @ nrm).max() < st.line_tol:
            stats.lines += 1
            segs.append(line_cubic(a, b))
            return

    # --- circular? (only with Snap Curves To Lines: st.arcs) ------------------------------------
    if st.snap and st.arcs and len(sm) > 12 and length > 1.0:
        cf = circle_fit(sm)
        if cf is not None and cf[2] < st.circle_tol and 2.5 < cf[1] < 400.0:
            a0, a1 = _sweep(sm, cf[0])
            if abs(a1 - a0) > 0.35:
                stats.arcs += 1
                first = len(segs)
                segs.extend(arc_cubics(cf[0], cf[1], a0, a1))
                segs[first] = segs[first].copy()
                segs[first][0] = a                     # end exactly on the corners, not on the circle
                segs[-1] = segs[-1].copy()
                segs[-1][3] = b
                return

    # --- general: Schneider Bezier fit ---------------------------------------------------------
    segs.extend(fit_cubics(sm, tangent(sm, 0, 4, 1), tangent(sm, len(sm) - 1, 4, -1), st.tol))


def _sane(segs: list) -> bool:
    return len(segs) > 0 and all(np.isfinite(c).all() for c in segs)


def fit_open(p: np.ndarray, st: Style, stats: Stats) -> list:
    """An OPEN run as cubics from its first point to its last (both kept exactly)."""
    first, last = p[0], p[-1]
    segs: list = []

    def whole():
        if np.hypot(*(first - last)) > 1e-9:
            segs.append(line_cubic(first, last))

    rs = resample_open(p, st.step)
    if rs is None:
        whole()
        return segs
    r, step = rs
    n = len(r)
    s0 = smooth_open(r, max(0.18 / step, 1.0))
    w = max(_round_half_up(st.corner_win / step), 2)

    # Turning angle at every sample (0 where the window would fall off the ends).
    angle = np.zeros(n)
    if n > 2 * w:
        i = np.arange(w, n - w)
        v1, v2 = s0[i] - s0[i - w], s0[i + w] - s0[i]
        angle[i] = np.degrees(np.abs(np.arctan2(cross2(v1, v2), dot2(v1, v2))))
    cuts = [0]
    for i in range(w, max(n - w, w)):
        here = angle[i]
        if here >= st.corner_deg and here >= angle[i - 1] and here > angle[i + 1] and i - cuts[-1] > w:
            cuts.append(i)
    cuts.append(n - 1)

    sg = st.sigma / step
    for a, b in zip(cuts[:-1], cuts[1:]):
        run = r[a: b + 1]
        before = len(segs)
        fit_run(run, sg, st, stats, segs)
        if len(segs) == before and np.hypot(*(run[0] - run[-1])) > 1e-9:   # every stretch must lead on
            segs.append(line_cubic(run[0], run[-1]))
    if not _sane(segs):
        segs.clear()
        whole()
    return segs


def styled_fit(p: np.ndarray, st: Style, stats: Stats):
    """The 'clever' closed-outline fit (see module docstring). None if it can't."""
    rs = resample_closed(p, st.step)
    if rs is None:
        return None
    r, step = rs
    sg = st.sigma / step
    light = max(0.18 / step, 1.0)
    s0 = smooth_closed(r, light)
    forced: list = []
    if st.restore is not None:
        nr, f = restore_corners(r, s0, step, st.restore)
        if f:
            stats.restored += len(f)
            s0 = nr.copy()
            r = nr
            forced = f
    w = max(_round_half_up(st.corner_win / step), 2)
    corners = find_corners(s0, w, st.corner_deg)
    corners = sorted(set(corners) | set(forced))
    n = len(r)

    # --- no corners at all: a smooth closed loop (maybe a perfect circle) ------------------------
    if not corners:
        s = smooth_closed(r, sg)
        if st.snap:
            cf = circle_fit(s)
            if cf is not None and cf[2] < st.circle_tol and cf[1] > 2.5:
                a0, a1 = _sweep(s, cf[0])
                direction = 1.0 if a1 >= a0 else -1.0     # keep the loop's own winding direction
                stats.circles += 1
                return arc_cubics(cf[0], cf[1], a0, a0 + direction * 2 * math.pi)
        t = s[1] - s[-1]
        ln = float(np.hypot(*t))
        if not ln > 1e-12:
            return None
        t = t / ln
        return fit_cubics(np.vstack([s, s[:1]]), t, -t, st.tol)

    # --- corners: fit each run between consecutive corners -------------------------------------
    segs: list = []
    for ci, i0 in enumerate(corners):
        i1 = corners[(ci + 1) % len(corners)]
        count = (i1 + n - i0 if i1 <= i0 else i1 - i0) + 1
        run = r[(i0 + np.arange(count)) % n]
        fit_run(run, sg, st, stats, segs)
    return segs or None


def fit_outline(p: np.ndarray, st: Style, stats: Stats):
    """A CLOSED outline as cubics, or None if it is degenerate.

    Includes the safety net: a fit that wanders more than ``st.max_dev`` from
    the raw contour is thrown away and retried without snapping/restoration;
    if that wanders too, a plain fit of the raw points is used.
    """
    if len(p) < 3 or not np.isfinite(p).all():
        return None
    local = Stats()
    rs = resample_closed(p, st.check_step / 0.3)
    reference = rs[0] if rs else p                      # raw outline, densified for the check

    def checked(style: Style, loc: Stats):
        s = styled_fit(p, style, loc)
        if s is None or not _sane(s):
            return None
        samples = sample_cubics(s, style.check_step)
        if all_within(samples, reference, style.max_dev) and all_within(reference, samples, style.max_dev):
            return s
        return None

    fitted = checked(st, local)
    if fitted is None:
        local = Stats()
        fitted = checked(replace(st, snap=False, restore=None), local)
    if fitted is not None:
        stats.restored += local.restored
        stats.lines += local.lines
        stats.arcs += local.arcs
        stats.circles += local.circles
        return fitted
    stats.reverted += 1
    pf = plain_fit(p, st.tol)
    return pf if _sane(pf) else None
