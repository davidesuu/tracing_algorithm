"""
bezier.py -- least-squares cubic Bezier curve fitting (Schneider's algorithm).

Port of ``crates/trace/src/logo/bezier.rs``.
Reference: P. J. Schneider, "An algorithm for automatically fitting digitized
curves", Graphics Gems (1990).

The problem
-----------
Given a run of points ``P0 ... Pn`` that lie along a smooth curve, find the
FEWEST cubic Beziers that stay within ``tol`` pixels of every point.

The algorithm
-------------
1. Assign each point a parameter ``u`` in [0, 1] by chord length
   (how far along the run it is, as a fraction of the total length).
2. The end points are fixed. The end *tangent directions* are given. The only
   unknowns are how far along each tangent the two inner control points sit
   (``alpha_left`` and ``alpha_right``). That is a tiny 2x2 least-squares
   problem (``fit_one``).
3. Measure the worst error. If it is small enough, done.
4. If it is close, *re-parameterise*: nudge each ``u`` with one Newton step so
   the point maps to the nearest spot on the curve (``reparam``), refit, repeat
   up to 4 times.
5. Otherwise split the run at the worst point and fit each half recursively,
   forcing the two halves to share a tangent at the split so the join is smooth.
"""
from __future__ import annotations

import numpy as np

from .geom import bez, dot2

MAX_DEPTH = 14


def _chord(p: np.ndarray) -> np.ndarray:
    """Chord-length parameterisation: ``u[i]`` = distance along the run / total length."""
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*(p[1:] - p[:-1]).T))])
    return d / d[-1] if d[-1] > 0 else d


def _fit_one(p: np.ndarray, u: np.ndarray, t1: np.ndarray, t2: np.ndarray):
    """Fit ONE cubic to ``p`` given parameters ``u`` and fixed end tangents.

    Control points:  P0, P0 + t1*alpha_l, P3 + t2*alpha_r, P3.
    ``alpha_l`` / ``alpha_r`` come from the normal equations of the least-squares
    problem  sum_i | B(u_i) - p_i |^2  -> min.
    """
    p0, p3 = p[0], p[-1]
    s = 1.0 - u
    b0, b1, b2, b3 = s ** 3, 3 * u * s * s, 3 * u * u * s, u ** 3
    a1 = t1[None, :] * b1[:, None]
    a2 = t2[None, :] * b2[:, None]
    c00, c01, c11 = dot2(a1, a1).sum(), dot2(a1, a2).sum(), dot2(a2, a2).sum()
    tmp = p - (p0[None, :] * (b0 + b1)[:, None] + p3[None, :] * (b2 + b3)[:, None])
    x0, x1 = dot2(a1, tmp).sum(), dot2(a2, tmp).sum()
    det = c00 * c11 - c01 * c01
    seg = float(np.hypot(*(p0 - p3)))
    if abs(det) > 1e-12:
        al, ar = (x0 * c11 - x1 * c01) / det, (c00 * x1 - c01 * x0) / det
    else:
        al = ar = 0.0
    # Degenerate / backwards handles: fall back to the classic 1/3 rule.
    if not (np.isfinite(al) and np.isfinite(ar)) or al <= 1e-6 * seg or ar <= 1e-6 * seg:
        al = ar = seg / 3.0
    return np.array([p0, p0 + t1 * al, p3 + t2 * ar, p3])


def _reparam(c: np.ndarray, p: np.ndarray, u: np.ndarray) -> np.ndarray:
    """One Newton-Raphson step moving each ``u`` towards the closest point on the curve."""
    s = 1.0 - u
    d = bez(c, u) - p
    v0, v1, v2 = c[1] - c[0], c[2] - c[1], c[3] - c[2]
    d1 = (v0[None, :] * (s * s)[:, None] + v1[None, :] * (2 * s * u)[:, None] + v2[None, :] * (u * u)[:, None]) * 3.0
    d2 = ((v1 - v0)[None, :] * s[:, None] + (v2 - v1)[None, :] * u[:, None]) * 6.0
    den = dot2(d1, d1) + dot2(d, d2)
    ok = np.abs(den) > 1e-12
    new = np.where(ok, u - dot2(d, d1) / np.where(ok, den, 1.0), u)
    return np.clip(new, 0.0, 1.0)


def _max_err(c: np.ndarray, p: np.ndarray, u: np.ndarray):
    """(largest distance from a point to its curve position, index of that point)."""
    e = np.hypot(*(bez(c, u) - p).T)
    i = int(np.argmax(e))
    if e[i] <= 0.0:
        return 0.0, len(p) // 2
    return float(e[i]), i


def _rec(p, t1, t2, tol, depth, out):
    if len(p) == 2:                                   # two points -> a straight-ish cubic
        d = float(np.hypot(*(p[0] - p[1]))) / 3.0
        out.append(np.array([p[0], p[0] + t1 * d, p[1] + t2 * d, p[1]]))
        return
    u = _chord(p)
    c = _fit_one(p, u, t1, t2)
    for _ in range(4):                                # fit, then refine parameters up to 4x
        e, _i = _max_err(c, p, u)
        if e < tol:
            out.append(c)
            return
        if e > tol * 6.0:                             # far off: re-parameterising won't save it
            break
        u = _reparam(c, p, u)
        c = _fit_one(p, u, t1, t2)
    e, split = _max_err(c, p, u)
    if e < tol or depth >= MAX_DEPTH or len(p) < 6:
        out.append(c)
        return
    s = min(max(split, 2), len(p) - 3)
    x, y = p[s - 1], p[s + 1]
    dist = float(np.hypot(*(x - y)))
    tc = (x - y) / dist if dist > 1e-12 else t1       # tangent shared by the two halves
    _rec(p[: s + 1], t1, tc, tol, depth + 1, out)
    _rec(p[s:], -tc, t2, tol, depth + 1, out)


def fit_cubics(p: np.ndarray, t1: np.ndarray, t2: np.ndarray, tol: float) -> list:
    """Cubics through ``p`` (first and last points are interpolated).

    ``t1`` points ALONG the run at its start; ``t2`` points BACK along the run
    at its end. Every point ends up within ``tol`` of the result.
    """
    out: list = []
    if len(p) >= 2 and tol > 0 and np.isfinite(p).all():
        _rec(p, np.asarray(t1, float), np.asarray(t2, float), tol, 0, out)
    return out


def tangent(p: np.ndarray, i: int, k: int, sign: int) -> np.ndarray:
    """Unit direction from ``p[i]`` towards ``p[i+k]`` (sign>0) or ``p[i-k]`` (sign<0)."""
    j = min(i + k, len(p) - 1) if sign > 0 else max(i - k, 0)
    d = float(np.hypot(*(p[j] - p[i])))
    return (p[j] - p[i]) / d if d > 1e-12 else np.array([1.0, 0.0])
