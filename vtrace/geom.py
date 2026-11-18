"""
geom.py -- small geometry helpers used by the curve fitter.

Port of ``crates/trace/src/logo/geom.rs``.

Conventions used everywhere in this package
-------------------------------------------
* A **polyline** is a NumPy array of shape ``(n, 2)`` holding ``(x, y)`` points.
  x grows to the right, y grows DOWN (image coordinates).
* A **cubic Bezier** is a NumPy array of shape ``(4, 2)``:
  ``[start, handle1, handle2, end]``.
* A **closed** polyline does not repeat its first point at the end; the
  closing edge from the last point back to the first is implicit.

The helpers here fall into four groups:

1. resampling   -- turn an irregular polyline into evenly spaced points
2. smoothing    -- Gaussian smoothing of closed / open polylines
3. corners      -- measure the turning angle and find corner points
4. primitives   -- straight-line fit, circle fit, arc -> Bezier, Bezier sampling
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

#: No step ever builds an outline longer than this many points (a safety valve).
MAX_POINTS = 400_000


def _round_half_up(x: float) -> int:
    """Rust's ``f64::round`` for positive numbers (Python's round() is banker's)."""
    return int(math.floor(x + 0.5))


def cross2(a: np.ndarray, b: np.ndarray):
    """2-D cross product ``a.x*b.y - a.y*b.x`` (works on arrays of vectors too)."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def dot2(a: np.ndarray, b: np.ndarray):
    """2-D dot product (works on arrays of vectors too)."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def bez(c: np.ndarray, t) -> np.ndarray:
    """Evaluate cubic Bezier ``c`` at parameter(s) ``t`` (scalar or 1-D array)."""
    t = np.asarray(t, dtype=float)
    u = 1.0 - t
    w0, w1, w2, w3 = u ** 3, 3 * u * u * t, 3 * u * t * t, t ** 3
    return w0[..., None] * c[0] + w1[..., None] * c[1] + w2[..., None] * c[2] + w3[..., None] * c[3]


# --------------------------------------------------------------------------- #
# 1. Resampling
# --------------------------------------------------------------------------- #
def resample_closed(p: np.ndarray, step: float):
    """Resample a CLOSED polyline at (about) ``step`` spacing.

    Returns ``(points, exact_spacing)`` or ``None`` if the loop is too small.

    Why: the traced outline is a staircase of unevenly spaced points. Smoothing
    and corner detection both want *evenly* spaced samples so that "5 samples"
    always means the same physical distance.
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n < 3 or not (step > 0) or not np.isfinite(p).all():
        return None
    seg = np.hypot(*(np.roll(p, -1, axis=0) - p).T)          # edge lengths, incl. closing edge
    cum = np.concatenate([[0.0], np.cumsum(seg)])            # distance travelled at each vertex
    total = cum[-1]
    if not total > step:
        return None
    m = min(max(int(total / step), 8), MAX_POINTS)           # number of output samples
    s = total * np.arange(m) / m                             # arc-length position of each sample
    j = np.clip(np.searchsorted(cum, s, side="left") - 1, 0, n - 1)   # segment each sample falls on
    c0, c1 = cum[j], cum[j + 1]
    d = c1 - c0
    t = np.where(d > 0, (s - c0) / np.where(d > 0, d, 1.0), 0.0)
    a, b = p[j % n], p[(j + 1) % n]
    return a + (b - a) * t[:, None], total / m


def resample_open(p: np.ndarray, step: float):
    """Resample an OPEN polyline at (about) ``step`` spacing, keeping both ends exactly."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n < 2 or not (step > 0) or not np.isfinite(p).all():
        return None
    seg = np.hypot(*(p[1:] - p[:-1]).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if not total > 0:
        return None
    m = min(max(_round_half_up(total / step), 1), MAX_POINTS)
    s = total * np.arange(1, m) / m
    j = np.clip(np.searchsorted(cum, s, side="left") - 1, 0, n - 2)
    c0, c1 = cum[j], cum[j + 1]
    d = c1 - c0
    t = np.where(d > 0, (s - c0) / np.where(d > 0, d, 1.0), 0.0)
    mid = p[j] + (p[j + 1] - p[j]) * t[:, None]
    out = np.vstack([p[:1], mid.reshape(-1, 2), p[-1:]])
    return out, total / m


# --------------------------------------------------------------------------- #
# 2. Smoothing
# --------------------------------------------------------------------------- #
def _kernel(sigma: float) -> np.ndarray:
    """Normalised Gaussian kernel, radius = 3 sigma."""
    s = max(sigma, 0.3)
    r = max(int(math.ceil(3.0 * s)), 1)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
    return k / k.sum()


def _kernel_of_radius(sigma: float, r: int) -> np.ndarray:
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return k / k.sum()


def smooth_closed(p: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of a closed polyline (sigma measured in SAMPLES).

    Each output point is a weighted average of its neighbours along the loop
    (wrapping around the ends). This removes the pixel-grid staircase wobble.
    """
    n = len(p)
    if n < 3:
        return p.copy()
    k = _kernel(sigma)
    r = (len(k) - 1) // 2
    idx = (np.arange(n)[:, None] + np.arange(-r, r + 1)[None, :]) % n
    return (p[idx] * k[None, :, None]).sum(axis=1)


def smooth_open(p: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of an OPEN run with both ends pinned in place.

    The run is first extended past each end by *odd reflection*
    (``end + (end - neighbour)``) so the smoothing keeps the tangent direction
    at the ends instead of pulling them inward. The two end points are then
    restored to their exact original positions.
    """
    n = len(p)
    if n < 5:
        return p.copy()
    s = max(min(max(sigma, 0.4), n / 8.0), 0.4)
    k = _kernel(s)
    r = (len(k) - 1) // 2
    r_ext = min(r, n - 2)
    first, last = p[0], p[-1]
    pre = np.array([first + (first - p[i]) for i in range(r_ext, 0, -1)]).reshape(-1, 2)
    post = np.array([last + (last - p[n - 1 - i]) for i in range(1, r_ext + 1)]).reshape(-1, 2)
    ext = np.vstack([pre, p, post])
    kk = k if r_ext == r else _kernel_of_radius(s, r_ext)
    idx = np.arange(n)[:, None] + np.arange(len(kk))[None, :]
    out = (ext[idx] * kk[None, :, None]).sum(axis=1)
    out[0], out[-1] = p[0], p[-1]
    return out


# --------------------------------------------------------------------------- #
# 3. Corners
# --------------------------------------------------------------------------- #
def turning(p: np.ndarray, w: int) -> np.ndarray:
    """Turning angle (degrees, 0..180) at each point of a CLOSED polyline.

    The angle is measured between the vector arriving from ``w`` samples back
    and the vector leaving to ``w`` samples ahead. A big window ignores tiny
    wiggles and only reacts to real bends.
    """
    n = len(p)
    i = np.arange(n)
    a = p[(i + n - w % n) % n]
    b = p[(i + w) % n]
    v1, v2 = p - a, b - p
    return np.degrees(np.abs(np.arctan2(cross2(v1, v2), dot2(v1, v2))))


def find_corners(p: np.ndarray, w: int, thr: float) -> list:
    """Indices of corners of a closed polyline.

    A corner is a *local maximum* of the turning angle that is at least ``thr``
    degrees, and at least ``w`` samples away from the previous corner.
    """
    n = len(p)
    if n < 8 or w == 0:
        return []
    ang = turning(p, w)
    prev, nxt = np.roll(ang, 1), np.roll(ang, -1)
    cand = np.flatnonzero((ang >= thr) & (ang >= prev) & (ang > nxt))
    idx: list = []
    for i in cand:
        if not idx or i - idx[-1] > w:
            idx.append(int(i))
    if len(idx) > 1 and idx[0] + n - idx[-1] <= w:      # first & last wrap around -> same corner
        idx.pop()
    return idx


# --------------------------------------------------------------------------- #
# 4. Primitives: line fit, circle fit, arcs, sampling
# --------------------------------------------------------------------------- #
def line_fit(pts: np.ndarray):
    """Best straight line through ``pts`` (total least squares).

    Returns ``(centroid, unit_direction, max_distance_of_any_point)`` or None.
    The direction points from the first to the last point.
    """
    n = len(pts)
    if n < 2:
        return None
    c = pts.mean(axis=0)
    v = pts - c
    sxx, sxy, syy = (v[:, 0] ** 2).sum(), (v[:, 0] * v[:, 1]).sum(), (v[:, 1] ** 2).sum()
    ang = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    d = np.array([math.cos(ang), math.sin(ang)])
    if dot2(d, pts[-1] - pts[0]) < 0:
        d = -d
    nrm = np.array([-d[1], d[0]])
    resid = float(np.abs(v @ nrm).max())
    return (c, d, resid) if math.isfinite(resid) else None


def circle_fit(pts: np.ndarray):
    """Algebraic (Kasa) least-squares circle through ``pts``.

    Returns ``(centre, radius, max_radial_error)`` or None if degenerate.
    Solves  2*cx*x + 2*cy*y + c = x^2 + y^2  in the least-squares sense.
    """
    if len(pts) < 3:
        return None
    m = pts.mean(axis=0)
    q = pts - m                                   # centre the data for numerical stability
    rows = np.column_stack([2 * q[:, 0], 2 * q[:, 1], np.ones(len(q))])
    rhs = q[:, 0] ** 2 + q[:, 1] ** 2
    a = rows.T @ rows
    b = rows.T @ rhs
    if abs(np.linalg.det(a)) < 1e-12:
        return None
    s = np.linalg.solve(a, b)
    r2 = s[2] + s[0] ** 2 + s[1] ** 2
    if not r2 > 1e-12:
        return None
    c = np.array([s[0] + m[0], s[1] + m[1]])
    r = math.sqrt(r2)
    resid = float(np.abs(np.hypot(*(pts - c).T) - r).max())
    return (c, r, resid) if math.isfinite(resid) and math.isfinite(r) else None


def arc_cubics(c: np.ndarray, r: float, a0: float, a1: float) -> list:
    """Approximate the circular arc (centre ``c``, radius ``r``, angles a0 -> a1) with Beziers.

    One cubic per quarter turn; the handle length ``4/3*tan(da/4)*r`` is the
    standard constant that makes a cubic hug a circle.
    """
    n = int(min(max(math.ceil(abs(a1 - a0) / (math.pi / 2)), 1), 64))
    da = (a1 - a0) / n
    k = 4.0 / 3.0 * math.tan(da / 4.0)
    out = []
    for j in range(n):
        s, e = a0 + j * da, a0 + (j + 1) * da
        p0 = np.array([c[0] + r * math.cos(s), c[1] + r * math.sin(s)])
        p3 = np.array([c[0] + r * math.cos(e), c[1] + r * math.sin(e)])
        out.append(np.array([
            p0,
            p0 + np.array([-math.sin(s), math.cos(s)]) * (k * r),
            p3 - np.array([-math.sin(e), math.cos(e)]) * (k * r),
            p3,
        ]))
    return out


def sample_cubics(cs: list, spacing: float) -> np.ndarray:
    """Points along a chain of cubics, about ``spacing`` apart (used to *check* fits)."""
    out = []
    for c in cs:
        length = np.hypot(*(c[1] - c[0])) + np.hypot(*(c[2] - c[1])) + np.hypot(*(c[3] - c[2]))
        n = int(min(max(int(length / max(spacing, 1e-3)), 8), 20_000))
        out.append(bez(c, np.arange(n + 1) / n))
        if sum(len(o) for o in out) > MAX_POINTS * 4:
            break
    return np.vstack(out) if out else np.zeros((0, 2))


def all_within(a: np.ndarray, b: np.ndarray, lim: float) -> bool:
    """True when every point of ``a`` has a point of ``b`` within ``lim``.

    (The Rust code uses a hash grid; a KD-tree gives the same answer.)
    """
    if not (lim > 0):
        return False
    if len(a) == 0:
        return True
    if len(b) == 0:
        return False
    d, _ = cKDTree(b).query(a, distance_upper_bound=lim * (1 + 1e-12))
    return bool(np.isfinite(d).all())
