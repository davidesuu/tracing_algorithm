"""
fit.py -- fit one traced LOOP (outer boundary or hole) into a closed Bezier subpath.

Port of ``crates/trace/src/fit.rs``.

Where this sits in the pipeline
-------------------------------
    contour.trace_mask   ->  pixel-corner loops (staircases)
    subpixel.outline     ->  loop cut into PIECES, each placed to a fraction of a pixel
    fit.fit_loop (here)  ->  every piece fitted to Beziers, glued into one closed subpath

Why "pieces"?
    Two regions that touch share an edge. Both must receive *exactly* the same
    curve for that edge, otherwise a hairline gap or overlap opens between
    them. So the loop is cut wherever the neighbour changes ("junctions"), and
    each stretch is fitted on its own, in a canonical direction (always
    starting from its left-most end) so both regions compute identical curves.
    A loop with no junctions is one closed piece.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .bezier import fit_cubics, tangent
from .geom import all_within, dot2, resample_closed, resample_open, sample_cubics
from .outline_fit import Stats, fit_open, fit_outline
from .restore import Style
from .subpixel import outline as subpixel_outline

# Lines within this of level/upright are made exactly so (tangent of 2.5 degrees).
NEARLY_UPRIGHT = 0.0437
# The farthest (px) levelling a line may move either of its ends.
LEVEL_REACH = 1.6


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _seg_dist(p, a, b) -> float:
    """Distance from point ``p`` to the segment ``a``-``b``."""
    ab = b - a
    l2 = float(ab @ ab)
    if l2 < 1e-18:
        return float(np.hypot(*(p - a)))
    t = min(max(float((p - a) @ ab) / l2, 0.0), 1.0)
    return float(np.hypot(*(p - (a + ab * t))))


def _is_straight(c: np.ndarray) -> bool:
    return _seg_dist(c[1], c[0], c[3]) < 1e-6 and _seg_dist(c[2], c[0], c[3]) < 1e-6


def _rdp(pts: np.ndarray, a: int, b: int, tol: float, keep: list) -> None:
    """Douglas-Peucker line simplification on ``pts[a..=b]``, marking kept indices."""
    stack = [(a, b)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        pa, pb = pts[a], pts[b % len(pts)]
        best, bi = 0.0, a
        for i in range(a + 1, b):
            d = _seg_dist(pts[i], pa, pb)
            if d > best:
                best, bi = d, i
        if best > tol:
            keep[bi] = True
            stack.append((a, bi))
            stack.append((bi, b))


def _reversed(segs: list) -> list:
    """The same curve run backwards: reverse the chain AND each cubic."""
    return [c[::-1].copy() for c in reversed(segs)]


def _before_key(p):
    """Canonical ordering of points: left-most first, ties broken by smaller y."""
    return (float(p[0]), float(p[1]))


def _polygon(pts: np.ndarray, tol: float) -> list:
    """The closed outline as straight pieces within ``tol`` of it (Douglas-Peucker)."""
    n = len(pts)
    if n == 0:
        return []
    d2 = ((pts - pts[0]) ** 2).sum(axis=1)
    far = int(np.flatnonzero(d2 == d2.max())[-1])
    ext = np.vstack([pts, pts[:1]])
    keep = [False] * (n + 1)
    keep[0] = keep[far] = True
    _rdp(ext, 0, far, tol, keep)
    _rdp(ext, far, n, tol, keep)
    kept = [pts[i] for i in range(n) if keep[i]]
    m = len(kept)
    return [np.array([kept[i], kept[i], kept[(i + 1) % m], kept[(i + 1) % m]]) for i in range(m)]


def cubic_signed_area(c: np.ndarray) -> float:
    """Signed area under a cubic (kurbo's closed form); summed over a closed chain = enclosed area."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = c
    return (x0 * (6 * y1 + 3 * y2 + y3)
            + 3 * (x1 * (-2 * y0 + y2 + y3) - x2 * (y0 + y1 - 2 * y3))
            - x3 * (y0 + 3 * y1 + 6 * y2)) / 20.0


def _chain_area(segs: list, origin: np.ndarray) -> float:
    # Translate to a nearby origin first: avoids cancellation error far from (0, 0).
    return sum(cubic_signed_area(c - origin) for c in segs)


# --------------------------------------------------------------------------- #
# Snap Curves To Lines: make near-level / near-upright straight pieces exact
# --------------------------------------------------------------------------- #
def snap_upright(segs: list, closed: bool, fixed_ends: bool) -> None:
    """Straight pieces within 2.5 degrees of level or upright, at least 2 px long, are
    made EXACTLY level/upright by moving both ends to their mean.

    (Unless that would move an end more than LEVEL_REACH px -- a long line a degree
    or two off is that line, not a level one drawn badly.) With ``fixed_ends`` the
    ends of an open chain stay put: they are junctions other shapes share.
    """
    n = len(segs)
    lines = [_is_straight(c) for c in segs]
    fixed = lambda i: (not closed) and fixed_ends and (i == 0 or i == n)

    def at(i):                                         # anchor i starts piece i; an open chain has one more
        return segs[i][0] if i < n else segs[n - 1][3]

    def put(i, p):                                     # move anchor i, dragging its neighbours' handles along
        i = i % n if closed else i
        d = p - at(i)
        if i < n:
            segs[i] = segs[i].copy()
            segs[i][0] = p
            segs[i][1] = segs[i][1] + d
        before = i - 1 if i > 0 else (n - 1 if closed else None)
        if before is not None:
            segs[before] = segs[before].copy()
            segs[before][3] = p
            segs[before][2] = segs[before][2] + d

    for i, line in enumerate(lines):
        ia, ib = i, ((i + 1) % n if closed else i + 1)
        a, b = at(ia), at(ib)
        d = b - a
        if not line or np.hypot(*d) < 2.0 or ia == ib:
            continue
        level = abs(d[1]) <= NEARLY_UPRIGHT * abs(d[0])
        if not level and abs(d[0]) > NEARLY_UPRIGHT * abs(d[1]):
            continue                                   # neither level nor upright
        pick = (lambda q: q[1]) if level else (lambda q: q[0])
        fa, fb = fixed(ia), fixed(ib)
        if fa and fb:
            continue
        to = pick(a) if fa else pick(b) if fb else (pick(a) + pick(b)) / 2.0
        if max(abs(pick(a) - to), abs(pick(b) - to)) > LEVEL_REACH:
            continue
        moved = (lambda q: np.array([q[0], to])) if level else (lambda q: np.array([to, q[1]]))
        if not fa:
            put(ia, moved(a))
        if not fb:
            put(ib, moved(b))
    # A straight piece has its handles ON it.
    for k, line in enumerate(lines):
        if line:
            c = segs[k]
            segs[k] = np.array([c[0], c[0] + (c[3] - c[0]) / 3.0, c[0] + (c[3] - c[0]) * (2.0 / 3.0), c[3]])


# --------------------------------------------------------------------------- #
# Closed pieces (a loop with no junctions)
# --------------------------------------------------------------------------- #
def _keeps_loop(segs: list, pts: np.ndarray, reach: float) -> bool:
    """Whether a closed chain of >=2 cubics keeps an area and follows ``pts`` within ``reach`` both ways."""
    if len(pts) == 0:
        return False
    area = _chain_area(segs, pts[0])
    return (len(segs) >= 2 and abs(area) > 0.25
            and all_within(sample_cubics(segs, 0.25), pts, reach)
            and all_within(pts, sample_cubics(segs, 0.1), reach))


def _tighter(style: Style, tol: float) -> Style:
    return replace(style, tol=tol, line_tol=min(style.line_tol, tol), circle_tol=min(style.circle_tol, tol))


def _collapsed_loop(pts: np.ndarray, style: Style):
    """Fewest-segment fit of a closed loop whose fit collapsed to a single cubic.

    Tries, in order: tighter tolerances; two cubics split at the far point;
    the outline simplified with Douglas-Peucker at ``tol`` and finer.
    """
    tol = style.tol
    first = pts[0]
    d2 = ((pts - first) ** 2).sum(axis=1)
    far = int(np.flatnonzero(d2[1:] == d2[1:].max())[0]) + 1 if len(pts) > 1 else 0
    reach = max(tol, 0.5) * 1.5
    t = tol
    for _ in range(3):
        t /= 1.5
        segs = fit_outline(pts, _tighter(style, t), Stats())
        if segs is not None and _keeps_loop(segs, pts, reach):
            return segs
    halves: list = []
    for half in (pts[: far + 1], np.vstack([pts[far:], first[None, :]])):
        if len(half) < 2:
            return None
        k = max(len(half) // 4, 1)
        halves.extend(fit_cubics(half, tangent(half, 0, k, 1), tangent(half, len(half) - 1, k, -1), float("inf")))
    if _keeps_loop(halves, pts, reach):
        return halves
    t = tol
    for _ in range(4):
        simple = _polygon(pts, t)
        if _keeps_loop(simple, pts, reach):
            return simple
        t /= 2.0
    return None


def fit_closed(pts: np.ndarray, style: Style):
    """A whole closed loop fitted.

    The same loop gives the same curves whichever way it runs and wherever it
    starts: it is always fitted clockwise from its left-most point. Hence the
    outline of a shape and the hole it leaves in the shape around it are ONE curve.
    """
    n = len(pts)
    nxt = np.roll(pts, -1, axis=0)
    area2 = float((pts[:, 0] * nxt[:, 1] - nxt[:, 0] * pts[:, 1]).sum())
    flip = area2 < 0
    if flip:
        pts = pts[::-1].copy()
    start = min(range(n), key=lambda i: _before_key(pts[i]))
    pts = np.roll(pts, -start, axis=0)
    stats = Stats()
    segs = fit_outline(pts, style, stats)
    if segs is None:
        return None
    # A fit that strayed was replaced by a plain one that nothing has checked; one that strays
    # too (a wild handle on a tiny or ragged loop) gives way to the outline's own points.
    if stats.reverted > 0:
        rs = resample_closed(pts, 0.5)
        ref = rs[0] if rs else pts
        if not all_within(sample_cubics(segs, 0.25), ref, 1.5):
            segs = _polygon(pts, style.tol)
    # A closed editable subpath needs at least two anchors.
    if len(segs) < 2:
        segs = _collapsed_loop(pts, style) or _polygon(pts, 0.0)
    if style.upright:
        snap_upright(segs, True, False)
    # Opposite sides of a thin loop can both straighten onto the same chord -> zero area.
    # Keep the positioned boundary when a non-degenerate loop loses all its area.
    if abs(area2) > 1e-8 and abs(_chain_area(segs, pts[0])) < 1e-8:
        segs = _polygon(pts, 0.0)
    return _reversed(segs) if flip else segs


# --------------------------------------------------------------------------- #
# Open pieces (stretches between two junctions)
# --------------------------------------------------------------------------- #
def fit_piece(pts: np.ndarray, style: Style, fixed_ends: bool) -> list:
    """A stretch of a loop between two junctions, fitted from its first point to its last.

    The same piece gives the same curve whichever way it runs (it is fitted from
    its further-left end): the edge two regions share is one curve.
    """
    n = len(pts)
    if n == 0:
        return []
    ka, kb = _before_key(pts[0]), _before_key(pts[-1])
    if ka == kb:
        flip = n > 3 and _before_key(pts[1]) > _before_key(pts[n - 2])
    else:
        flip = ka > kb
    if flip:
        pts = pts[::-1].copy()
    segs = fit_open(pts, style, Stats())
    # Nothing else checks this fit: one that strays gives way to the piece's own points.
    rs = resample_open(pts, 0.5)
    ref = rs[0] if rs else pts
    if not all_within(sample_cubics(segs, 0.25), ref, 1.5):
        keep = [False] * n
        _rdp(pts, 0, n - 1, style.tol, keep)
        keep[0] = keep[n - 1] = True
        kept = [pts[i] for i in range(n) if keep[i]]
        segs = [np.array([a, a, b, b]) for a, b in zip(kept[:-1], kept[1:]) if not np.array_equal(a, b)]
    if style.upright:
        snap_upright(segs, False, fixed_ends)
    # A fixed-end piece can return to its junction; fewer than two cubics can't form a closed boundary.
    if fixed_ends and np.array_equal(pts[0], pts[-1]) and len(segs) < 2:
        rel = pts - pts[0]
        area2 = float((rel[:-1, 0] * rel[1:, 1] - rel[:-1, 1] * rel[1:, 0]).sum())
        if np.isfinite(area2) and abs(area2) > 1e-8:
            reach = max(style.tol, 0.5) * 1.5
            t, found = style.tol, None
            for _ in range(3):
                t /= 1.5
                c = fit_open(pts, _tighter(style, t), Stats())
                if _keeps_loop(c, pts, reach):
                    found = c
                    break
            segs = found or [np.array([a, a, b, b]) for a, b in zip(pts[:-1], pts[1:]) if not np.array_equal(a, b)]
    return _reversed(segs) if flip else segs


# --------------------------------------------------------------------------- #
# Anchors / subpaths (the editable form of a chain of cubics)
# --------------------------------------------------------------------------- #
@dataclass
class Anchor:
    """One anchor point of a path with its two Bezier handles (``h_in`` arrives, ``h_out`` leaves)."""
    p: np.ndarray
    h_in: np.ndarray
    h_out: np.ndarray
    smooth: bool        # handles are collinear -> a smooth point; otherwise a corner


@dataclass
class SubPath:
    anchors: list
    closed: bool = True


def make_subpath(segs: list, closed: bool = True):
    """A chain of cubics as a subpath. A straight piece has no handles; an anchor whose two
    handles line up is 'smooth'."""
    n = len(segs)
    if n < (2 if closed else 1):
        return None
    anchors = []
    for i in range(n if closed else n + 1):
        before = segs[i - 1] if i > 0 else (segs[-1] if closed else None)
        after = segs[i] if i < n else None
        p = after[0] if after is not None else before[3]
        h_in = before[2] if before is not None and not _is_straight(before) else p
        h_out = after[1] if after is not None and not _is_straight(after) else p
        vi, vo = p - h_in, h_out - p
        li, lo = float(np.hypot(*vi)), float(np.hypot(*vo))
        smooth = li > 1e-9 and lo > 1e-9 and abs(vi[0] * vo[1] - vi[1] * vo[0]) < 0.07 * li * lo and float(vi @ vo) > 0
        anchors.append(Anchor(p=p.copy(), h_in=h_in.copy(), h_out=h_out.copy(), smooth=bool(smooth)))
    return SubPath(anchors, closed)


# --------------------------------------------------------------------------- #
# The entry point
# --------------------------------------------------------------------------- #
def fit_loop(loop, w: int, h: int, across, junction_at, style: Style):
    """Fit one loop of a ``w x h`` image into a closed Bezier subpath (pixel coordinates).

    1. ``subpixel.outline`` places the loop's points to a fraction of a pixel and cuts it
       into pieces at junctions.
    2. Each piece is fitted. A *hard* piece (no anti-aliasing: the edge is only known to
       about half a pixel) gets looser tolerances and more smoothing.
    3. The cubics are glued into a closed subpath.

    Returns ``(SubPath, any_piece_was_hard)`` or None.
    """
    pieces = subpixel_outline(loop, w, h, across, junction_at)
    segs: list = []
    for piece in pieces:
        st = style
        if piece.hard:
            st = replace(style, tol=style.tol * 2.5, line_tol=style.line_tol * 1.8,
                         circle_tol=style.circle_tol * 1.8, sigma=0.9)
        pts = np.array(piece.pts, dtype=float)
        if piece.closed:
            segs = fit_closed(pts, st)
            if segs is None:
                return None
        else:
            segs.extend(fit_piece(pts, st, True))
    sp = make_subpath(segs, True)
    if sp is None:
        return None
    return sp, any(p.hard for p in pieces)
