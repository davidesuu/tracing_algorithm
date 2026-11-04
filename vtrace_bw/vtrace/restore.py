r"""
restore.py -- fitting style parameters and corner restoration.

Port of ``crates/trace/src/logo/restore.rs`` plus the ``Style`` struct from
``crates/trace/src/logo/outline.rs``.

Corner restoration
------------------
Anti-aliasing and blur round off every sharp corner a little. Smoothing the
outline (which we must do to kill the pixel staircase) rounds it even more.

For each candidate bend we:

    1. skip the rounded zone right at the bend (``l1`` px each side),
    2. fit a straight line to the edge on each side (``l2`` px long),
    3. intersect the two lines,
    4. put a sharp corner point at the intersection IF it is close to the bend.

    rounded:        restored:
       ____            ____
      /    \          |    |
     |      |   ->    |    |
      \____/          |____|
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geom import _round_half_up, cross2, line_fit, turning


@dataclass(frozen=True)
class RestoreParams:
    """Thresholds for corner restoration. Lengths are in pixels."""
    turn: float = 35.0          # a bend must turn at least this many degrees (over ~1 px) to be a candidate
    l1: float = 1.3             # rounded zone skipped each side of the bend
    l2: float = 3.5             # edge length (from the bend) used for each line fit
    line_res: float = 0.15      # each fitted line may deviate at most this much from its samples
    min_turn: float = 30.0      # the two edges must meet at >= this angle to count as a corner
    corner_dist: float = 1.3    # the restored corner may be at most this far from the rounded bend


@dataclass(frozen=True)
class Style:
    """How an outline is fitted. Lengths are in (working) pixels.

    Built from the user-facing sliders by ``params.outline_style``.
    """
    step: float = 0.12            # resampling spacing
    sigma: float = 0.5            # Gaussian smoothing, px
    corner_deg: float = 60.0      # turning angle that makes a corner  (<- "Corners" slider)
    corner_win: float = 0.7       # window (px) the turning angle is measured over
    tol: float = 0.12             # max Bezier fit error, px           (<- "Paths" slider)
    snap: bool = True             # straight runs -> exact lines
    arcs: bool = True             # (with snap) circular runs -> true arcs  (<- "Snap Curves To Lines")
    line_tol: float = 0.22        # how straight a run must be to become a line, px
    circle_tol: float = 0.25      # how round a run must be to become an arc, px
    restore: RestoreParams | None = field(default_factory=RestoreParams)
    max_dev: float = 1.4          # a fit straying farther than this from the raw contour is rejected
    check_step: float = 0.03      # spacing of the points used for that check
    upright: bool = False         # near-level / near-upright lines become exactly level / upright


def restore_corners(r: np.ndarray, s0: np.ndarray, step: float, p: RestoreParams):
    """Replace each restorable rounded bend of ``r`` with ONE sharp corner vertex.

    ``r``  : the evenly resampled closed outline.
    ``s0`` : a lightly smoothed copy (same length), used to *find* bends.

    Returns ``(new_points, indices_of_the_restored_corner_vertices)``.
    """
    n = len(r)
    if n < 16 or len(s0) != n or not step > 0:
        return r.copy(), []
    w = max(_round_half_up(1.0 / step), 3)
    ang = turning(s0, w)
    sup = _round_half_up(1.5 / step)
    circ = lambda a, b: (a + n - b) % n                       # circular distance a - b

    # --- 1. candidate bends: strongest turning angles first, suppressing neighbours -------------
    order = np.argsort(-ang, kind="stable")
    blocked = np.zeros(n, dtype=bool)
    cand: list = []
    for i in order:
        if ang[i] < p.turn:
            break
        if blocked[i]:
            continue
        cand.append(int(i))
        for d in range(0, min(sup, n // 2) + 1):
            blocked[(i + d) % n] = True
            blocked[(i + n - d) % n] = True
    cand.sort()

    # --- 2. for each candidate, fit a line to each side and intersect ---------------------------
    l1 = _round_half_up(p.l1 / step)
    l2_full = _round_half_up(p.l2 / step)
    accepted: list = []                                       # (index, corner point)
    for ci, i in enumerate(cand):
        prev, nxt = cand[(ci + len(cand) - 1) % len(cand)], cand[(ci + 1) % len(cand)]
        room_l, room_r = (circ(i, prev), circ(nxt, i)) if len(cand) > 1 else (n, n)
        l2 = min(l2_full, room_l // 2 + l1 // 2, room_r // 2 + l1 // 2)
        if l2 < l1 + _round_half_up(0.8 / step) or l2 >= n // 2:
            continue

        def pick(frm, to, back):
            return np.array([s0[(i + n - k) % n if back else (i + k) % n] for k in range(frm, to + 1)])

        left = pick(l1, l2, True)[::-1]                       # side before the bend, in travel order
        right = pick(l1, l2, False)                           # side after the bend
        la, lb = line_fit(left), line_fit(right)
        if la is None or lb is None:
            continue
        if la[2] > p.line_res or lb[2] > p.line_res:           # a side isn't straight -> not a corner
            continue
        turn_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(la[1] @ lb[1])))))
        if turn_deg < p.min_turn or turn_deg > 160.0:
            continue
        cr = float(cross2(la[1], lb[1]))
        if abs(cr) < 1e-6:
            continue
        s = float(cross2(lb[0] - la[0], lb[1])) / cr           # intersection of the two lines
        x = la[0] + la[1] * s
        if not np.isfinite(x).all() or float(np.hypot(*(x - r[i]))) > p.corner_dist:
            continue
        if accepted and circ(i, accepted[-1][0]) <= 2 * l1 + 1:  # neighbours may not share replaced stretches
            continue
        accepted.append((i, x))
    if len(accepted) > 1 and circ(accepted[0][0], accepted[-1][0]) <= 2 * l1 + 1:
        accepted.pop()
    if not accepted:
        return r.copy(), []

    # --- 3. splice: drop the rounded stretch around each corner, insert the sharp point ---------
    skip = np.zeros(n, dtype=bool)
    ins: list = [None] * n
    for i, x in accepted:
        for t in range(0, 2 * l1 + 1):
            skip[(i + n - l1 + t) % n] = True
        ins[(i + n - l1) % n] = x
    out, forced = [], []
    for i in range(n):
        if ins[i] is not None:
            forced.append(len(out))
            out.append(ins[i])
        if not skip[i]:
            out.append(r[i])
    return np.array(out), forced
