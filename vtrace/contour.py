"""
contour.py -- STAGE 3: follow the boundary of every region along the pixel cracks.

Port of ``crates/trace/src/contour.rs``.

Coordinates
-----------
*Pixel corners* are the grid points ``(x, y)`` with ``0 <= x <= w`` and ``0 <= y <= h``.
Pixel ``(px, py)`` is the unit square whose top-left corner is ``(px, py)``.
A *crack* is the unit edge between two neighbouring pixels.

The walk
--------
Every crack between an inside pixel and an outside pixel becomes a directed edge with the
INSIDE pixel on the RIGHT-hand side of the direction of travel (remember y points down).
Consequently outer boundaries run clockwise (on screen) and holes counter-clockwise.

At each corner we look at the two pixels straight ahead:

        ahead-left | ahead-right        (relative to the direction of travel)

    * ahead-right is OUTSIDE            -> turn right  (hug the inside pixel)
    * else ahead-left is OUTSIDE        -> go straight
    * else (both inside)                -> turn left

At a *saddle* (two inside pixels touching only diagonally) both rules are consistent with
turning right, so diagonal neighbours stay separate (= 4-connectivity, the same rule used to
label components).

Only the TURNING vertices are stored, so a straight run of 50 cracks is one edge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

R, D, L, U = 1, 2, 4, 8


def _right_of(d: int) -> int:
    return {R: D, D: L, L: U, U: R}[d]


_STEP = {R: (1, 0), D: (0, 1), L: (-1, 0), U: (0, -1)}


@dataclass
class Loop:
    """A closed boundary: corner vertices in order (the loop closes back to the first)."""
    pts: list                  # [(x, y), ...] turning vertices only
    area2: int                 # twice the signed area: >0 outer boundary, <0 hole


@dataclass
class Component:
    """One 4-connected component of the mask: its outer loop, its holes, its pixel count."""
    outer: Loop
    holes: list = field(default_factory=list)
    pixels: int = 0


def trace_mask(mask: np.ndarray) -> list:
    """Trace every 4-connected component of a boolean ``mask`` (shape ``(h, w)``).

    A loop starts at the TOP edge of an inside pixel whose upper neighbour is outside, found by
    scanning pixels in raster order; a ``walked`` flag per pixel (set whenever the walk runs
    rightwards along that pixel's top edge) makes sure each loop is walked exactly once.
    """
    h, w = mask.shape
    comp, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, 1))
    sizes = np.bincount(comp.ravel(), minlength=count + 1)[1:]
    m = mask.ravel().tolist()
    owner_of = comp.ravel().tolist()

    def inside(x: int, y: int) -> bool:
        return 0 <= x < w and 0 <= y < h and m[y * w + x]

    def ahead(x: int, y: int, d: int):
        """(is the pixel ahead-right inside?, is the pixel ahead-left inside?)"""
        nw, ne, se, sw = (x - 1, y - 1), (x, y - 1), (x, y), (x - 1, y)
        r, l = {R: (se, ne), D: (sw, se), L: (nw, sw), U: (ne, nw)}[d]
        return inside(*r), inside(*l)

    longest = 4 * (w + 1) * (h + 1)                          # no loop has more edges than this
    walked = bytearray(w * h)
    outers: list = [None] * count
    holes: list = [[] for _ in range(count)]

    ys, xs = np.nonzero(mask)                                # raster order
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        s = sy * w + sx
        # Start only on the top edge of a pixel whose upper neighbour is outside, once per loop.
        if walked[s] or (sy > 0 and m[s - w]):
            continue
        walked[s] = 1
        pts = [(sx, sy)]
        x, y, d = sx + 1, sy, R
        area2 = 0
        px, py = sx, sy
        for _ in range(longest):
            area2 += px * y - x * py                         # shoelace term for edge (px,py)->(x,y)
            px, py = x, y
            right, left = ahead(x, y, d)
            if not right:
                nxt = _right_of(d)
            elif not left:
                nxt = d
            else:
                nxt = _right_of(_right_of(_right_of(d)))      # three rights = one left
            if (x, y) == (sx, sy) and nxt == R:              # back at the start, heading the start way
                break
            if nxt == R and 0 <= y < h and 0 <= x < w:
                walked[y * w + x] = 1                        # we run along the top edge of pixel (x, y)
            if nxt != d:
                pts.append((x, y))                           # keep only turning vertices
            dx, dy = _STEP[nxt]
            x, y, d = x + dx, y + dy, nxt
        loop = Loop(pts, area2)
        o = owner_of[s] - 1                                  # component index (labels start at 1)
        if area2 > 0:
            outers[o] = loop
        else:
            holes[o].append(loop)

    return [Component(outer=o, holes=holes[i], pixels=int(sizes[i]))
            for i, o in enumerate(outers) if o is not None]
