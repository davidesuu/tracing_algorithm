"""
subpixel.py -- STAGE 4: place outlines to a fraction of a pixel, and cut them at junctions.

Port of ``crates/trace/src/subpixel.rs`` plus the ``shift`` / ``shift_one_side`` / ``junction``
methods of ``Layers`` in ``lib.rs`` (Black-and-White branch only).

The problem
-----------
The boundary loop from contour.py runs along pixel cracks, so it is a STAIRCASE that is only
right to about half a pixel. But an anti-aliased or slightly blurred image tells us where the
edge really is: the pixels beside it are PART one tone and PART the other.

    Anti-aliased black disc edge (luma values):          crack at x = 3
        255 255 255 | 190  60   0   0                     "white" pixels left of the crack
                    ^ the edge is really ~0.25 px to the right of the crack

THE KEY IDEA -- ``shift_one_side``
    Look at the column of pixels ACROSS a crack. A straight edge crossing that column leaves,
    summed over the column, exactly as much 'inside' tone as its distance along the column.
    So sum up how much 'inside tone' the pixels beyond the crack have (and how much 'outside
    tone' the pixels before it have) and you know how far the edge is from the crack,
    to a fraction of a pixel.  ``shift`` is that distance in pixels, -0.5 .. +0.5, positive
    towards the OUTSIDE pixel.

Black-and-White adds ONE more term: the *threshold*. A pixel is only called black below
``threshold``, so with a threshold far from the middle the labelled edge sits away from the true
50 % edge. We correct for that (see the end of ``shift_one_side``).

Then ``outline`` turns the loop of cracks into POINTS (one per crack, or a true corner) and
cuts it into ``Piece`` s at *junctions* so that regions sharing an edge get identical points.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .quantize import BLACK, TRANSPARENT, WHITE


# --------------------------------------------------------------------------- #
# Crack = a unit edge between an 'inside' pixel and the pixel one step away
# --------------------------------------------------------------------------- #
# A crack is a plain tuple ``(ix, iy, sx, sy)``:
#     (ix, iy) = the INSIDE pixel,     (sx, sy) = step from the inside pixel to the OUTSIDE one.
# ``at(k)`` is the pixel k steps out from the inside pixel:
#     k = 0 inside pixel, k = 1 outside pixel, k = -1 the pixel behind the inside pixel, ...
def crack_flipped(c):
    """The same crack seen from the other side: the outside pixel becomes the inside one."""
    return (c[0] + c[2], c[1] + c[3], -c[2], -c[3])


def border(d) -> int:
    """What lies across a crack on the image's border: one 'neighbour' per side, so that the
    image's corners are junctions and stay exact."""
    return 1_000_000 - {(1, 0): 0, (0, 1): 1, (-1, 0): 2}.get(d, 3)


def _sign(v: int) -> int:
    return (v > 0) - (v < 0)


# --------------------------------------------------------------------------- #
# EdgeReader: everything that needs to LOOK AT the pixels
# --------------------------------------------------------------------------- #
class EdgeReader:
    """Computes edge shifts and junction points for one image.

    Parameters
    ----------
    labels      : 2-D array of 0/1/2 labels (after denoise)
    lum         : 2-D float array, un-rounded luma of every pixel
    tones       : ``(white_tone, black_tone)`` from ``quantize.bw_tones``
    threshold   : the Threshold slider (0..255)
    """

    def __init__(self, labels, lum, tones, threshold: int):
        self.h, self.w = labels.shape
        self.labels = labels.tolist()          # python lists: far faster for scalar access
        self.lum = lum.tolist()
        self.white_tone, self.black_tone = float(tones[0]), float(tones[1])
        self.threshold = float(threshold)
        self._cache: dict = {}

    # ---- pixel access ------------------------------------------------------------------------
    def _px(self, c, k):
        x, y = c[0] + k * c[2], c[1] + k * c[3]
        return (x, y) if 0 <= x < self.w and 0 <= y < self.h else None

    def _label(self, c, k):
        """Label of the pixel k steps out, or None if off-image OR transparent."""
        p = self._px(c, k)
        if p is None:
            return None
        l = self.labels[p[1]][p[0]]
        return None if l == TRANSPARENT else l

    def neighbour(self, c) -> int:
        """The region across a crack: the OUTSIDE pixel's label (stacked layers don't use this)."""
        p = self._px(c, 1)
        return TRANSPARENT if p is None else self.labels[p[1]][p[0]]

    # ---- the heart: how far is the edge from the crack? -------------------------------------
    def shift(self, c) -> float:
        """Distance (px, -0.5..0.5, + = towards the outside pixel) of the true edge from crack ``c``.

        The estimate is read from EACH side of the crack and the two readings averaged, so
        that the result is exactly antisymmetric (``shift(c) == -shift(flipped(c))``): the two
        regions on either side of a crack then get the very same point.
        """
        r = self._cache.get(c)
        if r is None:
            r = (self._shift_one_side(c) - self._shift_one_side(crack_flipped(c))) / 2.0
            self._cache[c] = r
        return r

    def _shift_one_side(self, c) -> float:
        label = lambda k: self._label(c, k)
        la, lb = label(0), label(1)
        if la is None or lb is None or la == lb:
            return 0.0

        # --- the two tones: 'black' is the image's black; 'white' is the lightest pixel within a
        #     few pixels of the crack (the paper AS IT IS HERE: a scan's paper is darker in one
        #     corner than another, and read against the image's lightest white it would count
        #     as part ink). -----------------------------------------------------------------
        black = self.black_tone
        lights = []
        for k in range(-3, 5):
            p = self._px(c, k)
            if p is not None:
                lights.append(self.lum[p[1]][p[0]])
        lo = min(black + 32.0, self.white_tone)
        white = min(max(max(lights) if lights else -math.inf, lo), self.white_tone)
        tone = lambda l: black if l == BLACK else white
        ca, cb = tone(la), tone(lb)                  # tone of the inside / outside pixel's side
        d = cb - ca
        if d * d <= 0.0:
            return 0.0

        def inside(k: int) -> float:
            """How much of the INSIDE tone (0..1) the pixel k steps out has. Off-image: none."""
            p = self._px(c, k)
            if p is None:
                return 0.0
            v = self.lum[p[1]][p[0]]
            return 1.0 - min(max((v - ca) / d, 0.0), 1.0)

        def thin(frm: int, direction: int, n: int, tone_fn):
            """Edge from a THIN run of ``n`` pixels (<= 3) of one side, squeezed between the other.
            Returns (middle, width) measured from the crack towards the outside, or None."""
            near, far = frm - direction, frm + n * direction
            cells = [(frm + j * direction, tone_fn(frm + j * direction)) for j in range(n)]
            cells += [(near, tone_fn(near)), (far, tone_fn(far))]
            for edge, out in ((near, -direction), (far, direction)):
                if label(edge + out) == label(edge):
                    cells.append((edge + out, min(tone_fn(edge + out), tone_fn(edge))))
            width = sum(t for _, t in cells)
            if width > 1e-6:
                return sum(t * (k - 0.5) for k, t in cells) / width, width    # pixel k's centre is k-0.5 from the crack
            return None

        def depth(direction: int, own: int, other: int):
            """How many pixels deep a side is here (up to 3 of its own label, then the other's)."""
            start = 0 if direction < 0 else 1
            n = 1
            while n <= 3 and label(start + n * direction) == own:
                n += 1
            return n if n <= 3 and label(start + n * direction) == other else None

        steep = 1.0
        shift = None
        n = depth(-1, la, lb)                         # is the INSIDE side a thin line?
        if n is not None:
            t = thin(0, -1, n, inside)
            if t is not None:
                shift = t[0] + t[1] / 2.0
        if shift is None:
            n = depth(1, lb, la)                      # is the OUTSIDE side a thin line?
            if n is not None:
                t = thin(1, 1, n, lambda k: 1.0 - inside(k))
                if t is not None:
                    shift = t[0] - t[1] / 2.0
        if shift is None:
            # --- the normal case: a wide edge. Sum the tone leaked across the crack. ---------
            a1, b1 = 1.0 - inside(0), inside(1)
            a2 = min(1.0 - inside(-1), a1) if label(-1) == la else None
            b2 = min(inside(2), b1) if label(2) == lb else None
            # The edge's steepness per pixel: 1 for an anti-aliased edge, less for a blurred one.
            changes = []
            if a2 is not None:
                changes.append(1.0 - a2 - b1)
            if b2 is not None:
                changes.append(1.0 - a1 - b2)
            if changes:
                steep = min(max(max(changes) ** 2, 0.35), 1.0)
            shift = b1 + (b2 or 0.0) - a1 - (a2 or 0.0)

        # --- Black-and-White: the THRESHOLD correction ----------------------------------------
        # 'turn' = the share of black at which a pixel turns black. With threshold in the middle
        # it is 0.5 and nothing happens; a higher threshold (Silhouettes: 200) calls lighter
        # pixels black, so the labelled boundary lies outside the true 50% edge and we must
        # move the edge back by (0.5 - turn), more the softer (less steep) the edge is.
        turn = min(max((white - self.threshold) / max(white - black, 1.0), 0.02), 0.98)
        shift += (1.0 if la == BLACK else -1.0) * (0.5 - turn) / steep
        return shift

    # ---- junctions ---------------------------------------------------------------------------
    def junction(self, x: int, y: int):
        """The junction point at pixel corner ``(x, y)``, or None if there isn't one.

        A junction is where three or more regions meet, or two parts of a region touch by the
        corner only (a saddle), or an edge reaches the image's border. It lies at the corner,
        moved across by what the edges running up/down from it say, and up/down by what those
        to its left/right say. It depends on the corner alone, so every region around it gets
        the same point.
        """
        def label(px, py):
            return self.labels[py][px] if 0 <= px < self.w and 0 <= py < self.h else None

        around = [label(x - 1, y - 1), label(x, y - 1), label(x, y), label(x - 1, y)]   # NW NE SE SW
        edges = sum(1 for i in range(4) if around[i] != around[(i + 1) % 4])
        apart = lambda a, b, c, d: around[a] == around[c] and around[b] != around[a] and around[d] != around[a]
        touching = apart(0, 1, 2, 3) or apart(1, 2, 3, 0)
        if edges < 3 and not touching:
            return None

        def shift_of(a, step):
            la, lb = label(*a), label(a[0] + step[0], a[1] + step[1])
            if la is None or lb is None or la == lb:
                return None
            return min(max(self.shift((a[0], a[1], step[0], step[1])), -0.5), 0.5)

        def mean(a, b):
            if a is not None and b is not None:
                return (a + b) / 2.0
            return a if a is not None else (b if b is not None else 0.0)

        dx = mean(shift_of((x - 1, y - 1), (1, 0)), shift_of((x - 1, y), (1, 0)))
        dy = mean(shift_of((x - 1, y - 1), (0, 1)), shift_of((x, y - 1), (0, 1)))
        return (x + dx, y + dy)


# --------------------------------------------------------------------------- #
# Loop of cracks  ->  pieces of sub-pixel points
# --------------------------------------------------------------------------- #
@dataclass
class Piece:
    """A stretch of an outline with ONE neighbour along it."""
    pts: list            # [(x, y), ...] float points
    hard: bool           # edge has no anti-aliasing (known only to ~half a pixel)
    closed: bool         # the piece is a whole loop without junctions


def outline(loop, w: int, h: int, across, junction_at):
    """The outline of ``loop`` (a loop of a ``w x h`` image) as sub-pixel points, in pieces.

    ``across(crack) -> (shift, neighbour_label)`` and ``junction_at(x, y) -> point | None`` come
    from the EdgeReader.

    Simplification vs the Rust original: the 'notch' special case (a one-pixel notch between two
    long edges rebuilt as one sharp corner) is omitted; such a notch is left as two staircase
    steps, which the fitter's corner-restoration then usually repairs.
    """
    v = loop.pts
    n = len(v)
    at = lambda i: (float(v[i % n][0]), float(v[i % n][1]))
    if n < 4:
        return [Piece([at(i) for i in range(n)], True, True)]

    # ---- 1. one record per CRACK: its shift, the region across it, which edge it is on -------
    edges: list = []                      # (dirx, diry, length, index of first crack)
    shifts: list = []
    others: list = []
    edge_of: list = []
    for i in range(n):
        a, b = v[i], v[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        dirx, diry = _sign(dx), _sign(dy)
        ln = max(abs(dx), abs(dy))
        edges.append((dirx, diry, ln, len(shifts)))
        for k in range(ln):
            sx, sy = a[0] + dirx * k, a[1] + diry * k        # corner where this crack starts
            if (dirx, diry) == (1, 0):
                inside_px, outside_px = (sx, sy), (sx, sy - 1)
            elif (dirx, diry) == (0, 1):
                inside_px, outside_px = (sx - 1, sy), (sx, sy)
            elif (dirx, diry) == (-1, 0):
                inside_px, outside_px = (sx - 1, sy - 1), (sx - 1, sy)
            else:
                inside_px, outside_px = (sx, sy - 1), (sx - 1, sy - 1)
            crack = (inside_px[0], inside_px[1], outside_px[0] - inside_px[0], outside_px[1] - inside_px[1])
            if 0 <= outside_px[0] < w and 0 <= outside_px[1] < h:
                s, other = across(crack)
            else:
                s, other = 0.0, border((dirx, diry))         # the edge of the image is where it is
            s = min(max(s, -0.5), 0.5) if math.isfinite(s) else 0.0
            # Round to 1/1024 px so both regions get bit-identical values.
            shifts.append(math.copysign(math.floor(abs(s) * 1024 + 0.5), s) / 1024.0)
            others.append(other)
            edge_of.append(i)
    cracks = len(shifts)
    if cracks == 0:
        return [Piece([at(i) for i in range(n)], True, True)]

    def corner_of(c):
        i = edge_of[c]
        e = edges[i]
        k = c - e[3]
        return (v[i][0] + e[0] * k, v[i][1] + e[1] * k)

    # ---- 2. junctions: where the neighbour changes (or the callback says so) ---------------
    junctions: list = []
    for c in range(cracks):
        x, y = corner_of(c)
        changed = others[c] != others[(c + cracks - 1) % cracks]
        jp = junction_at(x, y)
        junctions.append(jp if jp is not None else ((float(x), float(y)) if changed else None))
    is_j = lambda c: junctions[c] is not None
    first = next((c for c in range(cracks) if is_j(c)), None)

    def is_hard(piece_shifts):
        """Hard edge: nearly every crack of the piece moves by the same amount (no AA info)."""
        s = sorted(piece_shifts)
        median = (s[len(s) // 2] + s[(len(s) - 1) // 2]) / 2.0
        return sum(1 for x in piece_shifts if abs(x - median) > 0.03) * 4 < len(piece_shifts)

    # ---- 3. which way the loop turns at each vertex; which edges are staircase steps -------
    def turn(i):
        a, b = edges[(i - 1) % n], edges[i % n]
        if junctions[b[3]] is not None:
            return 0                        # at a junction the loop goes its own way: says nothing
        return _sign(a[0] * b[1] - a[1] * b[0])

    step = [edges[i][2] == 1 and turn(i) * turn(i + 1) < 0 for i in range(n)]
    outward = lambda d: (float(d[1]), float(-d[0]))          # unit vector pointing OUT of the shape

    # ---- 4. corner pixels: a pixel at a true corner lacks inside tone because of BOTH edges ---
    #         Read as one edge's alone it would dent the corner, so each edge takes the shift of
    #         the NEXT crack along it instead (edges of >= 3 px only).
    shifts = list(shifts)
    for i in range(n):
        b_i, e_i = (i - 1) % n, i
        if step[b_i] or step[e_i] or turn(i) == 0:
            continue
        before, edge = edges[b_i], edges[e_i]
        if before[2] >= 3 and not any(is_j(before[3] + before[2] - k) for k in (1, 2)):
            shifts[before[3] + before[2] - 1] = shifts[before[3] + before[2] - 2]
        if edge[2] >= 3 and not any(is_j(edge[3] + k) for k in (1, 2)):
            shifts[edge[3]] = shifts[edge[3] + 1]

    # ---- 5. build the pieces: walk the cracks, emit points, cut at junctions ---------------
    pieces: list = []
    start = first if first is not None else 0
    c = 0
    while c < cracks:
        length = next((k for k in range(1, cracks - c) if is_j((start + c + k) % cracks)), cracks - c)
        own = [shifts[(start + c + k) % cracks] for k in range(length)]
        hard = is_hard(own)
        pts: list = []
        for k in range(length):
            crack = (start + c + k) % cracks
            i = edge_of[crack]
            e = edges[i]
            along = crack - e[3]
            corner = (v[i][0] + e[0] * along, v[i][1] + e[1] * along)
            if k == 0 and first is not None:
                # the junction itself: the one point every region around it has
                pts.append(junctions[crack] if junctions[crack] is not None else (float(corner[0]), float(corner[1])))
            elif along == 0 and not step[i] and not step[(i - 1) % n]:
                # a TRUE corner: where its two edges, each moved by its own shift, meet
                bd = edges[(i - 1) % n]
                ob, oe = outward(bd), outward(e)
                sb, se = shifts[(crack - 1) % cracks], shifts[crack]
                pts.append((corner[0] + ob[0] * sb + oe[0] * se, corner[1] + ob[1] * sb + oe[1] * se))
            if step[i] or not hard:
                # the middle of the crack, moved by its shift (hard edges: only staircase steps)
                oe = outward(e)
                pts.append((corner[0] + e[0] * 0.5 + oe[0] * shifts[crack], corner[1] + e[1] * 0.5 + oe[1] * shifts[crack]))
        if first is not None:
            # it ends on the next junction: the corner the next piece starts on
            nxt = (start + c + length) % cracks
            x, y = corner_of(nxt)
            pts.append(junctions[nxt] if junctions[nxt] is not None else (float(x), float(y)))
        pieces.append(Piece(pts, hard, first is None))
        c += length

    if first is None and len(pieces[0].pts) < 3:
        return [Piece([at(i) for i in range(n)], True, True)]
    return pieces
