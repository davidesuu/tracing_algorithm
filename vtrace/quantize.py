"""
quantize.py -- STAGE 1 and 2 of the pipeline: pixels -> black/white labels -> denoised labels.

Ports of ``quantize.rs`` (the ``Mode::BlackAndWhite`` branch, ``denoise``) and
``bw_tones`` from ``lib.rs``.

Labels
------
Every pixel gets one label:

    0 = WHITE        1 = BLACK        2 = TRANSPARENT (not traced at all)

(The Rust palette is ``[white, black]`` in the same order.)
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy import ndimage

WHITE, BLACK, TRANSPARENT = 0, 1, 2


def luma_int(rgba: np.ndarray) -> np.ndarray:
    """Rec. 601 luma, rounded, 0..255 (identical to Rust's integer formula)."""
    r, g, b = (rgba[..., k].astype(np.int64) for k in range(3))
    return ((r * 299 + g * 587 + b * 114 + 500) // 1000).clip(0, 255)


def luma_float(rgba: np.ndarray) -> np.ndarray:
    """Un-rounded Rec. 601 luma. The edge-placement maths wants the exact value."""
    r, g, b = (rgba[..., k].astype(np.float64) for k in range(3))
    return (r * 299.0 + g * 587.0 + b * 114.0) / 1000.0


def quantize_bw(rgba: np.ndarray, threshold: int) -> np.ndarray:
    """THE Black-and-White decision, one pixel at a time.

        alpha < 128          -> TRANSPARENT
        luma  < threshold    -> BLACK
        otherwise            -> WHITE

    ``threshold`` is the only place the Threshold slider acts on the *labels*.
    (It acts a second time in subpixel.py, to place edges.)
    """
    lum = luma_int(rgba)
    labels = np.where(lum < threshold, BLACK, WHITE).astype(np.int8)
    labels[rgba[..., 3] < 128] = TRANSPARENT
    return labels


def bw_tones(rgba: np.ndarray, labels: np.ndarray):
    """The luminance of 'white' and of 'black' in THIS image: ``(white_tone, black_tone)``.

    Not the means. Where the black is thin lines (blurred or scanned) most black pixels are
    part paper, so their mean is a gray; read against that gray, the middle of every line
    would count as 'more than all ink' and put a ripple on its edges. Instead we take the
    98th percentile of the white pixels' luma and the 2nd percentile of the black pixels' luma
    (i.e. the lightest paper and the darkest ink actually present).
    """
    lum = luma_int(rgba)

    def percentile(vals: np.ndarray, at: float, none: float) -> float:
        if vals.size == 0:
            return none
        hist = np.bincount(vals, minlength=256)
        seen = np.cumsum(hist)
        return float(np.argmax(seen >= at * vals.size))     # first luma value reaching the share

    return (percentile(lum[labels == WHITE], 0.98, 255.0),
            percentile(lum[labels == BLACK], 0.02, 0.0))


# --------------------------------------------------------------------------- #
# Noise removal
# --------------------------------------------------------------------------- #
def denoise(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Merge every 4-connected same-label region smaller than ``min_area`` into its neighbour.

    * Regions are merged SMALLEST FIRST, into the label they share the longest border with
      (ties -> lower label). The merge order does not depend on ``min_area``: a higher value
      simply continues the same sequence.
    * After merging, the absorbed region and the neighbours of the chosen label that it now
      touches are all one region (so it may grow past ``min_area`` and stop being noise).
    * TRANSPARENT is a barrier: regions never merge into it and it is never merged.
    * An opaque island surrounded only by transparency has no neighbour to merge into -> kept.

    Returns a NEW labels array.
    """
    labels = labels.copy()
    h, w = labels.shape
    if min_area <= 1 or h == 0 or w == 0:
        return labels

    # ---- 1. label every 4-connected region of every (opaque) label ----------------------------
    at = np.full(h * w, -1, dtype=np.int64)                  # pixel -> region id
    region_label: list = []                                  # region id -> label
    region_pixels: list = []                                 # region id -> list of pixel indices
    four = ndimage.generate_binary_structure(2, 1)           # 4-connectivity
    flat = labels.ravel()
    for lab in (WHITE, BLACK):
        comp, count = ndimage.label(labels == lab, structure=four)
        if count == 0:
            continue
        idx = np.flatnonzero(comp.ravel())
        order = np.argsort(comp.ravel()[idx], kind="stable")
        idx, ids = idx[order], comp.ravel()[idx][order]
        bounds = np.flatnonzero(np.diff(ids)) + 1
        for chunk in np.split(idx, bounds):
            rid = len(region_label)
            region_label.append(lab)
            region_pixels.append(chunk.tolist())
            at[chunk] = rid

    def neighbours(i: int):
        x, y = i % w, i // w
        if x > 0:
            yield i - 1
        if x + 1 < w:
            yield i + 1
        if y > 0:
            yield i - w
        if y + 1 < h:
            yield i + w

    # ---- 2. merge the smallest regions first ---------------------------------------------------
    heap = [(len(px), rid) for rid, px in enumerate(region_pixels)]
    heapq.heapify(heap)
    while heap:
        area, rid = heapq.heappop(heap)
        if area >= min_area:
            break
        if len(region_pixels[rid]) != area:
            continue                                         # stale entry (absorbed or since grown)
        borders: dict = {}                                   # neighbouring region id -> shared border length
        for i in region_pixels[rid]:
            for j in neighbours(i):
                other = at[j]
                if other != rid and other != -1:
                    borders[other] = borders.get(other, 0) + 1
        by_colour: dict = {}                                 # label -> shared border length
        for other, cnt in borders.items():
            by_colour[region_label[other]] = by_colour.get(region_label[other], 0) + cnt
        if not by_colour:
            continue                                         # island surrounded only by transparency
        colour = max(by_colour, key=lambda lab: (by_colour[lab], -lab))
        same = [o for o in borders if region_label[o] == colour]
        target = max(same, key=lambda o: (len(region_pixels[o]), -o))
        for src in [rid] + [o for o in same if o != target]:
            moved, region_pixels[src] = region_pixels[src], []
            for i in moved:
                at[i] = target
            region_pixels[target].extend(moved)
        heapq.heappush(heap, (len(region_pixels[target]), target))

    # ---- 3. write the surviving regions' labels back ------------------------------------------
    for rid, px in enumerate(region_pixels):
        if px:
            flat[px] = region_label[rid]
    return flat.reshape(h, w)
