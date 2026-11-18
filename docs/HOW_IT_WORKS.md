# How the Black-and-White trace works

This document walks through the algorithm in the order the data flows. Each section names the
Python module and the Rust file it was ported from. Conventions: **x grows right, y grows down**;
pixel `(px, py)` is the unit square whose top-left corner is the grid point `(px, py)`.

```
 RGBA image
   │ §2  quantize_bw ........ luma < threshold → BLACK, else WHITE;  alpha < 128 → TRANSPARENT
   │ §3  denoise ............ merge tiny regions into the neighbour they share most border with
   │ §4  pick layers ........ black layer, white layer (white skipped by Ignore White)
   │
   │  for each layer, for each connected component:
   │ §5    trace_mask ....... boundary loops along pixel cracks (outer + holes)
   │ §6    subpixel.outline . move each crack to the true edge; cut loop into pieces at junctions
   │ §7    fit_loop ......... per piece: resample → smooth → restore corners → lines / arcs / Béziers
   │ §8    assemble ......... outer + holes = one path   →  SVG
   ▼
 TraceResult
```

---

## 1. The idea in one paragraph

A bitmap edge is a staircase. Tracing it naively (connect the pixel corners) gives jagged,
half-pixel-accurate polygons with thousands of points. This algorithm instead (a) decides black/white,
(b) follows the boundary, (c) uses the *grey levels of anti-aliased pixels* to recover where the edge
really is to ~1/100 px, and (d) fits the **fewest** smooth curves and straight lines that stay within a
tolerance of that precise outline, restoring sharp corners that blur and smoothing had rounded off.

---

## 2. Quantize — `quantize.py` (`quantize.rs`)

Every pixel becomes one of three labels:

| label | value | rule |
|---|---|---|
| WHITE | 0 | `luma ≥ threshold` |
| BLACK | 1 | `luma < threshold` |
| TRANSPARENT | 2 | `alpha < 128` (never traced) |

`luma = (299·R + 587·G + 114·B) / 1000` (Rec. 601), rounded for the decision.
Threshold 128 is "halfway". **Silhouettes uses 200**, so anything darker than 200 — light grey
included — becomes part of the shape.

`bw_tones` also measures the image's real paper and ink: the **98th percentile** of the white pixels'
luma and the **2nd percentile** of the black pixels' luma. Why not the mean? On thin or blurry black
lines most "black" pixels are partly paper, so their mean is a grey; measuring a line's middle
against that grey would make it look "darker than ink" and ripple its edges.

---

## 3. Denoise — `quantize.denoise` (`quantize.rs`)

All 4-connected same-label regions are collected. Then, **smallest first**, any region with fewer than
`noise` pixels is merged into a neighbour:

1. Count how long its border is with each neighbouring *label* (ties → the lower label).
2. Choose the biggest region of that label touching it as the **target** (ties → lower id).
3. Absorb the region — *and* every other region of that label that touches it — into the target.
4. Push the (now larger) target back on the heap; it may now be big enough to survive.

Consequences: specks of ink vanish, pin-holes in solid shapes fill, and raising `noise` just continues the same
merge sequence. Transparent pixels are a wall: regions never merge into them. An island whose only
neighbour is transparency is kept.

---

## 4. Layers — `pipeline.py`

One layer per label that actually occurs. With **Ignore White** the white layer is dropped (that
is the "no background rectangle" setting; every preset except *Default* turns it on). Each
layer is traced only inside the bounding box of its pixels. Because only two colours exist, two
layers never overlap; their shared edges are fitted identically (§6–7) so they abut perfectly.

---

## 5. Boundary following — `contour.py` (`contour.rs`)

`scipy.ndimage.label` finds 4-connected components. For each, loops are walked along the **cracks**
between inside and outside pixels, always keeping the inside pixel on the **right**:

```
at every pixel corner look at the two pixels straight ahead:

      ahead-left │ ahead-right
      ───────────┼────────────
         (we travel this way →)

   ahead-right outside  → turn right
   else ahead-left outside → go straight
   else                  → turn left
```

Diagonal-only neighbours always turn right, so they stay separate regions (that is what 4-connectivity
means). Only *turning* vertices are stored. The shoelace sum gives twice the signed area:
**positive ⇒ outer boundary, negative ⇒ hole** (holes run the other way round). A loop is started at
the top edge of an inside pixel whose upper neighbour is outside, and a `walked` flag per pixel makes
sure each loop is walked once.

Result: for each component, an outer `Loop` and a list of hole `Loop`s — all still staircases.

---

## 6. Sub-pixel edge placement — `subpixel.py` (`subpixel.rs`, `shift*` in `lib.rs`)

This is the heart of the quality. Look at a vertical edge crossing pixel 10, with black on the left and
the true edge at x = 10.3. Pixel 10 is 30 % black, so (with exact anti-aliasing) luma ≈ 178:

```
 x:      8     9    10    11
 luma:   0     0   178   255
 label:  B     B     W     W        ← the crack between 9 and 10 is where the staircase puts the edge
                ▲
              crack (x = 10)   true edge at 10.3  →  shift = +0.3
```

### 6.1 `shift_one_side` (the normal, "wide edge" case)

For the crack, let `inside(k)` = how much of the *inside* tone pixel `k` steps out contains (0..1),
where 0 is the pixel one step *inside*, 1 the first *outside* pixel, and so on.
Then

```
a1 = 1 − inside(0)         # how much *outside* tone leaked into the inside pixel
b1 = inside(1)             # how much *inside* tone leaked into the outside pixel
(a2, b2: the same for the next pixel further out each way, if it has the same label as its neighbour)
shift = b1 + b2 − a1 − a2
```

In the example, `b1 = 0.3`, everything else 0, so `shift = 0.3` ✓. The sum works because a straight edge
crossing the pixel column always leaves exactly its distance-from-the-crack of 'inside tone' beyond it.
Clamped to ±0.5 px.

### 6.2 Both sides, averaged

The estimate is computed once reading the crack from the black pixel's side and once from the white
pixel's side (`shift = (one_side(c) − one_side(flipped(c))) / 2`). That makes it exactly antisymmetric, so
the black region and the white region on either side of a crack compute **the same** point — no gaps.

### 6.3 Thin lines

If one side is a line only 1–3 pixels thick, its two edges share pixels and can't be read separately.
Then `thin()` reads the **centroid and width** of the whole column and places the crack's edge at
`middle ± width/2`.

### 6.4 The tone used for "white"

"Black" is the image's darkest ink tone. "White" is the lightest pixel within 3–4 pixels of the crack
(clamped to the image's paper tone), so uneven scanned paper isn't mistaken for ink.

### 6.5 The threshold correction (specific to Black-and-White)

The labels were decided at `threshold`, not at 50 %. A pixel turns black once it contains
`turn = (white − threshold) / (white − black)` of ink: 0.5 for threshold 128; **0.216 for 200**.
So the shape is *meant* to extend to where the image crosses the threshold:

```
shift += (+1 if the inside is black else −1) × (0.5 − turn) / steep
```

* threshold 128 → correction ≈ 0 → the outline sits on the true 50 % edge;
* threshold 200 → `(0.5 − 0.216) = +0.284 px` outward → the silhouette is slightly **fatter** than the original;
* `steep` (≤ 1) is the edge steepness measured from the pixels; on blurry edges it is smaller, so
  the correction is larger (a threshold matters more where the edge is soft).

### 6.6 From cracks to points — `outline()`

For every crack, a point is produced at the crack's midpoint moved outward by its shift. Special cases:

* **True corners** (two long edges meet): the point is where the two *shifted* edges intersect.
* **Corner pixels:** a pixel at a true corner lacks 'inside tone' because of *both* edges; read as one
  edge it would dent the corner. So each edge of ≥ 3 px takes the shift of its neighbouring crack at the corner.
* **Hard edges** (no anti-aliasing information, e.g. a GIF): nearly all shifts in a piece are equal. Only
  staircase steps (1-px jogs) are emitted, read as points on a sloped edge. Such pieces get looser fit tolerances (§7).
* **Junctions** — pixel corners where ≥ 3 regions meet, where two parts of a region touch diagonally, or
  where an edge hits the image border. The loop is **cut** there into `Piece`s. The junction's position
  depends only on the corner, so every region around it uses the same point.

(Omitted in this port: Rust's one-pixel *notch* special case.)

---

## 7. Fitting curves — `fit.py`, `outline_fit.py`, `restore.py`, `geom.py`, `bezier.py`

`params.outline_style()` turns the sliders into numbers:

| slider | → | formula |
|---|---|---|
| Paths *p* | Bézier tolerance `tol` | `0.08 + 0.7·(1 − p/100)^1.5`   (0.78 px … 0.08 px) |
| Corners *c* | corner angle | `90° − 55°·c/100`   (90° … 35°) |
| Snap Curves To Lines | `arcs`, `upright`, looser `line_tol`/`circle_tol` | 0.5/0.3 px (on) vs 0.2/0.15 px (off) |

Straight lines are *always* snapped (`snap=True`); *arcs* and *level/upright* snapping only with the checkbox.

### 7.1 Pieces and canonical direction (`fit.py`)

A loop without junctions is **one closed piece**; otherwise each stretch between two junctions is an open
piece whose end points are kept *exactly*. Every piece is fitted in a canonical direction (closed loops:
clockwise from the left-most point; open pieces: from the further-left end), so two regions sharing an
edge get the *same* curve and never leave a hairline gap.

### 7.2 The closed-outline pipeline (`outline_fit.styled_fit`)

```
raw points ─► resample every 0.25 px            (geom.resample_closed)
           ─► light Gaussian smooth (σ≈0.72 samples) → s0     (geom.smooth_closed)
           ─► restore corners  → inserts exact sharp vertices (restore.restore_corners)
           ─► find corners: local maxima of turning angle ≥ corner_deg (geom.find_corners)
           ─► for each run between corners:                    (fit_run)
                 straight?  every sample within line_tol of the chord → ONE line
                 circular?  (Snap on) circle fit within circle_tol, r in 2.5…400 → exact arc(s)
                 else       Schneider least-squares Béziers within tol
```

**Corner restoration.** Blur and smoothing round every sharp corner. For each strong bend: skip `l1 = 1.3 px`
around it, fit a straight line to each side (`l2 = 3.5 px` long, each must be straight within 0.15 px), intersect
the two lines, and — if the intersection is within 1.3 px of the bend and the corner angle is 30°…160° — replace
the rounded bend by that one sharp vertex.

```
 rounded bend:    ╭──╮       restored:   ┌──┐
                  │  │                   │  │
```

**Schneider's algorithm** (`bezier.py`) fits one cubic to a run: end points fixed, end tangents given, the
two handle *lengths* solved by a 2×2 least-squares system. If the worst error is within `tol` → done.
If close, re-parameterise (Newton step) and refit up to 4 times. Otherwise split at the worst point
(sharing a tangent so the join is smooth) and recurse on the halves.

### 7.3 Safety net (`outline_fit.fit_outline`)

The result is sampled and compared with the raw contour **both ways**; every point must be within
`max_dev = 1.4 px` of the other curve. If not, retry without snapping/restoration; if that still wanders, use a
plain Bézier fit of the raw points. `fit.py` adds more guards: a fit that collapses to < 2 cubics is
rebuilt tighter / as two halves / as a Douglas-Peucker polygon; a loop that fits to zero area is replaced by
the positioned polygon; pieces whose fit strays > 1.5 px fall back to polygon pieces.

### 7.4 Upright snapping (Snap Curves To Lines only)

Straight pieces ≥ 2 px long within 2.5° of level or upright are made *exactly* level/upright by moving both ends
to their mean — unless that would move an end > 1.6 px. Open pieces keep their junction ends fixed.

---

## 8. Assembly and output — `pipeline.py`, `svg.py`

The outer loop's subpath and its holes' subpaths (dropping holes smaller than `noise` px) form one `TracedPath`. Each
subpath is a list of `Anchor`s (`p`, `h_in`, `h_out`, `smooth`). SVG output writes `M` + `L`/`C` + `Z` per subpath; holes
wind opposite to outlines so the default `nonzero` fill rule cuts them out.

---

## 9. What was verified

| check | result |
|---|---|
| Disc, threshold 128 | radius error ≤ 0.01 px, 4 anchors |
| Disc, Silhouettes | +0.243 px (straight-edge formula: +0.284; Rust docs report ≈ +0.28 mean on curved shapes) |
| Disc, Sketched Art (threshold 150) | +0.070 px (the straight-edge formula `0.5 − (255−150)/255` predicts 0.088; the measured value on a curved edge is a bit lower — I have not investigated why) |
| Vertical AA edge at x = 10.3 / 17.5 / 22.8 | recovered to ±0.03 px |
| Pixel-aligned 40×40 square | corners at exactly (20,20)…(60,60) |
| Hard-edged blobs, noise 0 | area conserved within 0.4 %; outer + / hole − winding |
| Degenerate inputs (1×1, all white, all black, empty, saddles, checkerboard, border-touching) | no crash |

Run `python -m unittest discover -s tests -v` for the 26 automated tests.

---

## 10. Worked example: what Silhouettes does to a disc

`bw(200, 40.0, 60.0, 30, false)` means: threshold 200, paths 40, corners 60, noise 30, snap off, ignore white.

1. **Quantize:** pixels with luma < 200 are black — the anti-aliased rim (anything with ≥ 21.6 % ink) is included.
2. **Denoise:** regions under 30 px vanish.
3. **Contour:** one outer loop (the staircase around the disc), no holes.
4. **Edge placement:** each crack's `shift` finds the true 50 % edge, then the threshold term pushes it
   **+0.284 px outward** → the disc grows by ≈ 0.24–0.28 px all round.
5. **Fitting:** `paths 40` → tol ≈ 0.08 + 0.7·0.6^1.5 ≈ 0.40 px; `corners 60` → corner angle 57°. A
   smooth disc has no corners, so the whole loop is fitted with Béziers → **4 anchors**.

---

## 11. Glossary

* **Crack** — the unit edge between two adjacent pixels.
* **Loop** — a closed boundary: outer (positive area) or hole (negative).
* **Shift** — sub-pixel distance from a crack to the true edge, −0.5…+0.5 px, positive towards the outside pixel.
* **Junction** — a pixel corner shared by ≥ 3 regions (or image border / diagonal touch); loops are cut here.
* **Piece** — the stretch of a loop between two junctions (or a whole loop with none).
* **Hard edge** — an edge with no anti-aliasing: its position is only known to about half a pixel.
* **Tolerance (`tol`)** — the farthest a fitted curve may stray from the points it fits.
