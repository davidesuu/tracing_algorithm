# vtrace — VectorCraft's Black-and-White Image Trace, in documented Python

`vtrace` is a readable Python port of the **Black and White** branch of the Image Trace
algorithm in [VectorCraft](https://github.com/storytold/vectorcraft) (`crates/trace`, Rust).
It turns a raster image into clean vector outlines (SVG with Bézier curves).

The goal of this port is **understanding**: every module starts with a docstring explaining the
stage it implements, which Rust file it came from, and *why* it does what it does. For a
narrated tour of the whole algorithm read **[`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md)**.

```
pixels ──► black/white labels ──► remove specks ──► follow region boundaries
       ──► place edges to 1/100 px ──► cut at junctions ──► fix corners, snap lines/arcs
       ──► fit Bézier curves ──► SVG
```

## Install & run

Needs Python 3.10+ with `numpy`, `scipy`, `pillow` (no other dependencies).

```bash
pip install numpy scipy pillow

# command line
python -m vtrace input.png -o output.svg --preset Silhouettes
python -m vtrace input.png --threshold 150 --paths 70 --corners 60 --noise 20 --ignore-white
python -m vtrace input.png --dump-dir debug/        # also writes the intermediate label images

# python
python - <<'EOF'
from vtrace import trace_image, preset, save_svg
result = trace_image("input.png", preset("Line Art"))
print(len(result.paths), "paths,", result.anchor_count(), "anchors")
save_svg(result, "output.svg")
EOF

# tests (26) and examples
python -m unittest discover -s tests -v
python examples/make_examples.py          # writes SVG + PNG previews to examples/out/
```

## What the settings do

| setting | range | effect |
|---|---|---|
| `threshold` | 0–255 | Pixels with luma **below** this are black. Also shifts where the *edge* is placed (higher ⇒ fatter shapes). |
| `paths` | 0–100 | Fidelity. Sets the Bézier fit tolerance: 0.78 px at 0 → 0.08 px at 100. More ⇒ more anchors. |
| `corners` | 0–100 | Turn angle that counts as a corner: 90° at 0 → 35° at 100. More ⇒ more sharp corners kept. |
| `noise` | px | Regions and holes smaller than this many pixels are removed. |
| `ignore_white` | bool | Don't emit white shapes (no background rectangle). |
| `snap_curves_to_lines` | bool | Near-straight curves → lines; circular runs → true arcs; near-level/upright lines → exactly level/upright. |

### Presets (identical numbers to the Rust `preset()`)

| preset | threshold | paths | corners | noise | snap |
|---|---|---|---|---|---|
| Default | 128 | 50 | 75 | 25 | no (white kept) |
| Black and White Logo | 128 | 95 | 80 | 8 | yes |
| Sketched Art | 150 | 50 | 50 | 100 | no |
| **Silhouettes** | **200** | 40 | 60 | 30 | no |
| Line Art | 128 | 80 | 70 | 10 | no |
| Technical Drawing | 128 | 95 | 90 | 4 | yes |

(All except *Default* also set `ignore_white`.)

## Project layout — and where each file came from

Read in this order; it follows the data through the pipeline.

| # | file | stage | Rust original (`crates/trace/src/`) |
|---|---|---|---|
| 1 | `params.py` | settings, presets, sliders → fit numbers | `lib.rs` (`TraceParams`, `outline_style`, `preset`) |
| 2 | `quantize.py` | pixel → black/white label; reference tones; **denoise** | `quantize.rs`, `bw_tones` in `lib.rs` |
| 3 | `contour.py` | follow each region's boundary along pixel cracks | `contour.rs` |
| 4 | `subpixel.py` | **edge placement to a fraction of a pixel**, junctions, pieces | `subpixel.rs`, `shift*` / `junction` in `lib.rs` |
| 5 | `geom.py` | resample, smooth, corner angles, line/circle fits | `logo/geom.rs` |
| 6 | `restore.py` | corner restoration + the `Style` struct | `logo/restore.rs`, `logo/outline.rs` |
| 7 | `bezier.py` | Schneider least-squares cubic Bézier fitting | `logo/bezier.rs` |
| 8 | `outline_fit.py` | one outline → lines/arcs/Béziers, with safety checks | `logo/outline.rs` |
| 9 | `fit.py` | a whole loop → closed subpath (piece by piece) | `fit.rs` |
| 10 | `pipeline.py` | the driver, `trace()` | `trace_paths` / `Layers::layer` in `lib.rs` |
| – | `svg.py`, `preview.py`, `cli.py` | output, PNG preview, command line | (new) |

## Scope: what is and isn't ported

**Ported (Black and White mode):** thresholding, noise removal, boundary following, sub-pixel edge
placement with the threshold correction, junction/piece logic, corner restoration, line / arc /
upright snapping, Schneider Bézier fitting, hole handling, Ignore White, all six B/W presets.

**Not ported** (by request, or because they need other modes): Color, Grayscale and Flat-Logo modes;
*Create Strokes* (so the **Outline** preset); *Shapes* detection; *Auto Grouping*; Gradients;
Overlapping method; the 64-megapixel / anchor-count safety limits; parallel layer tracing.

## How well does it work?

Checked against anti-aliased synthetic shapes (see `tests/` and `docs/HOW_IT_WORKS.md` §9):

* a disc traced with threshold 128 is within **0.01 px** of the true radius and is exactly **4 anchors**;
* with **Silhouettes (threshold 200)** it lands **+0.24 px** outside the true edge — the Rust project's own
  quality notes describe the same behaviour (outline sits where the image crosses the threshold, ≈0.3 px
  outside on anti-aliased art);
* a vertical anti-aliased edge at x = 10.3 / 17.5 / 22.8 is recovered to **±0.03 px**;
* a pixel-aligned square comes back with its four corners at exactly `(20,20)…(60,60)`;
* on hard-edged random blobs the traced area equals the pixel area to within **0.4 %**, outer loops wind
  one way and holes the other.

## Known differences & honest caveats

* **Not verified bit-for-bit against the Rust code.** There is no Rust toolchain in the environment this was
  written in, so the port was checked by reading the Rust source closely, by the unit tests, and by comparing
  behaviour to the numbers in the Rust repo's own quality notes. Expect small numerical differences.
* **The "notch" special case in `subpixel.outline` is omitted.** (In Rust, a one-pixel notch between two long
  edges is rebuilt as one sharp corner.) Here it stays as two small steps, which the corner restoration
  usually repairs. This is the most likely source of any visible difference.
* **Very acute corners** (e.g. the apex of a thin triangle) can come out slightly truncated or with a small
  "hook" at some settings (Silhouettes' looser smoothing shows it in `examples/out`). I have not been able to
  check whether the Rust implementation behaves identically.
* `all_within` uses a KD-tree instead of Rust's hash grid (same result, different code).
* **Speed:** pure Python. About 0.1 s for an 80×80 image, ~1 s for 300×300; time grows with outline
  length, so a 2000×2000 line drawing will take a while. Downscale first for experiments.
* `preview.render_png` uses PIL polygon filling, which is up to one sub-pixel "fatter" than the SVG.
  Use the SVG for exact geometry.
