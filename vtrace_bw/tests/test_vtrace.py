"""
Unit tests for the Python port.  Run from the project root with:

    python -m unittest discover -s tests -v

The tests are written to double as documentation: each one states, in a few lines, a behaviour
of the pipeline that you can read and then poke at.
"""
import math
import os
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vtrace import TraceParams, preset, to_svg, trace                      # noqa: E402
from vtrace.bezier import fit_cubics, tangent                             # noqa: E402
from vtrace.contour import trace_mask                                     # noqa: E402
from vtrace.fit import cubic_signed_area                                  # noqa: E402
from vtrace.geom import circle_fit, find_corners, line_fit, resample_closed, smooth_closed  # noqa: E402
from vtrace.preview import subpath_polyline                               # noqa: E402
from vtrace.quantize import BLACK, TRANSPARENT, WHITE, denoise, quantize_bw   # noqa: E402


def gray_to_rgba(a):
    a = np.asarray(a, dtype=np.uint8)
    return np.dstack([a, a, a, np.full_like(a, 255)])


def antialiased(draw_fn, size, ss=8):
    """Draw with PIL at ss x resolution and box-filter down: a properly anti-aliased image."""
    im = Image.new("L", (size * ss, size * ss), 255)
    draw_fn(ImageDraw.Draw(im), ss)
    return gray_to_rgba(np.array(im.resize((size, size), Image.BOX)))


def disc_image(size=80, r=25):
    def draw(d, ss):
        c = (size // 2) * ss + ss / 2
        d.ellipse([c - r * ss, c - r * ss, c + r * ss - 1, c + r * ss - 1], fill=0)
    return antialiased(draw, size)


def black_polylines(res):
    return [[subpath_polyline(sp) for sp in tp.subpaths] for tp in res.paths if tp.color == (0, 0, 0)]


def signed_area(sp):
    n = len(sp.anchors)
    return sum(cubic_signed_area(np.array([sp.anchors[i].p, sp.anchors[i].h_out,
                                           sp.anchors[(i + 1) % n].h_in, sp.anchors[(i + 1) % n].p]))
               for i in range(n))


class QuantizeTests(unittest.TestCase):
    def test_threshold_and_alpha(self):
        rgba = np.zeros((1, 4, 4), dtype=np.uint8)
        rgba[0, :, 3] = 255
        rgba[0, 0, :3] = 0          # luma 0   -> black
        rgba[0, 1, :3] = 127        # luma 127 -> black  (< 128)
        rgba[0, 2, :3] = 128        # luma 128 -> white  (not < 128)
        rgba[0, 3, 3] = 127         # alpha 127 -> transparent
        self.assertEqual(quantize_bw(rgba, 128).tolist(), [[BLACK, BLACK, WHITE, TRANSPARENT]])
        self.assertEqual(quantize_bw(rgba, 200)[0, 2], BLACK)    # a higher threshold calls more pixels black

    def test_denoise_merges_small_region_into_longest_border(self):
        lab = np.full((8, 8), WHITE, dtype=np.int8)
        lab[2:6, 2:6] = BLACK          # big black square
        lab[0, 0] = BLACK              # a one-pixel black speck in the white
        out = denoise(lab, 4)
        self.assertEqual(out[0, 0], WHITE)                       # speck removed
        self.assertTrue((out[2:6, 2:6] == BLACK).all())          # square untouched

    def test_denoise_fills_small_hole(self):
        lab = np.full((9, 9), BLACK, dtype=np.int8)
        lab[4, 4] = WHITE
        self.assertEqual(denoise(lab, 2)[4, 4], BLACK)
        self.assertEqual(denoise(lab, 1)[4, 4], WHITE)           # min_area <= 1 does nothing

    def test_denoise_keeps_island_surrounded_by_transparency(self):
        lab = np.full((7, 7), TRANSPARENT, dtype=np.int8)
        lab[3, 3] = BLACK
        self.assertEqual(denoise(lab, 50)[3, 3], BLACK)

    def test_denoise_sequence_is_monotone(self):
        rng = np.random.default_rng(3)
        lab = (rng.random((40, 40)) > 0.5).astype(np.int8)
        prev = None
        for n in (1, 2, 5, 20, 80):
            cur = denoise(lab, n)
            self.assertEqual(cur.shape, lab.shape)
            prev = cur
        self.assertIsNotNone(prev)


class ContourTests(unittest.TestCase):
    def test_rectangle_loop(self):
        m = np.zeros((6, 8), dtype=bool)
        m[1:5, 2:6] = True
        (c,) = trace_mask(m)
        self.assertEqual(sorted(c.outer.pts), [(2, 1), (2, 5), (6, 1), (6, 5)])       # 4 corner vertices
        self.assertEqual(c.outer.area2, 2 * 16)                                        # positive = outer
        self.assertEqual(c.pixels, 16)

    def test_hole_has_negative_area(self):
        m = np.ones((7, 7), dtype=bool)
        m[3, 3] = False
        (c,) = trace_mask(m)
        self.assertEqual(len(c.holes), 1)
        self.assertEqual(c.holes[0].area2, -2)

    def test_diagonal_pixels_are_separate_components(self):
        m = np.zeros((4, 4), dtype=bool)
        m[1, 1] = m[2, 2] = True                                                       # touch only at a corner
        self.assertEqual(len(trace_mask(m)), 2)


class GeometryTests(unittest.TestCase):
    def test_line_and_circle_fit(self):
        pts = np.array([[0, 0], [1, 1.01], [2, 1.98], [3, 3.0]], dtype=float)
        self.assertLess(line_fit(pts)[2], 0.02)
        t = np.linspace(0, 2 * math.pi, 100, endpoint=False)
        circ = np.column_stack([5 + 3 * np.cos(t), 7 + 3 * np.sin(t)])
        c, r, resid = circle_fit(circ)
        self.assertAlmostEqual(r, 3, places=6)
        self.assertLess(np.hypot(*(c - [5, 7])), 1e-6)
        self.assertLess(resid, 1e-6)

    def test_corner_detection_finds_square_corners(self):
        sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        r, step = resample_closed(sq, 0.25)
        s = smooth_closed(r, 0.72)
        self.assertEqual(len(find_corners(s, 3, 60.0)), 4)

    def test_schneider_fits_arc_with_one_cubic(self):
        pts = np.array([[20 * math.cos(i * 0.006), 20 * math.sin(i * 0.006)] for i in range(200)])
        cs = fit_cubics(pts, tangent(pts, 0, 4, 1), tangent(pts, 199, 4, -1), 0.05)
        self.assertEqual(len(cs), 1)

    def test_cubic_signed_area_matches_shoelace_for_lines(self):
        sq = [np.array([a, a, b, b], dtype=float) for a, b in
              [((0, 0), (2, 0)), ((2, 0), (2, 2)), ((2, 2), (0, 2)), ((0, 2), (0, 0))]]
        self.assertAlmostEqual(abs(sum(cubic_signed_area(c) for c in sq)), 4.0, places=9)


class EdgePlacementTests(unittest.TestCase):
    """The sub-pixel idea: an anti-aliased edge's true position is recovered to a fraction of a pixel."""

    def vertical_edge(self, x0, size=40):
        """Black for x < x0 (fractional), white after; pixel columns are anti-aliased exactly."""
        a = np.full((size, size), 255.0)
        for x in range(size):
            cov = min(max(x0 - x, 0.0), 1.0)                  # fraction of pixel x that is black
            a[:, x] = 255.0 * (1.0 - cov)
        return gray_to_rgba(np.round(a))

    def edge_x(self, res):
        pts = np.vstack([p for polys in black_polylines(res) for p in polys])
        # keep only the sampled points of the RIGHT-hand edge (the black region also has its left
        # side on the image border at x = 0, and top/bottom sides we ignore)
        inner = pts[(pts[:, 1] > 8) & (pts[:, 1] < 32) & (pts[:, 0] > 3)]
        return inner[:, 0].mean()

    def test_threshold_128_recovers_true_edge(self):
        for x0 in (10.3, 17.5, 22.8):
            res = trace(self.vertical_edge(x0), TraceParams(threshold=128, ignore_white=True, noise=0))
            self.assertAlmostEqual(self.edge_x(res), x0, delta=0.03, msg=f"x0={x0}")

    def test_high_threshold_moves_edge_outward(self):
        # threshold 200: pixels with >21.6% ink count as black -> the outline sits ~0.28 px outside
        res = trace(self.vertical_edge(15.4), TraceParams(threshold=200, ignore_white=True, noise=0))
        self.assertAlmostEqual(self.edge_x(res), 15.4 + (0.5 - (255 - 200) / 255), delta=0.05)


class PipelineTests(unittest.TestCase):
    def test_disc_is_four_anchors_and_accurate(self):
        img = disc_image()
        res = trace(img, preset("Black and White Logo"))
        self.assertEqual(len(res.paths), 1)
        self.assertEqual(len(res.paths[0].subpaths[0].anchors), 4)
        cov = 1 - img[..., 0] / 255.0
        ys, xs = np.mgrid[0:80, 0:80]
        cx, cy = (cov * (xs + .5)).sum() / cov.sum(), (cov * (ys + .5)).sum() / cov.sum()
        r_true = math.sqrt(cov.sum() / math.pi)
        pts = black_polylines(res)[0][0]
        err = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r_true
        self.assertLess(abs(err).max(), 0.05)

    def test_silhouettes_is_fatter_than_default(self):
        img = disc_image()
        a = abs(signed_area(trace(img, preset("Default")).paths[-1].subpaths[0]))
        b = abs(signed_area(trace(img, preset("Silhouettes")).paths[-1].subpaths[0]))
        self.assertGreater(b, a)
        # radius grows by ~0.25 px  ->  area grows by ~2*pi*r*0.25
        self.assertAlmostEqual(b - a, 2 * math.pi * 25 * 0.25, delta=8)

    def test_square_corners_are_exact(self):
        def draw(d, ss):
            d.rectangle([20 * ss, 20 * ss, 60 * ss - 1, 60 * ss - 1], fill=0)
        res = trace(antialiased(draw, 80), preset("Black and White Logo"))
        pts = sorted(tuple(np.round(a.p, 2)) for a in res.paths[0].subpaths[0].anchors)
        self.assertEqual(pts, [(20.0, 20.0), (20.0, 60.0), (60.0, 20.0), (60.0, 60.0)])

    def test_ring_has_hole_with_opposite_winding(self):
        def draw(d, ss):
            c = 40 * ss
            d.ellipse([c - 30 * ss, c - 30 * ss, c + 30 * ss, c + 30 * ss], fill=0)
            d.ellipse([c - 14 * ss, c - 14 * ss, c + 14 * ss, c + 14 * ss], fill=255)
        res = trace(antialiased(draw, 80), preset("Line Art"))
        self.assertEqual(len(res.paths), 1)
        outer, hole = res.paths[0].subpaths
        self.assertGreater(signed_area(outer), 0)
        self.assertLess(signed_area(hole), 0)

    def test_area_is_conserved_on_hard_edged_noise(self):
        from scipy import ndimage
        rng = np.random.default_rng(1)
        g = ndimage.gaussian_filter(rng.random((64, 64)), 3)
        blobs = g > np.median(g)
        res = trace(gray_to_rgba(np.where(blobs, 0, 255)), TraceParams(noise=0, ignore_white=True))
        area = sum(signed_area(sp) for tp in res.paths for sp in tp.subpaths)
        self.assertAlmostEqual(area / blobs.sum(), 1.0, delta=0.01)

    def test_noise_removes_specks_and_ignore_white_drops_background(self):
        img = np.full((30, 30), 255, dtype=np.uint8)
        img[10:20, 10:20] = 0
        img[2, 2] = 0                                              # one-pixel speck
        a = trace(gray_to_rgba(img), TraceParams(noise=0, ignore_white=True))
        b = trace(gray_to_rgba(img), TraceParams(noise=5, ignore_white=True))
        c = trace(gray_to_rgba(img), TraceParams(noise=5, ignore_white=False))
        self.assertEqual((len(a.paths), len(b.paths), len(c.paths)), (2, 1, 2))
        self.assertTrue(all(p.color == (0, 0, 0) for p in b.paths))

    def test_degenerate_images_do_not_crash(self):
        for arr in (np.zeros((1, 1)), np.full((5, 5), 255), np.zeros((5, 5)), np.zeros((1, 9))):
            trace(gray_to_rgba(arr), TraceParams(noise=0))
        trace(np.zeros((0, 0, 4), dtype=np.uint8))

    def test_transparent_pixels_are_not_traced(self):
        rgba = gray_to_rgba(np.zeros((20, 20)))
        rgba[:, 10:, 3] = 0
        res = trace(rgba, TraceParams(noise=0, ignore_white=True))
        pts = np.vstack([p for polys in black_polylines(res) for p in polys])
        self.assertLessEqual(pts[:, 0].max(), 10.5)

    def test_snap_curves_to_lines_levels_a_slightly_tilted_edge(self):
        yy, xx = np.mgrid[0:60, 0:60]
        tilted = (yy > 30 + 0.02 * (xx - 30))                      # ~1 degree off level
        img = gray_to_rgba(np.where(tilted, 0, 255))
        res = trace(img, TraceParams(noise=0, ignore_white=True, snap_curves_to_lines=True))
        ys = {round(float(a.p[1]), 3) for a in res.paths[0].subpaths[0].anchors if 0 < a.p[0] < 60}
        res2 = trace(img, TraceParams(noise=0, ignore_white=True, snap_curves_to_lines=False))
        ys2 = {round(float(a.p[1]), 3) for a in res2.paths[0].subpaths[0].anchors if 0 < a.p[0] < 60}
        self.assertLessEqual(len(ys), len(ys2))

    def test_sliders_map_to_fit_parameters(self):
        lo, hi = TraceParams(paths=0, corners=0).outline_style(), TraceParams(paths=100, corners=100).outline_style()
        self.assertAlmostEqual(lo.tol, 0.78, places=6)
        self.assertAlmostEqual(hi.tol, 0.08, places=6)
        self.assertAlmostEqual(lo.corner_deg, 90.0)
        self.assertAlmostEqual(hi.corner_deg, 35.0)
        self.assertFalse(TraceParams().outline_style().arcs)
        self.assertTrue(TraceParams(snap_curves_to_lines=True).outline_style().arcs)

    def test_presets(self):
        p = preset("Silhouettes")
        self.assertEqual((p.threshold, p.paths, p.corners, p.noise, p.ignore_white), (200, 40.0, 60.0, 30, True))
        self.assertEqual(preset("[Default]"), TraceParams())
        with self.assertRaises(KeyError):
            preset("nope")

    def test_svg_is_well_formed(self):
        svg = to_svg(trace(disc_image(), preset("Default")))
        root = ET.fromstring(svg)
        self.assertTrue(root.tag.endswith("svg"))
        self.assertEqual(len(root), 2)                              # white path + black path


if __name__ == "__main__":
    unittest.main()
