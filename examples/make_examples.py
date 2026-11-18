"""
Generate a few anti-aliased test images, trace each with every Black-and-White preset, and write
SVG + PNG previews into ``examples/out``.   Run:   python examples/make_examples.py
"""
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vtrace import PRESETS, preset, save_svg, trace          # noqa: E402
from vtrace.preview import render_png                         # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def aa(draw_fn, size=160, ss=8):
    im = Image.new("L", (size * ss, size * ss), 255)
    draw_fn(ImageDraw.Draw(im), ss)
    return im.resize((size, size), Image.BOX)


def badge(d, ss):           # a disc with a hole, a rounded square and a triangle
    d.ellipse([10 * ss, 10 * ss, 90 * ss, 90 * ss], fill=0)
    d.ellipse([34 * ss, 34 * ss, 66 * ss, 66 * ss], fill=255)
    d.rounded_rectangle([100 * ss, 14 * ss, 150 * ss, 60 * ss], radius=9 * ss, fill=0)
    d.polygon([(125 * ss, 80 * ss), (155 * ss, 148 * ss), (95 * ss, 140 * ss)], fill=0)


def letters(d, ss):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",):
        try:
            f = ImageFont.truetype(path, 95 * ss)
            break
        except OSError:
            f = ImageFont.load_default()
    d.text((10 * ss, 14 * ss), "Rg", font=f, fill=0)


def scribble(d, ss):        # thin, curvy lines
    pts = [(10 + 140 * t / 60, 80 + 55 * np.sin(t / 6.0)) for t in range(61)]
    d.line([(x * ss, y * ss) for x, y in pts], fill=0, width=int(2.2 * ss), joint="curve")
    d.line([(20 * ss, 20 * ss), (140 * ss, 40 * ss)], fill=0, width=int(1.3 * ss))


for name, fn in (("badge", badge), ("letters", letters), ("scribble", scribble)):
    gray = aa(fn)
    gray.save(os.path.join(OUT, f"{name}_input.png"))
    rgba = np.array(gray.convert("RGBA"))
    for pname in PRESETS:
        t0 = time.time()
        res = trace(rgba, preset(pname))
        tag = f"{name}_{pname.lower().replace(' ', '_')}"
        save_svg(res, os.path.join(OUT, tag + ".svg"))
        render_png(res, 3).save(os.path.join(OUT, tag + ".png"))
        print(f"{name:9s} {pname:22s} paths={len(res.paths):2d} anchors={res.anchor_count():4d}  {time.time() - t0:.2f}s")
print("written to", OUT)
