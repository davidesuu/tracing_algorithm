"""
cli.py -- command line:  python -m vtrace input.png -o output.svg [options]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from . import PRESETS, load_rgba, preset, save_svg, trace, with_overrides


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vtrace", description="Black-and-White image trace (port of VectorCraft).")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None, help="SVG to write (default: input name + .svg)")
    ap.add_argument("--preset", default="Default", help=f"one of: {', '.join(PRESETS)}")
    ap.add_argument("--threshold", type=int, help="0-255; luma below this is black")
    ap.add_argument("--paths", type=float, help="0-100 fidelity")
    ap.add_argument("--corners", type=float, help="0-100 corner keeping")
    ap.add_argument("--noise", type=int, help="drop regions smaller than this many px")
    ap.add_argument("--ignore-white", dest="ignore_white", action="store_true", default=None)
    ap.add_argument("--keep-white", dest="ignore_white", action="store_false")
    ap.add_argument("--snap", dest="snap_curves_to_lines", action="store_true", default=None,
                    help="Snap Curves To Lines")
    ap.add_argument("--dump-dir", help="also write the intermediate label images (quantized / denoised) here")
    a = ap.parse_args(argv)

    params = with_overrides(preset(a.preset), threshold=a.threshold, paths=a.paths, corners=a.corners,
                            noise=a.noise, ignore_white=a.ignore_white, snap_curves_to_lines=a.snap_curves_to_lines)
    rgba = load_rgba(a.input)
    t0 = time.time()
    res = trace(rgba, params)
    out = a.output or os.path.splitext(a.input)[0] + ".svg"
    save_svg(res, out)
    print(f"{a.input}: {res.width}x{res.height}px, {len(res.paths)} paths, {res.anchor_count()} anchors, "
          f"{time.time() - t0:.2f}s -> {out}")
    print(f"params: {params}")
    if a.dump_dir:
        from PIL import Image
        os.makedirs(a.dump_dir, exist_ok=True)
        lut = np.array([[255, 255, 255], [0, 0, 0], [255, 0, 255]], dtype=np.uint8)   # white, black, transparent=magenta
        for name in ("quantized", "denoised"):
            Image.fromarray(lut[res.debug[name]]).save(os.path.join(a.dump_dir, f"{name}.png"))
        print(f"wrote intermediate images to {a.dump_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
