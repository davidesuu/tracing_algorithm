"""
preprocess.py -- Preprocessing pipeline for vtrace:
  1. Background removal using RMBG-1.4 (generates Alpha channel)
  2. Perspective rectification (vanishing-point homography preserving Alpha)
"""
from __future__ import annotations

import cv2
import numpy as np

_RMBG_MODEL = None
_DEVICE = None


def _get_rmbg_model():
    """Load the RMBG model on demand (Lazy Loading)."""
    global _RMBG_MODEL, _DEVICE
    if _RMBG_MODEL is None:
        import os
        import torch
        from transformers import AutoModelForImageSegmentation

        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Check local cache first for instant loading without network retries
            _RMBG_MODEL = AutoModelForImageSegmentation.from_pretrained(
                "briaai/RMBG-1.4", trust_remote_code=True, local_files_only=True
            )
        except Exception:
            # Download if not yet cached locally
            _RMBG_MODEL = AutoModelForImageSegmentation.from_pretrained(
                "briaai/RMBG-1.4", trust_remote_code=True
            )
        _RMBG_MODEL.to(_DEVICE).eval()
    return _RMBG_MODEL, _DEVICE


def remove_background(image: np.ndarray) -> np.ndarray:
    """Remove background from an RGB or RGBA uint8 image.
    
    Returns uint8 (H, W, 4) RGBA with transparent background.
    """
    import torch
    import torch.nn.functional as F

    image = np.asarray(image, dtype=np.uint8)
    if image.ndim == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        img_rgb = image[:, :, :3]
    else:
        img_rgb = image

    orig_h, orig_w = img_rgb.shape[:2]
    model, device = _get_rmbg_model()

    # Preprocess for model (1024x1024)
    img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1)
    img_tensor = ((img_tensor - mean) / std).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    # Post-processing: scale mask back to original resolution
    output_tensor = outputs[0][0]
    output_tensor = F.interpolate(output_tensor, size=(orig_h, orig_w), mode="bilinear")
    mi, ma = output_tensor.min(), output_tensor.max()
    if ma - mi > 1e-8:
        output_tensor = (output_tensor - mi) / (ma - mi)
    else:
        output_tensor = torch.zeros_like(output_tensor)
    mask = (output_tensor * 255.0).clamp(0, 255).squeeze().cpu().numpy().astype(np.uint8)

    return np.dstack([img_rgb, mask])


def _fit_vp(lines: np.ndarray) -> np.ndarray:
    # lines: Nx3 homogeneous lines; least-squares point closest to all of them
    A = lines / np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    return np.linalg.svd(A)[2][-1]


def rectify_perspective(image: np.ndarray) -> np.ndarray:
    """Rectify facade perspective while strictly preserving the Alpha channel.
    
    If given an RGBA image, newly added borders and existing background will have Alpha=0.
    """
    image = np.asarray(image)
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image

    if image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        border_val = (0, 0, 0, 0)
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        border_val = (0, 0, 0)
    else:
        gray = image
        border_val = 0

    lsd = cv2.createLineSegmentDetector()
    detected = lsd.detect(gray)[0]
    if detected is None or len(detected) < 4:
        return image

    segs = detected.reshape(-1, 4)

    # center coordinates for numerical stability
    c = np.array([w / 2.0, h / 2.0])
    p1 = np.c_[segs[:, :2] - c, np.ones(len(segs))]
    p2 = np.c_[segs[:, 2:] - c, np.ones(len(segs))]
    lines = np.cross(p1, p2)

    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    long_ = length > 0.05 * max(h, w)
    vert = long_ & (np.abs(ang - 90) < 25)
    horz = long_ & ((ang < 25) | (ang > 155))

    if np.sum(vert) < 2 or np.sum(horz) < 2:
        return image

    vp_v = _fit_vp(lines[vert])
    vp_h = _fit_vp(lines[horz])

    # step 1: send the line through both vanishing points to infinity
    l = np.cross(vp_v, vp_h)
    if abs(l[2]) < 1e-8:
        return image
    H1 = np.array([[1, 0, 0], [0, 1, 0], [l[0] / l[2], l[1] / l[2], 1]])

    # step 2: map the vertical/horizontal directions to y/x axes
    def direction(vp):
        q = H1 @ vp
        norm = np.linalg.norm(q[:2])
        return q[:2] / norm if norm > 1e-8 else q[:2]

    dv, dh = direction(vp_v), direction(vp_h)
    if dv[1] < 0:
        dv = -dv
    if dh[0] < 0:
        dh = -dh

    M = np.column_stack([dh, dv])
    if abs(np.linalg.det(M)) < 1e-6:
        return image

    A = np.linalg.inv(M)
    H2 = np.eye(3)
    H2[:2, :2] = A

    T = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]])
    H = H2 @ H1 @ T

    # fit output canvas to the warped image corners
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
    wc = H @ corners
    wc = (wc[:2] / wc[2]).T
    mn, mx = wc.min(0), wc.max(0)
    diff = mx - mn
    if np.any(diff <= 1e-3):
        return image

    scale = max(w, h) / max(diff)
    S = np.array([[scale, 0, -mn[0] * scale], [0, scale, -mn[1] * scale], [0, 0, 1]])
    size = (int(round(diff[0] * scale)), int(round(diff[1] * scale)))

    return cv2.warpPerspective(
        image, S @ H, size,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val
    )


def preprocess_image(
    image: np.ndarray,
    rectify: bool = True,
    remove_bg: bool = True
) -> np.ndarray:
    """Preprocess an input image before passing to vtrace.
    
    1. If remove_bg is True, segments foreground and assigns Alpha transparency.
    2. If rectify is True, aligns vertical and horizontal vanishing lines, preserving Alpha.
    
    Returns an (H, W, 4) uint8 RGBA array.
    """
    rgba = np.asarray(image, dtype=np.uint8)
    if rgba.ndim == 2:
        rgba = cv2.cvtColor(rgba, cv2.COLOR_GRAY2RGBA)
    elif rgba.ndim == 3 and rgba.shape[2] == 3:
        if remove_bg:
            rgba = remove_background(rgba)
        else:
            rgba = cv2.cvtColor(rgba, cv2.COLOR_RGB2RGBA)
    elif rgba.ndim == 3 and rgba.shape[2] == 4:
        if remove_bg:
            # If requested, re-estimate transparency with RMBG
            rgba = remove_background(rgba[:, :, :3])

    if rectify:
        rgba = rectify_perspective(rgba)

    return rgba
