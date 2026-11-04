import cv2
import numpy as np

def fit_vp(lines):
    # lines: Nx3 homogeneous lines; least-squares point closest to all of them
    A = lines / np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    return np.linalg.svd(A)[2][-1]

def rectify_facade(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    segs = cv2.createLineSegmentDetector().detect(gray)[0].reshape(-1, 4)

    # center coordinates for numerical stability
    c = np.array([w / 2, h / 2])
    p1 = np.c_[segs[:, :2] - c, np.ones(len(segs))]
    p2 = np.c_[segs[:, 2:] - c, np.ones(len(segs))]
    lines = np.cross(p1, p2)

    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    long_ = length > 0.05 * max(h, w)
    vert = long_ & (np.abs(ang - 90) < 25)
    horz = long_ & ((ang < 25) | (ang > 155))

    vp_v = fit_vp(lines[vert])   # add RANSAC here for real use
    vp_h = fit_vp(lines[horz])

    # step 1: send the line through both vanishing points to infinity
    l = np.cross(vp_v, vp_h)
    H1 = np.array([[1, 0, 0], [0, 1, 0], [l[0] / l[2], l[1] / l[2], 1]])

    # step 2: map the (now parallel) vertical/horizontal directions to the y/x axes
    def direction(vp):
        q = H1 @ vp
        d = q[:2] / np.linalg.norm(q[:2])
        return d
    dv, dh = direction(vp_v), direction(vp_h)
    if dv[1] < 0: dv = -dv
    if dh[0] < 0: dh = -dh
    A = np.linalg.inv(np.column_stack([dh, dv]))
    H2 = np.eye(3); H2[:2, :2] = A

    T = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]])
    H = H2 @ H1 @ T

    # fit output canvas to the warped image corners
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
    wc = H @ corners
    wc = (wc[:2] / wc[2]).T
    mn, mx = wc.min(0), wc.max(0)
    scale = max(w, h) / max(mx - mn)
    S = np.array([[scale, 0, -mn[0] * scale], [0, scale, -mn[1] * scale], [0, 0, 1]])
    size = tuple(((mx - mn) * scale).astype(int))
    return cv2.warpPerspective(img, S @ H, size)

img = cv2.imread('church_of_the_gesu-trivium-art-history.png')
if(img is None):
    raise ValueError("Image not found or unable to load.")
rectified = rectify_facade(img)
cv2.imwrite('rectified.jpg', rectified)