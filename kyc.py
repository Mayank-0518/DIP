"""Multimodal eKYC pipeline: image-quality gate, ID-document processing + OCR,
face matching (LBP baseline vs CNN), passive liveness (grey LBP vs colour-LBP+FFT), fusion.

Demo:  python kyc.py <selfie.jpg> <id_capture.jpg>   -> prints decision, writes results/demo.png
"""
import pickle
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from skimage.feature import local_binary_pattern

ROOT = Path(__file__).parent
_det = cv2.FaceDetectorYN.create(str(ROOT / "models/yunet.onnx"), "", (320, 320), 0.6)
_rec = cv2.FaceRecognizerSF.create(str(ROOT / "models/sface.onnx"), "")

# ---------------- ID-card template (shared with make_data.py) ----------------
CARD_W, CARD_H = 856, 540  # ID-1 aspect ratio (85.6 x 54 mm)
PHOTO_ROI = (40, 120, 250, 400)  # x0, y0, x1, y1
FIELDS = {"name": ("NAME", 290, 120), "dob": ("DATE OF BIRTH", 290, 220), "id_no": ("ID NUMBER", 290, 320)}
FIELD_ROIS = {k: (x - 10, y + 24, 830, y + 80) for k, (_, x, y) in FIELDS.items()}


# ---------------- 1. Image-quality gate ----------------
# ponytail: fixed thresholds set from simulated captures (p5 of live selfies); recalibrate on real device data
QT = dict(sharpness=40, bright_lo=40, bright_hi=210, contrast=25, glare=0.02)


def quality(img):
    s = min(1.0, 640 / max(img.shape[:2]))  # downscale only: upsampling fakes blur and fails sharp small selfies
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    q = dict(
        sharpness=float(cv2.Laplacian(g, cv2.CV_64F).var()),  # focus measure
        brightness=float(g.mean()),
        contrast=float(g.std()),
        glare=float(((hsv[..., 2] > 245) & (hsv[..., 1] < 40)).mean()),  # saturated, colourless pixels
    )
    q["ok"] = bool(q["sharpness"] > QT["sharpness"] and QT["bright_lo"] < q["brightness"] < QT["bright_hi"]
                   and q["contrast"] > QT["contrast"] and q["glare"] < QT["glare"])
    return q


# ---------------- 2. Document: detect, rectify, enhance, OCR ----------------
def order_corners(p):
    p = np.asarray(p, np.float32).reshape(4, 2)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.array([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]], np.float32)  # TL TR BR BL


def _quad_from_hull(c, s, size, min_frac=0.05):
    if cv2.contourArea(c) < min_frac * size:
        return None
    quad = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(quad) != 4:
        quad = cv2.approxPolyN(c, 4)  # best-fitting quadrilateral of the hull
    return order_corners(quad) / s


def find_card_baseline(img, debug=None):
    """Baseline: grey Gaussian blur -> Canny -> largest external contour -> quadrilateral."""
    s = 800 / max(img.shape[:2])
    g = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 30, 90)
    if debug is not None:
        debug["edges"] = edges
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _quad_from_hull(cv2.convexHull(max(cnts, key=cv2.contourArea)), s, edges.size) if cnts else None


LINK = 31  # dilation that links header text / card edges into one component


def find_card(img, debug=None):
    """Improved, two stages.
    1) coarse: median filter (kills table texture) -> Lab gradient magnitude -> Otsu -> dilate to link
       edges -> largest connected component -> hull of its edge pixels.
    2) refine: GrabCut colour segmentation seeded by the coarse box, so parts whose *colour* differs from
       the table (e.g. a dark-blue header on a dark table, weak in gradient) join the card.
    -> convex hull -> best-fit quadrilateral."""
    s = 800 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    med = cv2.medianBlur(small, 9)
    lab = cv2.cvtColor(med, cv2.COLOR_BGR2LAB).astype(np.float32)
    grad = np.max([cv2.magnitude(cv2.Sobel(lab[..., i], cv2.CV_32F, 1, 0), cv2.Sobel(lab[..., i], cv2.CV_32F, 0, 1))
                   for i in range(3)], axis=0)
    grad = np.clip(grad * 255 / np.percentile(grad, 99.5), 0, 255).astype(np.uint8)  # robust to glare peaks
    edges = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(edges, np.ones((LINK, LINK), np.uint8)))
    if n < 2:
        return None
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    hull = cv2.convexHull(cv2.findNonZero(((lbl == k) & (edges > 0)).astype(np.uint8)))

    # GrabCut: inside coarse hull = probable fg, a 12% ring around it = probable bg, rest = sure bg
    gc = np.zeros(edges.shape, np.uint8)  # GC_BGD
    x, y, w, h = cv2.boundingRect(hull)
    m = int(0.12 * max(w, h))
    gc[max(y - m, 0):y + h + m, max(x - m, 0):x + w + m] = cv2.GC_PR_BGD
    cv2.fillPoly(gc, [hull], cv2.GC_PR_FGD)
    cv2.grabCut(med, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
    fg = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg)
    if debug is not None:
        debug.update(edges=edges, region=fg)
    if n < 2:
        return _quad_from_hull(hull, s, edges.size)
    region = (lbl == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    return _quad_from_hull(cv2.convexHull(cv2.findNonZero(region)), s, edges.size)


def rectify(img, corners):
    dst = np.float32([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]])
    M = cv2.getPerspectiveTransform(np.float32(corners), dst)
    return cv2.warpPerspective(img, M, (CARD_W, CARD_H), flags=cv2.INTER_CUBIC)


def flat_field(g):
    """Illumination correction: closing with an SE larger than the strokes removes dark text and
    leaves the background/shading; dividing by it flattens shading gradients and glare."""
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    return cv2.divide(g, bg, scale=255), bg


def enhance_doc(card, debug=None):
    """Flat-field -> edge-preserving bilateral denoise -> unsharp mask. (Global Otsu binarisation was
    tested and hurts Tesseract by 4-6 pts on blurred strokes; see evaluate.py docablation.)"""
    g = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    flat, bg = flat_field(g)
    den = cv2.bilateralFilter(flat, 7, 30, 5)
    out = cv2.addWeighted(den, 2.0, cv2.GaussianBlur(den, (0, 0), 2), -1.0, 0)
    if debug is not None:
        debug.update(gray=g, background=bg, flat=flat, enhanced=out)
    return out


WHITELIST = {"name": "ABCDEFGHIJKLMNOPQRSTUVWXYZ ", "dob": "0123456789/", "id_no": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"}


def ocr_fields(card_gray, whitelist=True):
    out = {}
    for k, (x0, y0, x1, y1) in FIELD_ROIS.items():
        roi = cv2.copyMakeBorder(card_gray[y0:y1, x0:x1], 12, 12, 12, 12, cv2.BORDER_REPLICATE)
        cfg = "--psm 7" + (f' -c tessedit_char_whitelist="{WHITELIST[k]}"' if whitelist else "")
        out[k] = pytesseract.image_to_string(roi, config=cfg).strip()
    return out


def ocr_fullpage(img):
    """Baseline: OCR the raw capture and parse 'LABEL' -> next line."""
    lines = [l.strip() for l in pytesseract.image_to_string(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).splitlines() if l.strip()]
    out = {k: "" for k in FIELDS}
    for k, (label, _, _) in FIELDS.items():
        for i, l in enumerate(lines[:-1]):
            if label in l.upper():
                out[k] = lines[i + 1]
                break
    return out


def validate_fields(f):
    return bool(len(f.get("name", "")) >= 3
                and re.fullmatch(r"\d{2}/\d{2}/\d{4}", norm(f.get("dob", "")))
                and re.fullmatch(r"[A-Z]{3}\d{6}", norm(f.get("id_no", ""))))


def norm(s):
    return re.sub(r"\s+", " ", s.upper()).strip()


# ---------------- 3. Face: detect, enhance, align, match ----------------
def detect_face(img):
    h, w = img.shape[:2]
    _det.setInputSize((w, h))
    f = _det.detect(img)[1]
    return None if f is None else f[np.argmax(f[:, 2] * f[:, 3])]


def align(img, face):
    return _rec.alignCrop(img, face)  # 5-landmark similarity warp -> 112x112


def face_crop(img):
    f = detect_face(img)
    return None if f is None else align(img, f)


def enhance_face(img, debug=None):
    """Gentle restoration for low-res printed ID photos: bicubic x3 -> 1-99 percentile contrast stretch
    (undoes washed-out print) -> light NL-means (JPEG/print noise). CLAHE/strong sharpening were tested and
    *hurt* the CNN (pushes the image off the natural-photo distribution); see evaluate.py face."""
    up = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC) if max(img.shape[:2]) < 200 else img
    lo, hi = np.percentile(up, (1, 99))
    st = np.clip((up.astype(np.float32) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    out = cv2.fastNlMeansDenoisingColored(st, None, 3, 3, 7, 21)
    if debug is not None:
        debug.update(upsampled=up, stretched=st, denoised=out)
    return out


def enhance_face_aggressive(img):
    """Negative ablation: classic 'make it look better' chain (NL-means h=7, CLAHE, unsharp)."""
    up = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC) if max(img.shape[:2]) < 200 else img
    lab = cv2.cvtColor(cv2.fastNlMeansDenoisingColored(up, None, 7, 7, 7, 21), cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
    eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(eq, 1.6, cv2.GaussianBlur(eq, (0, 0), 2), -0.6, 0)


def emb_cnn(crop):
    return _rec.feature(crop)


def score_cnn(a, b):
    return float(_rec.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))


def emb_lbp(crop, grid=7):
    lbp = local_binary_pattern(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 8, 1, "uniform")
    h = np.concatenate([np.bincount(c.astype(int).ravel(), minlength=10)
                        for r in np.array_split(lbp, grid) for c in np.array_split(r, grid, axis=1)]).astype(np.float32)
    return h / h.sum()


def score_lbp(a, b):
    return float(-0.5 * np.sum((a - b) ** 2 / (a + b + 1e-9)))  # negative chi-square distance


SFACE_T, REJECT_T = 0.363, 0.25  # OpenCV-zoo defaults; overridden by `evaluate.py calib` (printed-ID domain)
_thr = ROOT / "models/threshold.json"
if _thr.exists():
    import json
    SFACE_T, REJECT_T = (json.loads(_thr.read_text())[k] for k in ("accept", "reject"))


# ---------------- 4. Liveness (presentation-attack detection) ----------------
def _lbp_hist(ch, P, R):
    return np.bincount(local_binary_pattern(ch, P, R, "uniform").astype(int).ravel(), minlength=P + 2) / ch.size


def live_feat_baseline(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return np.concatenate([_lbp_hist(g, 8, 1), _lbp_hist(g, 8, 2), _lbp_hist(g, 16, 2)])


def fft_feat(g, bins=24):
    """Radially-averaged log power spectrum: recapture adds halftone/moire peaks, removes fine detail."""
    F = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))))
    h, w = g.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
    idx = np.minimum((r * bins).astype(int), bins)
    prof = np.bincount(idx.ravel(), F.ravel(), bins + 1)[:bins] / np.bincount(idx.ravel(), minlength=bins + 1)[:bins]
    return prof / prof.max()


def live_feat_improved(crop):
    ycc, hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb), cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    chans = [ycc[..., i] for i in range(3)] + [hsv[..., i] for i in range(3)]
    tex = np.concatenate([np.concatenate([_lbp_hist(c, 8, 1), _lbp_hist(c, 8, 2)]) for c in chans])
    return np.concatenate([tex, fft_feat(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32))])


def load_liveness():
    p = ROOT / "models/liveness.pkl"
    return pickle.loads(p.read_bytes()) if p.exists() else None


# ---------------- 5. Fusion ----------------
def verify(selfie, id_capture, live_model=None):
    r, dbg = {"reasons": []}, {}
    r["q_selfie"], r["q_doc"] = quality(selfie), quality(id_capture)

    corners = find_card(id_capture, dbg)
    if corners is None:
        return dict(r, decision="REJECT", reasons=["no document found"]), dbg
    card = rectify(id_capture, corners)
    card_enh = enhance_doc(card, dbg)
    r["fields"] = ocr_fields(card_enh)
    r["doc_ok"] = validate_fields(r["fields"])

    x0, y0, x1, y1 = PHOTO_ROI
    id_photo = enhance_face(card[y0:y1, x0:x1])
    s_crop, d_crop = face_crop(selfie), face_crop(id_photo)
    dbg.update(corners=corners, card=card, id_photo=id_photo, selfie_crop=s_crop, id_crop=d_crop)
    if s_crop is None or d_crop is None:
        return dict(r, decision="REJECT", reasons=["face not found in " + ("selfie" if s_crop is None else "document")]), dbg
    r["match"] = score_cnn(emb_cnn(s_crop), emb_cnn(d_crop))

    live_model = live_model or load_liveness()
    r["live"] = float(live_model.decision_function([live_feat_improved(s_crop)])[0]) if live_model else None

    # ponytail: rule-based fusion with hand-set margins; learn a logistic fuser once real labelled sessions exist
    if r["match"] < REJECT_T:
        r["reasons"].append("face mismatch")
    if r["live"] is not None and r["live"] < -0.5:
        r["reasons"].append("presentation attack suspected")
    if r["reasons"]:
        return dict(r, decision="REJECT"), dbg
    soft = [m for m, bad in [("weak face match", r["match"] < SFACE_T), ("borderline liveness", r["live"] is not None and r["live"] < 0),
                             ("document fields invalid", not r["doc_ok"]), ("poor selfie quality", not r["q_selfie"]["ok"]),
                             ("poor document quality", not r["q_doc"]["ok"])] if bad]
    return dict(r, decision="REVIEW" if soft else "ACCEPT", reasons=soft), dbg


def demo_figure(selfie, id_capture, r, dbg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im
    cap = id_capture.copy()
    if "corners" in dbg:
        cv2.polylines(cap, [dbg["corners"].astype(int)], True, (0, 255, 0), 6)
    panels = [("ID capture + detected card", cap), ("edges + closing", dbg.get("edges")), ("rectified card", dbg.get("card")),
              ("illumination (closing)", dbg.get("background")), ("flat-field + NL-means", dbg.get("enhanced")),
              ("selfie", selfie), ("selfie aligned", dbg.get("selfie_crop")), ("ID photo enhanced+aligned", dbg.get("id_crop"))]
    fig, ax = plt.subplots(2, 5, figsize=(22, 8))
    for a, (t, im) in zip(ax.ravel(), panels):
        a.set_title(t), a.axis("off")
        if im is not None:
            a.imshow(rgb(im), cmap="gray")
    for a in ax.ravel()[len(panels):]:
        a.axis("off")
    txt = "\n".join([f"DECISION: {r['decision']}", *[f"- {x}" for x in r["reasons"]], "",
                     *[f"{k}: {v}" for k, v in r.get("fields", {}).items()],
                     f"match (cosine): {r.get('match', float('nan')):.3f}  [t={SFACE_T}]",
                     f"liveness score: {r.get('live')}",
                     f"selfie quality ok: {r['q_selfie']['ok']}", f"doc quality ok: {r['q_doc']['ok']}"])
    ax[1, 4].text(0, 0.5, txt, fontsize=11, family="monospace", va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=80)


if __name__ == "__main__":
    selfie, idcap = cv2.imread(sys.argv[1]), cv2.imread(sys.argv[2])
    res, dbg = verify(selfie, idcap)
    (ROOT / "results").mkdir(exist_ok=True)
    demo_figure(selfie, idcap, res, dbg, ROOT / "results/demo.png")
    print({k: v for k, v in res.items()})
