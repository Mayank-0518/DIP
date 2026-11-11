"""Synthetic data with exact ground truth, built from LFW faces (identity-disjoint where it matters).

  data/docs/        ID-card photos on a table (perspective, shading, glare, blur, noise, JPEG) + gt.json
  data/liveness_synth/{live,print,replay}/   live faces and simulated print / screen-replay attacks

python make_data.py [--cards 200] [--faces 600]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from kyc import CARD_H, CARD_W, FIELDS, PHOTO_ROI

ROOT = Path(__file__).parent
LFW = ROOT / "data/lfw_home/lfw_funneled"
FONT = "/System/Library/Fonts/Supplemental/Arial{}.ttf"
FIRST = "ARJUN PRIYA RAHUL ANANYA VIKRAM MEERA KABIR ISHA ROHAN NEHA SAMEER DIVYA ADITYA KAVYA NIKHIL POOJA".split()
LAST = "SHARMA VERMA IYER REDDY NAIR GUPTA MEHTA RAO KAPOOR JOSHI DAS PATEL SINGH BOSE MENON KULKARNI".split()


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT.format(" Bold" if bold else ""), size)
    except OSError:
        return ImageFont.load_default(size)


def jpeg(img, q):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])[1], cv2.IMREAD_COLOR)


def u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def random_fields(rng):
    return dict(name=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                dob=f"{rng.integers(1, 29):02d}/{rng.integers(1, 13):02d}/{rng.integers(1955, 2006)}",
                id_no="".join(rng.choice(list("ABCDEFGHJKLMNPRSTUVWXYZ"), 3)) + f"{rng.integers(0, 10**6):06d}")


def make_card(face, fields, rng):
    card = np.full((CARD_H, CARD_W, 3), (236, 238, 232), np.uint8)
    xx = np.arange(CARD_W)
    for k in range(14):  # light guilloche security pattern
        ys = (CARD_H / 14 * k + 18 * np.sin(xx / rng.uniform(25, 60) + k)).astype(np.int32)
        cv2.polylines(card, [np.stack([xx, ys], 1)], False, (215, 222, 205), 1, cv2.LINE_AA)
    cv2.rectangle(card, (0, 0), (CARD_W, 90), (110, 60, 20), -1)
    x0, y0, x1, y1 = PHOTO_ROI
    fh, fw = face.shape[:2]  # LFW face sits in the centre; take a 3:4 portrait crop
    crop = face[int(fh * .18):int(fh * .86), int(fw * .24):int(fw * .76)]
    card[y0:y1, x0:x1] = cv2.resize(crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    d.text((40, 25), "DIPLAND NATIONAL IDENTITY CARD", font=font(36, True), fill=(255, 255, 255))
    for k, (label, x, y) in FIELDS.items():
        d.text((x, y), label, font=font(20), fill=(90, 90, 90))
        d.text((x, y + 30), fields[k], font=font(36, True), fill=(20, 20, 30))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def capture_card(card, rng, W=1280, H=960):
    """Photograph the card: table texture, perspective, shading, glare, defocus, noise, JPEG."""
    light_bg = rng.random() < 0.2
    tex = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), rng.uniform(2, 15))
    base = rng.uniform(170, 215) if light_bg else rng.uniform(50, 130)
    bg = u8(base + 25 * tex[..., None] / tex.std() + rng.uniform(-20, 20, 3))

    w = rng.uniform(0.55, 0.75) * W
    h = w * CARD_H / CARD_W
    a = np.deg2rad(rng.uniform(-20, 20))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    box = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) @ R.T
    dst = (box + [W / 2 + rng.uniform(-80, 80), H / 2 + rng.uniform(-60, 60)] + rng.normal(0, 0.03 * w, (4, 2))).astype(np.float32)
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]]), dst)
    warped = cv2.warpPerspective(card, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones((CARD_H, CARD_W), np.float32), M, (W, H))[..., None]
    img = warped * mask + bg * (1 - mask)

    yy, xx = np.mgrid[:H, :W] / max(H, W)
    t = rng.uniform(0, 2 * np.pi)
    img *= (rng.uniform(0.6, 0.85) + rng.uniform(0.2, 0.5) * (np.cos(t) * xx + np.sin(t) * yy))[..., None]
    glare = rng.random() < 0.4
    if glare:
        cx, cy, sg = rng.uniform(.2, .8) * W, rng.uniform(.2, .8) * H, rng.uniform(40, 110)
        img += rng.uniform(120, 230) * np.exp(-((xx * max(H, W) - cx) ** 2 + (yy * max(H, W) - cy) ** 2) / (2 * sg ** 2))[..., None]
    blur = rng.uniform(0, 2.2)
    if blur > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    noise, q = rng.uniform(2, 10), rng.uniform(40, 90)
    img = jpeg(u8(img + rng.normal(0, noise, img.shape)), q)
    return img, order_pts(dst), dict(light_bg=bool(light_bg), glare=bool(glare), blur=round(blur, 2), noise=round(noise, 1), jpeg=int(q))


def order_pts(p):
    from kyc import order_corners
    return order_corners(p).tolist()


def degrade_id_photo(img, rng, scale=0.3):
    """Printed-and-scanned ID photo: low resolution, ink blur, washed-out contrast, noise, JPEG."""
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 0.7).astype(np.float32)
    small = small * rng.uniform(0.55, 0.75) + rng.uniform(40, 70) + rng.normal(0, 6, small.shape)
    return jpeg(u8(small), rng.uniform(25, 45))


# ---- presentation attacks (synthetic; replace/augment with real captures from capture.py) ----
def camera(img, rng):
    """Shared capture pipeline so both classes carry the same camera noise / compression."""
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), rng.uniform(0.3, 1.0))
    img *= rng.uniform(0.85, 1.15)
    return jpeg(u8(img + rng.normal(0, rng.uniform(2, 5), img.shape)), rng.uniform(70, 92))


def spoof_print(face, rng):
    f = face.astype(np.float32)
    f = 0.7 * f + 0.3 * f.mean(2, keepdims=True)  # ink gamut: desaturation
    f = f * rng.uniform(0.65, 0.8) + rng.uniform(25, 45)  # paper: reduced dynamic range
    h, w = f.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    p, a = rng.uniform(2.2, 3.5), rng.uniform(0, np.pi / 4)
    u, v = xx * np.cos(a) + yy * np.sin(a), -xx * np.sin(a) + yy * np.cos(a)
    f *= (0.88 + 0.12 * np.cos(2 * np.pi * u / p) * np.cos(2 * np.pi * v / p))[..., None]  # halftone screen
    f += rng.normal(0, 4, (h, w, 1))  # paper grain
    return camera(f, rng)


def spoof_replay(face, rng):
    h, w = face.shape[:2]
    big = cv2.resize(face, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST).astype(np.float32)
    sub = np.full_like(big, 0.88)  # faint sub-pixel modulation
    for c in range(3):  # RGB-stripe sub-pixel layout (BGR order)
        sub[:, 2 - c::3, c] = 1.0
    big *= sub * rng.uniform(1.0, 1.2)
    big[..., 0] += rng.uniform(5, 20)  # LCD blue cast
    M = cv2.getRotationMatrix2D((big.shape[1] / 2, big.shape[0] / 2), rng.uniform(-4, 4), rng.uniform(0.97, 1.03))
    big = cv2.warpAffine(big, M, big.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
    f = cv2.resize(big, (w, h), interpolation=cv2.INTER_LINEAR)  # resampling the pixel grid -> moire
    yy, xx = np.mgrid[:h, :w] / h
    f += rng.uniform(10, 40) * (xx + yy)[..., None] / 2  # screen specular sheen
    return camera(f, rng)


def lfw_people():
    return sorted(p for p in LFW.iterdir() if p.is_dir())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=int, default=200)
    ap.add_argument("--faces", type=int, default=600)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    people = lfw_people()
    rng.shuffle(people)

    out = ROOT / "data/docs"
    out.mkdir(parents=True, exist_ok=True)
    gt = []
    for i, person in enumerate(people[:args.cards]):
        face = cv2.imread(str(sorted(person.glob("*.jpg"))[0]))
        fields = random_fields(rng)
        img, corners, cond = capture_card(make_card(face, fields, rng), rng)
        cv2.imwrite(str(out / f"id_{i:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        gt.append(dict(file=f"id_{i:04d}.jpg", person=person.name, fields=fields, corners=corners, cond=cond))
    (out / "gt.json").write_text(json.dumps(gt, indent=1))
    print(f"{len(gt)} ID captures -> {out}")

    out = ROOT / "data/liveness_synth"
    for c in ("live", "print", "replay"):
        (out / c).mkdir(parents=True, exist_ok=True)
    for i, person in enumerate(people[:args.faces]):  # one image per identity; attack type alternates
        face = cv2.imread(str(sorted(person.glob("*.jpg"))[0]))
        cv2.imwrite(str(out / "live" / f"{person.name}.jpg"), camera(face, rng))
        kind, fn = ("print", spoof_print) if i % 2 else ("replay", spoof_replay)
        cv2.imwrite(str(out / kind / f"{person.name}.jpg"), fn(face, rng))
    print(f"{args.faces} live + {args.faces} spoof faces -> {out}")
