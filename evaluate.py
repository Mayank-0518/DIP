"""Experiments: baseline vs improved for each eKYC signal. Writes results/*.png and results/metrics.json.

python evaluate.py [doc|face|live|fusion|all] [--n N]
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import kyc
import make_data as md

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)


def save_metrics(key, val):
    p = RES / "metrics.json"
    m = json.loads(p.read_text()) if p.exists() else {}
    m[key] = val
    p.write_text(json.dumps(m, indent=1))


def table(rows, title):
    print(f"\n== {title} ==")
    keys = list(rows[0])
    print(" | ".join(f"{k:>14}" for k in keys))
    for r in rows:
        print(" | ".join(f"{v:>14.4f}" if isinstance(v, float) else f"{str(v):>14}" for v in r.values()))


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def verif_metrics(y, s):
    """y=1 genuine/live. Returns AUC, EER, TAR@FAR=1%."""
    fpr, tpr, thr = roc_curve(y, s)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return dict(auc=float(roc_auc_score(y, s)), eer=float((fpr[i] + 1 - tpr[i]) / 2),
                tar_at_far1=float(np.interp(0.01, fpr, tpr))), (fpr, tpr)


def quad_iou(a, b, shape):
    ma, mb = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    cv2.fillPoly(ma, [np.int32(a)], 1), cv2.fillPoly(mb, [np.int32(b)], 1)
    return (ma & mb).sum() / max((ma | mb).sum(), 1)


# ---------------------------------------------------------------- documents
def exp_doc(n):
    gt = json.loads((ROOT / "data/docs/gt.json").read_text())[:n]
    # ablation, one DIP step at a time; E uses ground-truth corners to separate OCR errors from detection errors
    configs = ["A_fullpage_raw", "B_canny_rect_raw", "C_improved_rect_raw", "D_improved_rect_enhanced", "E_oracle_rect_enhanced"]
    P = "D_improved_rect_enhanced"
    per = {c: {k: [] for k in kyc.FIELDS} for c in configs}
    cer = {c: [] for c in configs}
    iou = {"canny (baseline)": [], "median+Lab grad+GrabCut": []}
    empty = {k: "" for k in kyc.FIELDS}
    rows_doc, t = [], 0.0
    for g in gt:
        img = cv2.imread(str(ROOT / "data/docs" / g["file"]))
        grey = lambda c: cv2.cvtColor(kyc.rectify(img, c), cv2.COLOR_BGR2GRAY)
        pred = {"A_fullpage_raw": kyc.ocr_fullpage(img)}
        cb = kyc.find_card_baseline(img)
        pred["B_canny_rect_raw"] = empty if cb is None else kyc.ocr_fields(grey(cb), whitelist=False)
        t0 = time.time()
        ci = kyc.find_card(img)
        pred[P] = empty if ci is None else kyc.ocr_fields(kyc.enhance_doc(kyc.rectify(img, ci)))
        t += time.time() - t0
        pred["C_improved_rect_raw"] = empty if ci is None else kyc.ocr_fields(grey(ci), whitelist=False)
        pred["E_oracle_rect_enhanced"] = kyc.ocr_fields(kyc.enhance_doc(kyc.rectify(img, np.float32(g["corners"]))))
        for name, c in zip(iou, (cb, ci)):
            iou[name].append(0.0 if c is None else quad_iou(c, g["corners"], img.shape[:2]))
        for c in configs:
            for k, v in g["fields"].items():
                p = kyc.norm(pred[c][k])
                per[c][k].append(p == v)
                cer[c].append(edit_distance(p, v) / len(v))
        ok = all(kyc.norm(pred[P][k]) == v for k, v in g["fields"].items())
        rows_doc.append(dict(file=g["file"], iou=float(iou["median+Lab grad+GrabCut"][-1]), all_ok=ok,
                             quality_ok=kyc.quality(img)["ok"], **g["cond"], pred=pred[P], gt=g["fields"]))

    rows = [dict(config=c, **{f"acc_{k}": float(np.mean(per[c][k])) for k in kyc.FIELDS},
                 doc_all_ok=float(np.mean(np.all([per[c][k] for k in kyc.FIELDS], axis=0))),
                 mean_CER=float(np.mean(cer[c]))) for c in configs]
    table(rows, f"Document OCR ({len(gt)} synthetic ID captures; proposed = D, {t / len(gt):.2f}s/doc)")
    det = [dict(detector=k, mean_iou=float(np.mean(v)), rate_iou90=float(np.mean(np.array(v) > 0.9))) for k, v in iou.items()]
    table(det, "Card localisation")

    # failure analysis by capture condition, and does the quality gate predict OCR failure?
    by = []
    for name, sel in [("all", lambda r: True), ("glare", lambda r: r["glare"]), ("no glare", lambda r: not r["glare"]),
                      ("blur>1.2", lambda r: r["blur"] > 1.2), ("light bg", lambda r: r["light_bg"]),
                      ("gate pass", lambda r: r["quality_ok"]), ("gate reject", lambda r: not r["quality_ok"])]:
        s = [r for r in rows_doc if sel(r)]
        if s:
            by.append(dict(subset=name, n=len(s), det_rate=float(np.mean([r["iou"] > 0.9 for r in s])),
                           doc_all_ok=float(np.mean([r["all_ok"] for r in s]))))
    table(by, "Proposed (D) by capture condition")
    fails = [r for r in rows_doc if not r["all_ok"]][:8]
    for r in fails:
        print("FAIL", r["file"], {k: (r["pred"][k], r["gt"][k]) for k in r["gt"] if kyc.norm(r["pred"][k]) != r["gt"][k]}, "iou=%.2f" % r["iou"])
    save_metrics("document", dict(ocr=rows, detection=det, by_condition=by))

    # stage figure: one success, one failure
    picks = [r for r in rows_doc if r["all_ok"]][:1] + fails[:2]
    fig, ax = plt.subplots(len(picks), 7, figsize=(30, 4 * len(picks)), squeeze=False)
    for row, r in zip(ax, picks):
        img = cv2.imread(str(ROOT / "data/docs" / r["file"]))
        dbg = {}
        c = kyc.find_card(img, dbg)
        vis = img.copy()
        if c is not None:
            cv2.polylines(vis, [c.astype(int)], True, (0, 255, 0), 6)
            card = kyc.rectify(img, c)
            kyc.enhance_doc(card, dbg)
            x0, y0, x1, y1 = kyc.FIELD_ROIS["id_no"]
            dbg["ID-number ROI"] = dbg["enhanced"][y0:y1, x0:x1]
        for a, (t_, im) in zip(row, [(r["file"] + (" OK" if r["all_ok"] else " FAIL"), vis), ("Lab gradient + Otsu", dbg.get("edges")),
                                     ("GrabCut card region", dbg.get("region")),
                                     ("rectified grey", dbg.get("gray")), ("background (closing 31px)", dbg.get("background")),
                                     ("flat-field + bilateral + unsharp", dbg.get("enhanced")), ("ID-number ROI -> OCR", dbg.get("ID-number ROI"))]):
            a.set_title(t_), a.axis("off")
            if im is not None:
                a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im, cmap="gray")
    fig.tight_layout()
    fig.savefig(RES / "doc_stages.png", dpi=70)


def exp_docablation(n):
    """Contribution of each enhancement stage, with ground-truth corners so detection errors don't interfere."""
    gt = json.loads((ROOT / "data/docs/gt.json").read_text())[:n]
    unsharp = lambda g: cv2.addWeighted(g, 2.0, cv2.GaussianBlur(g, (0, 0), 2), -1.0, 0)
    otsu = lambda g: cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    flat = lambda g: kyc.flat_field(g)[0]
    V = {"raw grey": lambda g: g, "+ Otsu": otsu, "flat-field": flat, "flat + Otsu": lambda g: otsu(flat(g)),
         "flat + NL-means": lambda g: cv2.fastNlMeansDenoising(flat(g), h=12),
         "flat + bilateral + unsharp (proposed)": lambda g: unsharp(cv2.bilateralFilter(flat(g), 7, 30, 5))}
    acc = {k: [] for k in V}
    for g in gt:
        grey = cv2.cvtColor(kyc.rectify(cv2.imread(str(ROOT / "data/docs" / g["file"])), np.float32(g["corners"])), cv2.COLOR_BGR2GRAY)
        for name, f in V.items():
            p = kyc.ocr_fields(f(grey))
            acc[name] += [kyc.norm(p[k]) == v for k, v in g["fields"].items()]
    rows = [dict(enhancement=k, field_acc=float(np.mean(v))) for k, v in acc.items()]
    table(rows, f"Enhancement ablation (oracle corners, {len(gt)} cards)")
    save_metrics("doc_enhancement_ablation", rows)


# ---------------------------------------------------------------- face matching
def lfw_pairs():
    from sklearn.datasets import fetch_lfw_pairs
    p = fetch_lfw_pairs(subset="10_folds", color=True, resize=1.0, slice_=None, data_home=str(ROOT / "data"))
    return (p.pairs[..., ::-1] * 255).astype(np.uint8), p.target  # RGB float -> BGR uint8


def exp_face(n):
    pairs, y = lfw_pairs()
    idx = np.random.default_rng(1).permutation(len(y))[:n]
    pairs, y = pairs[idx], y[idx]
    rng = np.random.default_rng(2)
    conds = {
        "clean (upper bound)": lambda b: b,
        "printed, raw": lambda b: b,  # degraded image as-is
        "printed, bicubic x3": lambda b: cv2.resize(b, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
        "printed, CLAHE+unsharp": kyc.enhance_face_aggressive,
        "printed, proposed": kyc.enhance_face,
    }
    S = {(m, c): [] for m in ("LBP", "SFace") for c in conds}
    fta = {c: 0 for c in conds}
    example = None
    for (a, b) in pairs:
        ca = kyc.face_crop(a)
        bd = md.degrade_id_photo(b, rng)
        ea = None if ca is None else (kyc.emb_lbp(ca), kyc.emb_cnn(ca))
        for c, f in conds.items():
            cb = kyc.face_crop(f(b if c.startswith("clean") else bd))
            if ea is None or cb is None:
                fta[c] += 1
                S[("LBP", c)].append(-np.inf), S[("SFace", c)].append(-np.inf)
                continue
            S[("LBP", c)].append(kyc.score_lbp(ea[0], kyc.emb_lbp(cb)))
            S[("SFace", c)].append(kyc.score_cnn(ea[1], kyc.emb_cnn(cb)))
        if example is None and ca is not None:
            dbg = {}
            kyc.enhance_face(bd, dbg)
            example = (a, b, bd, dbg, ca)

    rows, fig, ax = [], *plt.subplots(1, 2, figsize=(13, 5.5))
    for k, m in enumerate(("LBP", "SFace")):
        for c in conds:
            s = np.array(S[(m, c)])
            s[np.isinf(s)] = s[~np.isinf(s)].min() - 1  # failure to acquire = lowest score
            met, (fpr, tpr) = verif_metrics(y, s)
            extra = dict(FAR_at_t=float("nan"), FRR_at_t=float("nan"))
            if m == "SFace":  # operating point at the published threshold -> shows need for recalibration
                extra = dict(FAR_at_t=float(np.mean(s[y == 0] >= kyc.SFACE_T)), FRR_at_t=float(np.mean(s[y == 1] < kyc.SFACE_T)))
            rows.append(dict(matcher=m, condition=c, **met, FTA=fta[c] / len(y), **extra))
            ax[k].plot(fpr, tpr, label=f"{c} (AUC {met['auc']:.3f})")
        ax[k].plot([0, 1], [0, 1], "k:", lw=0.8)
        ax[k].set(title=f"{m}: selfie vs ID photo (LFW, {len(y)} pairs)", xlabel="FAR", ylabel="TAR", xscale="log", xlim=(1e-3, 1))
        ax[k].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RES / "face_roc.png", dpi=90)
    table(rows, "Face verification")
    save_metrics("face", rows)

    a, b, bd, dbg, ca = example
    ims = [("selfie", a), ("ID photo (clean)", b), ("printed ID photo", bd), ("bicubic x3", dbg["upsampled"]),
           ("contrast stretch", dbg["stretched"]), ("light NL-means", dbg["denoised"]), ("aligned 112px", kyc.face_crop(dbg["denoised"])),
           ("CLAHE+unsharp (hurts)", kyc.enhance_face_aggressive(bd))]
    fig, ax = plt.subplots(1, len(ims), figsize=(3 * len(ims), 3.4))
    for x, (t_, im) in zip(ax, ims):
        x.set_title(t_), x.axis("off")
        if im is not None:
            x.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    fig.tight_layout()
    fig.savefig(RES / "face_stages.png", dpi=80)


# ---------------------------------------------------------------- liveness
def load_live_set():
    """Returns crops, labels (1=live), attack type, group (identity) for synthetic + any real captures."""
    X, y, kind, grp = [], [], [], []
    for root, classes in [(ROOT / "data/liveness_synth", ("live", "print", "replay")), (ROOT / "data/liveness_real", ("live", "spoof"))]:
        for c in classes:
            for f in sorted((root / c).glob("*.jpg")):
                crop = kyc.face_crop(cv2.imread(str(f)))
                if crop is not None:
                    X.append(crop), y.append(int(c == "live")), kind.append(f"{root.name}/{c}"), grp.append(f"{root.name}/{f.stem}")
    return X, np.array(y), np.array(kind), np.array(grp)


def pad_metrics(y, s):
    met, roc = verif_metrics(y, s)
    apcer, bpcer = float(np.mean(s[y == 0] >= 0)), float(np.mean(s[y == 1] < 0))
    return dict(APCER=apcer, BPCER=bpcer, ACER=(apcer + bpcer) / 2, AUC=met["auc"], EER=met["eer"]), roc


def svm():
    return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced"))


def exp_live(_n):
    crops, y, kind, grp = load_live_set()
    print("liveness samples:", dict(zip(*np.unique(kind, return_counts=True))))
    F = {"grey LBP (baseline)": np.array([kyc.live_feat_baseline(c) for c in crops])}
    imp = np.array([kyc.live_feat_improved(c) for c in crops])
    F["colour LBP (YCrCb+HSV)"] = imp[:, :-24]
    F["FFT spectrum only"] = imp[:, -24:]
    F["colour LBP + FFT (proposed)"] = imp

    ug = np.unique(grp)
    test_g = set(np.random.default_rng(3).permutation(ug)[: int(0.3 * len(ug))])  # identity-disjoint split
    te = np.array([g in test_g for g in grp])
    rows, fig, axr = [], *plt.subplots(figsize=(6.5, 5.5))
    for name, X in F.items():
        m = svm().fit(X[~te], y[~te])
        met, (fpr, tpr) = pad_metrics(y[te], m.decision_function(X[te]))
        rows.append(dict(features=name, protocol="intra", **met))
        axr.plot(fpr, tpr, label=f"{name} (AUC {met['AUC']:.3f})")
    axr.plot([0, 1], [0, 1], "k:", lw=0.8)
    axr.set(title="Liveness ROC (identity-disjoint test)", xlabel="APCER (spoof accepted)", ylabel="1 - BPCER")
    axr.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "liveness_roc.png", dpi=90)

    # cross-attack generalisation: train on one synthetic attack, test on the unseen one
    for tr_k, te_k in [("print", "replay"), ("replay", "print")]:
        tr = ~te & ((kind == "liveness_synth/live") | (kind == f"liveness_synth/{tr_k}"))
        ts = te & ((kind == "liveness_synth/live") | (kind == f"liveness_synth/{te_k}"))
        for name in ("grey LBP (baseline)", "colour LBP + FFT (proposed)"):
            m = svm().fit(F[name][tr], y[tr])
            rows.append(dict(features=name, protocol=f"{tr_k}->{te_k}", **pad_metrics(y[ts], m.decision_function(F[name][ts]))[0]))
    table(rows, "Liveness / presentation-attack detection")
    save_metrics("liveness", rows)

    final = svm().fit(imp, y)  # deployable model on all data
    (ROOT / "models/liveness.pkl").write_bytes(pickle.dumps(final))
    print("saved models/liveness.pkl")

    # what the features see: mean FFT profile per class
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k in np.unique(kind):
        ax.plot(F["FFT spectrum only"][kind == k].mean(0), label=k)
    ax.set(title="Mean radial log-spectrum (normalised)", xlabel="radial frequency bin (low -> high)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "liveness_fft.png", dpi=90)


def pipeline_match_scores(pairs, y, rng):
    """Selfie-vs-card match score through the real pipeline (capture -> find_card -> rectify -> ID photo)."""
    out = []
    for a, b in pairs:
        cap, _, _ = md.capture_card(md.make_card(b, md.random_fields(rng), rng), rng)
        r, _ = kyc.verify(a, cap, None)
        out.append(r.get("match", np.nan))
    return np.array(out)


def exp_calib(n):
    """Set accept/reject thresholds on printed-ID-domain pairs disjoint from fusion's test pairs."""
    pairs, y = lfw_pairs()
    idx = np.random.default_rng(1).permutation(len(y))[2000:2000 + n]
    s = pipeline_match_scores(pairs[idx], y[idx], np.random.default_rng(5))
    yy = y[idx][~np.isnan(s)]
    s = s[~np.isnan(s)]
    t = dict(accept=float(np.percentile(s[yy == 0], 99)),  # FAR = 1% on impostors
             reject=float(np.percentile(s[yy == 1], 2)),  # only 2% of genuine fall below
             n=int(len(s)), n_genuine=int((yy == 1).sum()), failed_to_acquire=int(n - len(s)))
    assert t["reject"] < t["accept"], t
    (ROOT / "models/threshold.json").write_text(json.dumps(t, indent=1))
    print("thresholds:", t, "| default was 0.363 / 0.25")
    save_metrics("calibration", t)


# ---------------------------------------------------------------- end-to-end fusion
def exp_fusion(n):
    pairs, y = lfw_pairs()
    rng = np.random.default_rng(4)
    idx = np.random.default_rng(1).permutation(len(y))[3000:3000 + n]  # disjoint from face-exp and calib pairs
    live_model = kyc.load_liveness()
    out, t = {"genuine": [], "impostor": [], "spoof": []}, []
    for j, i in enumerate(idx):
        a, b = pairs[i]
        kind = "impostor" if y[i] == 0 else ("spoof" if j % 2 else "genuine")
        if kind == "spoof":
            a = (md.spoof_print if rng.random() < 0.5 else md.spoof_replay)(a, rng)
        cap, _, _ = md.capture_card(md.make_card(b, md.random_fields(rng), rng), rng)
        t0 = time.time()
        r, _ = kyc.verify(a, cap, live_model)
        t.append(time.time() - t0)
        out[kind].append(r["decision"])
        if j == 0:
            kyc.demo_figure(a, cap, r, _, RES / "fusion_example.png")
    rows = [dict(attempt=k, n=len(v), **{d: float(np.mean([x == d for x in v])) for d in ("ACCEPT", "REVIEW", "REJECT")})
            for k, v in out.items() if v]
    table(rows, f"End-to-end decisions (mean {np.mean(t):.2f}s/verification)")
    save_metrics("fusion", dict(rows=rows, sec_per_verification=float(np.mean(t))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", nargs="?", default="all", choices=["doc", "docablation", "face", "live", "fusion", "calib", "all"])
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    defaults = dict(doc=200, docablation=100, face=2000, live=None, calib=500, fusion=300)
    for e in (["doc", "docablation", "face", "live", "calib", "fusion"] if a.exp == "all" else [a.exp]):
        globals()[f"exp_{e}"](a.n or defaults[e])
