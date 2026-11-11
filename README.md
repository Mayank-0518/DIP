# Multimodal eKYC with Face, Document and Liveness Signals (DIP project, topic 029)

A user submits a **selfie** and a **photo of an ID card**. The pipeline localises and rectifies the card, reads its
fields, matches the selfie to the ID photo, checks the selfie for presentation attacks, and fuses the signals into
ACCEPT / REVIEW / REJECT. Every DIP stage is paired with a baseline and a measured ablation.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
brew install tesseract                      # OCR engine (any Tesseract >= 5)
# models/yunet.onnx, models/sface.onnx: OpenCV Zoo (face_detection_yunet_2023mar, face_recognition_sface_2021dec)
.venv/bin/python make_data.py               # synthetic ID captures + print/replay spoofs (needs LFW via scikit-learn)
.venv/bin/python evaluate.py all            # doc, docablation, face, live, calib, fusion  -> results/
.venv/bin/streamlit run app.py              # web demo: upload card + selfie (or webcam), see decision + all DIP stages
.venv/bin/python kyc.py selfie.jpg id.jpg   # same, command line -> results/demo.png
.venv/bin/python capture.py                 # optional: collect real liveness frames (with consent)
```

## Pipeline (DIP stages in **bold**)

```
ID photo ─► quality gate ─► **median + Lab gradient + Otsu** ─► **connected components** ─► **GrabCut** ─► hull→quad
            (Laplacian var,                                                                                  │
             brightness,                                                                        **perspective warp**
             contrast, glare)                                                                                │
                                          **flat-field (morph. closing) → bilateral → unsharp** ─► Tesseract ─► regex validation
                                                                                                             │
                                    card photo region ─► **bicubic ×3 → contrast stretch → light NL-means**  │
                                                          ─► YuNet detect ─► 5-pt align ─► SFace CNN ─┐      │
Selfie ─► YuNet ─► align ─┬─► SFace embedding ──────────────────────────────────────────────────────────┴─► cosine match
                          └─► **YCrCb/HSV LBP + FFT radial spectrum** ─► SVM ─► liveness score
                                                                                  │
                                           rule-based fusion (calibrated thresholds) ─► ACCEPT / REVIEW / REJECT
```

## Data (all reproducible, no personal data)

- **Faces:** LFW (6,000 verification pairs, via scikit-learn). Identity-disjoint splits for liveness.
- **ID cards:** synthetic template ("DIPLAND" card) with random name / DOB / ID number and an LFW portrait, photographed
  with random perspective, shading, glare, defocus, noise, JPEG and table texture (200 captures, exact ground truth).
- **Printed-ID-photo degradation:** low-res, ink blur, washed-out contrast, JPEG.
- **Spoofs:** simulated print (desaturation, halftone, reduced range) and screen replay (sub-pixel grid, moiré, sheen),
  passed through the same camera model as live images (600 live, 300 print, 300 replay).

## Results

**1. Document: fully-correct fields per capture (200 captures)**

| Config | name | DOB | ID no. | all 3 correct | char error |
|---|---|---|---|---|---|
| A: OCR the raw full photo | 14% | 8% | 8% | 3% | 0.88 |
| B: Canny card detector + OCR | 1% | 1% | 0.5% | 0.5% | 1.00 |
| C: proposed detector, no enhancement | 66% | 67% | 65% | 59% | 0.31 |
| **D: proposed detector + enhancement** | **74%** | **74%** | **73%** | **71%** | **0.23** |
| E: ground-truth corners + enhancement | 99.5% | 100% | 97% | 97% | 0.002 |

Card localisation IoU: Canny baseline 0.49 (4.5% of cards > 0.9) vs proposed 0.90 (77% > 0.9). **The detector, not
OCR, is the bottleneck (D vs E).** Enhancement ablation with oracle corners (100 cards): raw grey 0.90, + Otsu 0.89,
flat-field 0.97, **flat + bilateral + unsharp 0.98**. Global binarisation does not help Tesseract on blurred strokes.

**2. Face verification, selfie vs printed ID photo (SFace, 2,000 LFW pairs)**

| ID-photo preprocessing | AUC | EER | TAR @ FAR 1% |
|---|---|---|---|
| clean photo (upper bound) | 0.997 | 0.9% | 99.0% |
| printed, raw | 0.953 | 10.5% | 71.3% |
| printed, CLAHE + unsharp | 0.947 | 13.1% | 61.1% |
| **printed, proposed (stretch + light NL-means)** | **0.971** | **8.4%** | **75.5%** |

The LBP handcrafted matcher is near chance (AUC 0.54 to 0.61). Aggressive "beautifying" (CLAHE, sharpening) makes the
CNN **worse** than no enhancement. The published SFace threshold (0.363) rejects 56 to 62% of genuine printed pairs,
so thresholds were recalibrated (below).

**3. Liveness (identity-disjoint test, synthetic attacks)**

| Features | APCER | BPCER | AUC |
|---|---|---|---|
| grey LBP (baseline) | 46% | 26% | 0.72 |
| colour LBP (YCrCb + HSV) | 4% | 7% | 0.98 |
| FFT spectrum only | 45% | 24% | 0.68 |
| **colour LBP + FFT** | **4%** | **8%** | **0.98** |

Cross-attack (train on one attack, test on the other): AUC only 0.64 / 0.73 and APCER 100%. **The detector does not
generalise to unseen attack types.** The FFT features add almost nothing over colour LBP.

**4. End-to-end fusion (300 attempts, pairs disjoint from calibration and from the face experiment)**

Thresholds set on 480 separate pairs through the real pipeline: accept = 1% FAR on impostors (0.233), reject = 2nd
percentile of genuine (0.163).

| Attempt | n | ACCEPT | REVIEW | REJECT |
|---|---|---|---|---|
| genuine | 82 | 33% | 59% | 9% |
| impostor | 152 | 1.3% | 9.9% | 88.8% |
| spoof | 66 | 1.5% | 9.1% | 89.4% |

~0.8 s per verification on CPU.

## Success and failure cases (`results/doc_stages.png`, `face_stages.png`, `fusion_example.png`)

- **Card detection fails on light backgrounds** (IoU worst-case, 55% detected vs 77% overall): the card edge has little
  contrast against the table. GrabCut can also leak into the table and the fitted quad degenerates (a triangle), giving
  a black rectified card and empty OCR.
- **Blur is the next OCR failure:** with a good quad, digits are still misread under heavy defocus ("MPHS37270",
  15/11 read as 18/11). A fixed quality gate does not predict these failures well (gate-pass 72% vs gate-reject 69% fully correct).
- **Genuine users are over-flagged** (59% REVIEW): mostly invalid OCR fields and blurry document photos, not face
  matching. This is the safe failure direction, but it costs usability.
- **Quality gate lesson:** my first gate rescaled small selfies *up* to 640 px, which fakes blur and flagged most good selfies;
  fixed by down-scaling only.

## Limitations (please state these in the report)

- Documents, spoofs and card photos are **synthetic**; real-world numbers will be lower. Liveness in particular needs
  real print/replay captures (`capture.py`) before any claim about real attacks.
- The liveness model overfits to the simulated attack types (see cross-attack row).
- Quality-gate thresholds were set from simulated captures; fusion margins are hand-set rules, not learned.
- Single document template; OCR field positions are fixed to it. Impostor/spoof accepts of about 1% each are small
  counts (2/152 and 1/66), so the confidence intervals are wide.
- No deepfake detection; "liveness" here means passive print/replay detection only.

## Files

`kyc.py` pipeline | `make_data.py` synthetic data | `evaluate.py` experiments | `capture.py` real-data capture |
`results/` figures, `metrics.json`, logs | `models/` ONNX models, liveness SVM, calibrated thresholds
