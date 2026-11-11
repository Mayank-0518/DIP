"""Collect real liveness data with a webcam (only with the consent of everyone captured).

python capture.py        keys:  l = save LIVE frame   s = save SPOOF frame (photo/phone screen held up)   q = quit
Frames go to data/liveness_real/{live,spoof}/; evaluate.py live picks them up automatically.
"""
import time
from pathlib import Path

import cv2

out = Path(__file__).parent / "data/liveness_real"
for c in ("live", "spoof"):
    (out / c).mkdir(parents=True, exist_ok=True)
cam = cv2.VideoCapture(0)
while True:
    ok, frame = cam.read()
    if not ok:
        break
    cv2.imshow("l=live  s=spoof  q=quit", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k in (ord("l"), ord("s")):
        c = "live" if k == ord("l") else "spoof"
        cv2.imwrite(str(out / c / f"{c}_{time.time_ns()}.jpg"), frame)
        print("saved", c, len(list((out / c).glob("*.jpg"))))
cam.release()
cv2.destroyAllWindows()
