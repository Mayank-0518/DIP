"""Web demo:  streamlit run app.py"""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import kyc

DEMO = Path(__file__).parent / "demo"
SAMPLES = {"Genuine user": "selfie_genuine.jpg", "Impostor (someone else's face)": "selfie_impostor.jpg",
           "Print attack (photo of a photo)": "selfie_print_attack.jpg", "Screen replay attack": "selfie_replay_attack.jpg"}
COLOR = {"ACCEPT": "green", "REVIEW": "orange", "REJECT": "red"}

st.set_page_config(page_title="eKYC demo", layout="wide")
st.title("Multimodal eKYC: document + face + liveness")


def read(f):
    return cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)


@st.cache_resource
def live_model():
    return kyc.load_liveness()


mode = st.radio("Input", ["Sample scenarios", "Upload / webcam"], horizontal=True)
selfie = card = None
if mode == "Sample scenarios":
    pick = st.selectbox("Scenario (same ID card, different selfie)", list(SAMPLES))
    selfie, card = cv2.imread(str(DEMO / SAMPLES[pick])), cv2.imread(str(DEMO / "id_card.jpg"))
else:
    c1, c2 = st.columns(2)
    cf = c1.file_uploader("ID card photo (jpg/png)", ["jpg", "jpeg", "png"])
    sf = c2.file_uploader("Selfie", ["jpg", "jpeg", "png"])
    cam = c2.camera_input("...or take a selfie")
    sf = cam or sf
    if cf and sf:
        card, selfie = read(cf), read(sf)
    st.caption("Field reading is tuned to the demo card layout (see demo/id_card.jpg); face match and liveness work on any photo.")

if selfie is not None and card is not None:
    with st.spinner("Processing..."):
        res, dbg = kyc.verify(selfie, card, live_model())
    st.markdown(f"## Decision: :{COLOR[res['decision']]}[{res['decision']}]")
    if res["reasons"]:
        st.write("Reasons: " + ", ".join(res["reasons"]))
    m1, m2, m3 = st.columns(3)
    m1.metric("Face match (cosine)", f"{res['match']:.3f}" if "match" in res else "n/a", f"threshold {kyc.SFACE_T:.2f}", delta_color="off")
    m2.metric("Liveness score", f"{res['live']:.2f}" if res.get("live") is not None else "n/a", "live if > 0", delta_color="off")
    m3.metric("Document fields valid", "yes" if res.get("doc_ok") else "no")
    if "fields" in res:
        st.table({"field": list(res["fields"]), "OCR result": list(res["fields"].values())})
    with tempfile.NamedTemporaryFile(suffix=".png") as t:
        kyc.demo_figure(selfie, card, res, dbg, t.name)
        st.image(t.name, caption="Processing stages: card detection, rectification, illumination correction, face alignment")
    st.caption(f"Selfie quality ok: {res['q_selfie']['ok']} | document quality ok: {res['q_doc']['ok']}")
