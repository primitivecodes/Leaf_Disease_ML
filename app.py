"""Streamlit UI: uptime, visualizations, predict, upload, train/retrain."""
import os, time, random
import numpy as np, pandas as pd, requests, streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Leaf Disease ML", layout="wide")
page = st.sidebar.radio("Menu", ["Dashboard", "Visualizations", "Predict", "Upload & Retrain"])

def get(path):
    try:
        return requests.get(f"{API}{path}", timeout=10).json()
    except Exception:
        return None

if page == "Dashboard":
    st.title("Model Dashboard")
    h = get("/health")
    if h:
        c1, c2, c3 = st.columns(3)
        c1.metric("API status", h["status"].upper())
        c2.metric("Uptime (min)", round(h["uptime_seconds"] / 60, 1))
        c3.metric("Model loaded", "Yes" if h["model_loaded"] else "No")
    else:
        st.error("API unreachable")
    m = get("/metrics")
    if m and "accuracy" in m:
        cols = st.columns(5)
        for c, k in zip(cols, ["accuracy", "precision", "recall", "f1", "loss"]):
            c.metric(k.capitalize(), round(m[k], 4))
        st.caption(f"Last trained: {m['trained_at']}")
        cm = np.array(m["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Greens")
        ax.set_xticks(range(len(m["classes"]))); ax.set_yticks(range(len(m["classes"])))
        ax.set_xticklabels(m["classes"], rotation=45, ha="right"); ax.set_yticklabels(m["classes"])
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        st.pyplot(fig)

elif page == "Visualizations":
    st.title("Dataset Insights")
    st.caption("Computed on a sample of up to 60 images per class from data/train")
    root = "data/train"
    classes = [c for c in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, c))] if os.path.isdir(root) else []
    if not classes:
        st.warning("No training data found.")
    else:
        @st.cache_data
        def features(classes):
            rows = []
            for c in classes:
                files = [f for f in os.listdir(f"{root}/{c}") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                for f in random.sample(files, min(60, len(files))):
                    a = np.array(Image.open(f"{root}/{c}/{f}").convert("RGB").resize((128, 128)), dtype=float)
                    r, g, b = a[..., 0].mean(), a[..., 1].mean(), a[..., 2].mean()
                    rows.append({"class": c, "R": r, "G": g, "B": b, "brightness": a.mean(),
                                 "green_ratio": g / (r + g + b + 1e-9), "total_files": len(files)})
            return pd.DataFrame(rows)
        df = features(tuple(classes))
        counts = df.groupby("class")["total_files"].first()
        st.subheader("1. Class distribution"); st.bar_chart(counts)
        st.markdown("**Story:** shows whether classes are balanced. Imbalance biases the model toward majority classes, so use per-class recall/F1, not just accuracy.")
        st.subheader("2. Green ratio per class"); st.bar_chart(df.groupby("class")["green_ratio"].mean())
        st.markdown("**Story:** healthy leaves are greener; blight and mould shift colour towards brown/yellow, lowering green share. Colour is a strong cue the CNN can exploit.")
        st.subheader("3. Brightness distribution per class")
        fig, ax = plt.subplots()
        ax.boxplot([df[df["class"] == c]["brightness"] for c in classes], tick_labels=classes)
        plt.xticks(rotation=30, ha="right"); ax.set_ylabel("Mean pixel intensity")
        st.pyplot(fig)
        st.markdown("**Story:** wide overlap means lighting varies within classes, which is why brightness augmentation and a pretrained backbone help generalisation.")

elif page == "Predict":
    st.title("Predict a leaf image")
    f = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    if f:
        st.image(f, width=300)
        if st.button("Predict"):
            r = requests.post(f"{API}/predict", files={"file": (f.name, f.getvalue())})
            if r.ok:
                res = r.json()
                st.success(f"{res['label']} ({res['confidence']:.1%})")
                st.bar_chart(pd.Series(res["probabilities"]))
            else:
                st.error(r.text)

else:
    st.title("Upload data & retrain")
    s = get("/data/stats")
    if s:
        st.write("Training images:", s["train"]); st.write("Pending uploads:", s["pending_uploads"])
    label = st.text_input("Class label for these images (new labels create a new class)")
    files = st.file_uploader("Bulk upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if st.button("Upload") and files and label:
        r = requests.post(f"{API}/upload", data={"label": label},
                          files=[("files", (x.name, x.getvalue())) for x in files])
        st.json(r.json())
    st.divider()
    if st.button("Trigger retraining"):
        r = requests.post(f"{API}/retrain")
        st.info(r.json().get("message", r.text) if r.ok else r.text)
    st.json(get("/retrain/status"))
    if st.button("Refresh status"):
        st.rerun()
