"""Streamlit UI: uptime, visualizations, predict, upload, train/retrain."""
import hashlib, hmac, os, time, random
import numpy as np, pandas as pd, requests, streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

API = os.getenv("API_URL", "http://localhost:8000")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
UI_USERNAME = os.getenv("UI_USERNAME", "")
UI_PASSWORD = os.getenv("UI_PASSWORD", "")
st.set_page_config(page_title="Leaf Disease ML", layout="wide")

def require_ui_login():
    if not UI_USERNAME or not UI_PASSWORD:
        st.error("UI login is not configured. Set UI_USERNAME and UI_PASSWORD before starting Streamlit.")
        st.stop()
    if st.session_state.get("ui_authenticated"):
        if st.sidebar.button("Log out"):
            st.session_state.pop("ui_authenticated", None)
            st.rerun()
        return
    st.title("Leaf Disease ML — Sign in")
    with st.form("ui_login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        valid_user = hmac.compare_digest(username.encode(), UI_USERNAME.encode())
        valid_password = hmac.compare_digest(password.encode(), UI_PASSWORD.encode())
        if valid_user and valid_password:
            st.session_state["ui_authenticated"] = True
            st.rerun()
        st.error("Incorrect username or password")
    st.stop()

require_ui_login()
page = st.sidebar.radio("Menu", ["Dashboard", "Visualizations", "Predict", "Upload & Retrain"])

def admin_headers():
    return {"X-API-Key": ADMIN_API_KEY} if ADMIN_API_KEY else {}

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
    st.subheader("Production evaluation")
    production = get("/metrics/production")
    if production and production["count"]:
        st.caption(f"Based on {production['count']} predictions with submitted verified labels; updated {production['updated_at']}.")
        cols = st.columns(5)
        for c, k in zip(cols, ["accuracy", "precision", "recall", "f1", "macro_f1"]):
            c.metric(k.replace("_", " ").capitalize(), round(production[k], 4))
        st.caption("These metrics are feedback-based and separate from the held-out offline test metrics.")
    else:
        st.info("No labeled production feedback has been recorded yet.")

elif page == "Visualizations":
    st.title("Dataset Insights")
    st.caption("Computed on a sample of up to 60 images per class from data/train")
    root = "data/train"
    classes = [c for c in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, c))] if os.path.isdir(root) else []
    if not classes:
        st.warning("No training data found.")
    else:
        @st.cache_data
        def features(classes, directory_versions):
            rows = []
            for c in classes:
                files = [f for f in os.listdir(f"{root}/{c}") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                for f in random.sample(files, min(60, len(files))):
                    a = np.array(Image.open(f"{root}/{c}/{f}").convert("RGB").resize((128, 128)), dtype=float)
                    r, g, b = a[..., 0].mean(), a[..., 1].mean(), a[..., 2].mean()
                    rows.append({"class": c, "R": r, "G": g, "B": b, "brightness": a.mean(),
                                 "green_ratio": g / (r + g + b + 1e-9), "total_files": len(files)})
            return pd.DataFrame(rows)
        directory_versions = tuple(
            (c, os.stat(os.path.join(root, c)).st_mtime_ns) for c in classes
        )
        df = features(tuple(classes), directory_versions)
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
        image_bytes = f.getvalue()
        image_id = hashlib.sha256(image_bytes).hexdigest()
        if st.button("Predict"):
            r = requests.post(
                f"{API}/predict", files={"file": (f.name, image_bytes)}, timeout=60
            )
            if r.ok:
                st.session_state["last_prediction"] = {"image_id": image_id, "result": r.json()}
            else:
                st.error(r.text)
        saved = st.session_state.get("last_prediction", {})
        res = saved.get("result") if saved.get("image_id") == image_id else None
        if res:
            st.success(f"{res['label']} ({res['confidence']:.1%})")
            st.bar_chart(pd.Series(res["probabilities"]))
            metrics = get("/metrics")
            classes = metrics.get("classes", []) if metrics else []
            if classes:
                actual_label = st.selectbox(
                    "Verified actual class (for production evaluation)",
                    classes,
                    index=None,
                    placeholder="Choose the verified class",
                )
                if st.button("Submit labeled feedback", disabled=actual_label is None):
                    response = requests.post(
                        f"{API}/feedback",
                        files={"file": (f.name, image_bytes)},
                        data={"label": actual_label},
                        headers=admin_headers(),
                        timeout=60,
                    )
                    if response.ok:
                        st.success("Feedback recorded for production evaluation.")
                    else:
                        st.error(response.text)

else:
    st.title("Upload data & retrain")
    s = get("/data/stats")
    if s:
        st.write("Training images:", s["train"]); st.write("Pending uploads:", s["pending_uploads"])
    label = st.text_input("Existing training class label (for example Tomato___healthy)")
    files = st.file_uploader("Bulk upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if st.button("Upload") and files and label:
        r = requests.post(f"{API}/upload", data={"label": label},
                          files=[("files", (x.name, x.getvalue())) for x in files],
                          headers=admin_headers(), timeout=120)
        st.json(r.json() if r.headers.get("content-type", "").startswith("application/json") else {"error": r.text})
    st.divider()
    if st.button("Trigger retraining"):
        r = requests.post(f"{API}/retrain", headers=admin_headers(), timeout=30)
        st.info(r.json().get("message", r.text) if r.ok else r.text)
    st.json(get("/retrain/status"))
    if st.button("Refresh status"):
        st.rerun()
