"""Prediction: loads the latest saved model (auto-reloads after retraining)."""
import json, os, threading
import numpy as np
import tensorflow as tf
from src.preprocessing import preprocess_bytes
from src.model import MODEL_PATH, CLASSES_PATH, METRICS_PATH, EARLY_BLIGHT_THRESHOLD, classify_probabilities

_cache = {"mtime": None, "model": None, "names": None, "threshold": EARLY_BLIGHT_THRESHOLD}
_cache_lock = threading.Lock()

def _load():
    with _cache_lock:
        mt = os.stat(MODEL_PATH).st_mtime_ns
        if _cache["mtime"] != mt:
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(CLASSES_PATH, encoding="utf-8") as class_file:
                names = json.load(class_file)
            threshold = EARLY_BLIGHT_THRESHOLD
            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, encoding="utf-8") as metrics_file:
                    metrics = json.load(metrics_file)
                threshold = metrics.get("decision_policy", {}).get(
                    "early_blight_threshold", threshold
                )
            _cache.update(model=model, names=names, threshold=threshold, mtime=mt)
        return _cache["model"], _cache["names"], _cache["threshold"]

def predict(image_bytes: bytes):
    model, names, threshold = _load()
    probs = model.predict(preprocess_bytes(image_bytes), verbose=0)[0]
    i = int(classify_probabilities(probs.reshape(1, -1), names, threshold)[0])
    return {"label": names[i], "confidence": float(probs[i]),
            "probabilities": {n: float(p) for n, p in zip(names, probs)}}
