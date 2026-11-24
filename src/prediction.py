"""Prediction: loads the latest saved model (auto-reloads after retraining)."""
import json, os
import numpy as np
import tensorflow as tf
from src.preprocessing import preprocess_bytes
from src.model import MODEL_PATH, CLASSES_PATH

_cache = {"mtime": None, "model": None, "names": None}

def _load():
    mt = os.path.getmtime(MODEL_PATH)
    if _cache["mtime"] != mt:
        _cache["model"] = tf.keras.models.load_model(MODEL_PATH)
        _cache["names"] = json.load(open(CLASSES_PATH))
        _cache["mtime"] = mt
    return _cache["model"], _cache["names"]

def predict(image_bytes: bytes):
    model, names = _load()
    probs = model.predict(preprocess_bytes(image_bytes), verbose=0)[0]
    i = int(np.argmax(probs))
    return {"label": names[i], "confidence": float(probs[i]),
            "probabilities": {n: float(p) for n, p in zip(names, probs)}}
