"""FastAPI service: health/uptime, predict, upload, retrain trigger, stats."""
import json, os, threading, time, uuid
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from src.preprocessing import count_images, IMG_EXT
from src.prediction import predict
from src import model as M

app = FastAPI(title="Leaf Disease Classifier")
START = time.time()
THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", 50))   # auto-retrain trigger
STATUS_PATH = "models/retrain_status.json"
_lock = threading.Lock()

def _set(state, msg=""):
    json.dump({"state": state, "message": msg, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, open(STATUS_PATH, "w"))

def _status():
    return json.load(open(STATUS_PATH)) if os.path.exists(STATUS_PATH) else {"state": "idle", "message": ""}

def _run_retrain():
    try:
        _set("running", "Retraining started")
        m = M.retrain()
        _set("done", f"accuracy={m['accuracy']:.3f}, merged={m['new_images_merged']} new images")
    except Exception as e:
        _set("failed", str(e))
    finally:
        _lock.release()

def start_retrain():
    """Returns False if a retrain is already running (this container or another sharing the volume)."""
    if not _lock.acquire(blocking=False):
        return False
    st = _status()
    if st["state"] == "running" and time.time() - os.path.getmtime(STATUS_PATH) < 3600:
        _lock.release()
        return False
    threading.Thread(target=_run_retrain, daemon=True).start()
    return True

@app.get("/health")
def health():
    return {"status": "ok", "uptime_seconds": round(time.time() - START, 1),
            "model_loaded": os.path.exists(M.MODEL_PATH)}

@app.get("/metrics")
def metrics():
    if not os.path.exists(M.METRICS_PATH):
        raise HTTPException(404, "No trained model yet. Call /retrain or run src/model.py")
    return json.load(open(M.METRICS_PATH))

@app.post("/predict")
async def predict_ep(file: UploadFile = File(...)):
    if not os.path.exists(M.MODEL_PATH):
        raise HTTPException(503, "Model not trained yet")
    return predict(await file.read())

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...), label: str = Form(...)):
    label = "".join(ch for ch in label if ch.isalnum() or ch in "_-")
    if not label:
        raise HTTPException(400, "Invalid label")
    d = os.path.join("data/uploads", label)
    os.makedirs(d, exist_ok=True)
    saved = 0
    for f in files:
        if f.filename.lower().endswith(IMG_EXT):
            open(os.path.join(d, f"{uuid.uuid4().hex}_{os.path.basename(f.filename)}"), "wb").write(await f.read())
            saved += 1
    pending = sum(count_images("data/uploads").values())
    triggered = pending >= THRESHOLD and start_retrain()
    return {"saved": saved, "pending_total": pending, "threshold": THRESHOLD, "auto_retrain_triggered": triggered}

@app.post("/retrain")
def retrain_ep():
    if not start_retrain():
        raise HTTPException(409, "Retraining already running")
    return {"message": "Retraining started"}

@app.get("/retrain/status")
def retrain_status():
    return _status()

@app.get("/data/stats")
def stats():
    return {"train": count_images("data/train"), "test": count_images("data/test"),
            "pending_uploads": count_images("data/uploads")}
