"""FastAPI service: health/uptime, predict, upload, retrain trigger, stats."""
import asyncio
import hmac
import json, os, threading, time, uuid
from fastapi import Depends, FastAPI, File, Form, Header, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from src.preprocessing import count_images, IMG_EXT
from src.prediction import predict
from src.monitoring import production_metrics, record_feedback
from src import model as M

app = FastAPI(title="Leaf Disease Classifier")
START = time.time()
THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", 50))   # auto-retrain trigger
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
REQUIRE_ADMIN_API_KEY = os.getenv("REQUIRE_ADMIN_API_KEY", "false").lower() == "true"
STATUS_PATH = "models/retrain_status.json"
_lock = threading.Lock()
_predict_lock = asyncio.Lock()

def require_admin_key(x_api_key: str = Header(default="")):
    if not ADMIN_API_KEY:
        if REQUIRE_ADMIN_API_KEY:
            raise HTTPException(503, "Admin API key is required but not configured")
        return
    if not hmac.compare_digest(x_api_key, ADMIN_API_KEY):
        raise HTTPException(401, "A valid X-API-Key is required")

def _set(state, msg=""):
    with open(STATUS_PATH, "w") as status_file:
        json.dump(
            {"state": state, "message": msg, "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
            status_file,
        )

def _status():
    if not os.path.exists(STATUS_PATH):
        return {"state": "idle", "message": ""}
    with open(STATUS_PATH) as status_file:
        return json.load(status_file)

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
    with open(M.METRICS_PATH) as metrics_file:
        return json.load(metrics_file)

@app.get("/metrics/production")
def production_evaluation():
    return production_metrics()

@app.post("/predict")
async def predict_ep(file: UploadFile = File(...)):
    if not os.path.exists(M.MODEL_PATH):
        raise HTTPException(503, "Model not trained yet")
    image = await file.read()
    async with _predict_lock:
        return await run_in_threadpool(predict, image)

@app.post("/feedback", dependencies=[Depends(require_admin_key)])
async def feedback(file: UploadFile = File(...), label: str = Form(...)):
    if not os.path.exists(M.MODEL_PATH):
        raise HTTPException(503, "Model not trained yet")
    with open(M.CLASSES_PATH) as class_file:
        names = json.load(class_file)
    if label not in names:
        raise HTTPException(422, "Label must match one of the model's classes")
    image = await file.read()
    async with _predict_lock:
        result = await run_in_threadpool(predict, image)
    count = await run_in_threadpool(
        record_feedback, label, result["label"], result["probabilities"]
    )
    return {
        "recorded": True,
        "actual_label": label,
        "predicted_label": result["label"],
        "production_feedback_count": count,
    }

@app.post("/upload")
async def upload(
    files: list[UploadFile] = File(...),
    label: str = Form(...),
    _admin=Depends(require_admin_key),
):
    label = "".join(ch for ch in label if ch.isalnum() or ch in "_-")
    if not label:
        raise HTTPException(400, "Invalid label")
    known_labels = {name for name, count in count_images("data/train").items() if count}
    if label not in known_labels:
        raise HTTPException(422, "Label must match an existing training class")
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
def retrain_ep(_admin=Depends(require_admin_key)):
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
