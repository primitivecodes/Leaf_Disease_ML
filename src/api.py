"""FastAPI service: health/uptime, predict, upload, retrain trigger, stats."""
import asyncio
import hmac
import json, os, threading, time, uuid
from contextlib import contextmanager
from io import BytesIO
from fastapi import Depends, FastAPI, File, Form, Header, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from src.preprocessing import count_images, IMG_EXT
from src.prediction import predict
from src.monitoring import production_metrics, record_feedback
from src import model as M

app = FastAPI(title="Leaf Disease Classifier")
START = time.time()
THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", 50))   # auto-retrain trigger
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", 10 * 1024 * 1024))
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", 100))
MAX_UPLOAD_TOTAL_BYTES = int(os.getenv("MAX_UPLOAD_TOTAL_BYTES", 50 * 1024 * 1024))
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
REQUIRE_ADMIN_API_KEY = os.getenv("REQUIRE_ADMIN_API_KEY", "false").lower() == "true"
STATUS_PATH = "models/retrain_status.json"
RETRAIN_LOCK_PATH = "models/retrain.lock"
_local_retrain_lock = threading.Lock()
_retrain_lock_file = None
_predict_lock = asyncio.Lock()


@contextmanager
def _process_lock(file_obj, blocking=False):
    """Lock a shared file across processes/containers (and support local Windows use)."""
    if os.name == "nt":
        import msvcrt
        file_obj.seek(0)
        if os.fstat(file_obj.fileno()).st_size == 0:
            file_obj.write("0")
            file_obj.flush()
        file_obj.seek(0)
        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        msvcrt.locking(file_obj.fileno(), mode, 1)
        try:
            yield
        finally:
            file_obj.seek(0)
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        mode = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(file_obj.fileno(), mode)
        try:
            yield
        finally:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)


def _acquire_retrain_lock():
    os.makedirs(os.path.dirname(RETRAIN_LOCK_PATH), exist_ok=True)
    file_obj = open(RETRAIN_LOCK_PATH, "a+")
    try:
        lock = _process_lock(file_obj)
        lock.__enter__()
        return file_obj, lock
    except (BlockingIOError, OSError):
        file_obj.close()
        return None, None


def _read_image(file: UploadFile):
    if not file.filename or not file.filename.lower().endswith(IMG_EXT):
        raise HTTPException(415, "Upload a JPG, JPEG, or PNG image")
    data = file.file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"Image exceeds the {MAX_IMAGE_BYTES}-byte limit")
    if not data:
        raise HTTPException(422, "Image is empty")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.width * image.height > 12_000_000:
                raise HTTPException(413, "Image dimensions are too large")
            image.verify()
    except HTTPException:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise HTTPException(422, "File is not a valid image") from exc
    return data

def require_admin_key(x_api_key: str = Header(default="")):
    if not ADMIN_API_KEY:
        if REQUIRE_ADMIN_API_KEY:
            raise HTTPException(503, "Admin API key is required but not configured")
        return
    if not hmac.compare_digest(x_api_key, ADMIN_API_KEY):
        raise HTTPException(401, "A valid X-API-Key is required")

def _set(state, msg=""):
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    temp_path = f"{STATUS_PATH}.{uuid.uuid4().hex}.tmp"
    with open(temp_path, "w") as status_file:
        json.dump(
            {"state": state, "message": msg, "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
            status_file,
        )
    os.replace(temp_path, STATUS_PATH)

def _status():
    if not os.path.exists(STATUS_PATH):
        return {"state": "idle", "message": ""}
    with open(STATUS_PATH) as status_file:
        return json.load(status_file)

def _run_retrain():
    global _retrain_lock_file
    try:
        _set("running", "Retraining started")
        m = M.retrain()
        _set("done", f"accuracy={m['accuracy']:.3f}, merged={m['new_images_merged']} new images")
    except Exception as e:
        _set("failed", str(e))
    finally:
        if _retrain_lock_file is not None:
            _retrain_lock_file[1].__exit__(None, None, None)
            _retrain_lock_file[0].close()
            _retrain_lock_file = None
        _local_retrain_lock.release()

def start_retrain():
    """Start at most one retraining job across processes sharing the models volume."""
    global _retrain_lock_file
    if not _local_retrain_lock.acquire(blocking=False):
        return False
    lock_file, process_lock = _acquire_retrain_lock()
    if lock_file is None:
        _local_retrain_lock.release()
        return False
    _retrain_lock_file = (lock_file, process_lock)
    try:
        _set("running", "Retraining started")
        threading.Thread(target=_run_retrain, daemon=True).start()
    except Exception:
        process_lock.__exit__(None, None, None)
        lock_file.close()
        _retrain_lock_file = None
        _local_retrain_lock.release()
        raise
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
    image = _read_image(file)
    async with _predict_lock:
        try:
            return await run_in_threadpool(predict, image)
        except (ValueError, UnidentifiedImageError) as exc:
            raise HTTPException(422, "Could not decode image") from exc

@app.post("/feedback", dependencies=[Depends(require_admin_key)])
async def feedback(file: UploadFile = File(...), label: str = Form(...)):
    if not os.path.exists(M.MODEL_PATH):
        raise HTTPException(503, "Model not trained yet")
    with open(M.CLASSES_PATH) as class_file:
        names = json.load(class_file)
    if label not in names:
        raise HTTPException(422, "Label must match one of the model's classes")
    image = _read_image(file)
    async with _predict_lock:
        try:
            result = await run_in_threadpool(predict, image)
        except (ValueError, UnidentifiedImageError) as exc:
            raise HTTPException(422, "Could not decode image") from exc
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
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(413, f"At most {MAX_UPLOAD_FILES} images may be uploaded at once")
    if not files:
        raise HTTPException(422, "Select at least one image")
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in label):
        raise HTTPException(400, "Invalid class label")
    known_labels = {name for name, count in count_images("data/train").items() if count}
    if label not in known_labels:
        raise HTTPException(422, "Label must match an existing training class")
    d = os.path.join("data/uploads", label)
    os.makedirs(d, exist_ok=True)
    images = []
    total_bytes = 0
    for uploaded_file in files:
        image = _read_image(uploaded_file)
        total_bytes += len(image)
        if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
            raise HTTPException(413, f"Combined uploads exceed the {MAX_UPLOAD_TOTAL_BYTES}-byte limit")
        images.append((uploaded_file, image))
    saved = 0
    for f, image in images:
        destination = os.path.join(d, f"{uuid.uuid4().hex}_{os.path.basename(f.filename)}")
        with open(destination, "wb") as uploaded:
            uploaded.write(image)
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
