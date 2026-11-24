# Tomato Leaf Disease Classifier – End-to-End ML Pipeline

**Video demo:** <YOUTUBE_LINK>
**Live UI:** <UI_URL> | **API docs:** <API_URL>/docs
**GitHub:** <REPO_LINK>

## Description
Image classification of tomato leaves (healthy, early blight, late blight, leaf mold) using MobileNetV2 transfer learning.
Includes data acquisition, preprocessing, training, evaluation, an API (FastAPI), a UI (Streamlit), bulk upload with
manual **and automatic** retraining (triggered when pending uploads reach `RETRAIN_THRESHOLD`, default 50),
Docker deployment behind an nginx load balancer, and Locust load testing.

## Structure
```
README.md | notebook/Leaf_Disease_ML.ipynb | src/{data_acquisition,preprocessing,model,prediction,api}.py
app.py (UI) | locust/locustfile.py | Dockerfile | docker-compose.yml | data/{train,test,uploads} | models/leaf_model.h5
```

## Setup
```bash
git clone <REPO_LINK> && cd Leaf_Disease_ML
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m src.data_acquisition        # download data (or put your own images in data/train/<class>/ and data/test/<class>/)
python -m src.model                   # train -> models/leaf_model.h5
uvicorn src.api:app --port 8000       # API
API_URL=http://localhost:8000 streamlit run app.py   # UI at :8501
```

## Docker + scaling
```bash
docker compose up --build --scale api=1     # then 2, 3 ... for the comparison
```
API via load balancer: http://localhost:8000 — UI: http://localhost:8501

## Load test (Locust)
```bash
locust -f locust/locustfile.py --host http://localhost:8000 -u 100 -r 10 -t 60s --headless --csv results_1container
```
Repeat after `docker compose up --scale api=2`, `=3`, renaming the CSV each time.

### Flood simulation results (fill in from your CSVs)
| Containers | Users | RPS | Median (ms) | 95th % (ms) | Failures |
|---|---|---|---|---|---|
| 1 | 100 | | | | |
| 2 | 100 | | | | |
| 3 | 100 | | | | |

## Cloud deployment
Push the image to a registry and run the compose stack on a VM (e.g., AWS EC2 / Azure VM / GCP Compute Engine), or use
Render / Cloud Run for the API and UI containers. Mount a persistent volume for `data/` and `models/`.

## Endpoints
`GET /health` (uptime) · `POST /predict` · `POST /upload` (files + label) · `POST /retrain` · `GET /retrain/status` · `GET /metrics` · `GET /data/stats`
