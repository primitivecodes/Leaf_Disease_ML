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
The API moves inference off the event loop, and Docker limits TensorFlow to one intra-op and one inter-op thread per container to avoid CPU oversubscription on shared hosts.

## Load test (Locust)
```bash
# Start with the one-container stack from the Docker section above
locust -f locust/locustfile.py --host http://localhost:8000 -u 100 -r 10 -t 60s --stop-timeout 60 --headless --csv results_1container

docker compose up -d --scale api=2
sleep 10
locust -f locust/locustfile.py --host http://localhost:8000 -u 100 -r 10 -t 60s --stop-timeout 60 --headless --csv results_2containers

docker compose up -d --scale api=3
sleep 10
locust -f locust/locustfile.py --host http://localhost:8000 -u 100 -r 10 -t 60s --stop-timeout 60 --headless --csv results_3containers
```

`--stop-timeout 60` lets in-flight requests finish when the 60-second run ends, instead of counting client-side shutdown resets as failures.
Nginx now rediscovers scaled API containers through Docker DNS, so restarting the load balancer is unnecessary.
For the live charts, run the same Locust command without `--headless` or `--csv`, then open http://localhost:8089.

### Flood simulation results (corrected laptop run)
| Containers | Users | RPS | Median (ms) | 95th % (ms) | Failures |
|---|---|---|---|---|---|
| 1 | 100 | 1.74 | 49,000 | 65,000 | 0 |
| 2 | 100 | 2.18 | 34,000 | 51,000 | 1 |
| 3 | 100 | 3.13 | 25,000 | 40,000 | 42 |

These runs were performed on a laptop, where the containers share CPU resources. Throughput increased with container count, but not linearly. The two-container run had one upstream reset; Locust reported 42 client-side connection resets in the three-container run while nginx/API logs showed successful responses during that interval. The latter cause is unresolved, so treat these as local demonstrations, not a cloud performance benchmark.

## Classification operating point
The API uses a 0.30 probability threshold for early blight, calibrated on the training validation split to improve early-blight recall. On the 283-image held-out test set, this changed accuracy from 84.1% to 85.2%, macro F1 from 0.837 to 0.851, and early-blight recall from 60.8% to 70.3% (precision: 81.3%). The recall gain can increase false positives; the model is a screening aid, not a definitive diagnosis. Recalibrate the threshold after retraining. `/metrics` reports held-out performance using the same decision rule.

## Cloud deployment
Push the image to a registry and run the compose stack on a VM (e.g., AWS EC2 / Azure VM / GCP Compute Engine), or use
Render / Cloud Run for the API and UI containers. Mount a persistent volume for `data/` and `models/`.

## Endpoints
`GET /health` (uptime) · `POST /predict` · `POST /upload` (files + label) · `POST /retrain` · `GET /retrain/status` · `GET /metrics` · `GET /data/stats`
