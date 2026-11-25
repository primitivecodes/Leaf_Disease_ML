# Tomato Leaf Disease Classifier

An end-to-end image classification project for healthy tomato leaves, early blight, late blight, and leaf mold. It uses MobileNetV2 transfer learning and includes data acquisition, preprocessing, training, offline evaluation, prediction, labeled production feedback, retraining, a Streamlit dashboard, Docker Compose scaling, and Locust load tests.

**GitHub:** [primitivecodes/Leaf_Disease_ML](https://github.com/primitivecodes/Leaf_Disease_ML)
**Video demo:** Not published yet. Add the YouTube URL after recording the demo.
**Live deployment:** Not deployed yet. Follow the EC2 instructions below to create one; the local Docker UI is `http://localhost:8501`.

## Project layout

```text
notebook/Leaf_Disease_ML.ipynb
src/
  data_acquisition.py
  preprocessing.py
  model.py
  prediction.py
  monitoring.py
  api.py
app.py
data/{train,test,uploads}/
models/{leaf_model.h5,class_names.json,metrics.json}
locust/locustfile.py
tests/
Dockerfile
docker-compose.yml
```

## Run locally

From the repository root:

```powershell
git clone https://github.com/primitivecodes/Leaf_Disease_ML.git
Set-Location Leaf_Disease_ML
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.data_acquisition
python -m src.model
```

The acquisition command exports a subset of four PlantVillage tomato classes to `data/train` and `data/test`. Alternatively, provide your own images in `data/<split>/<class>/`. Start the API and UI in separate terminals:

```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

```powershell
$env:API_URL = "http://localhost:8000"
streamlit run app.py
```

The dashboard is at `http://localhost:8501`; FastAPI documentation is at `http://localhost:8000/docs`. The notebook walks through acquisition, preprocessing, visualization, model training, testing, and single-image prediction.

## Prediction, feedback, and retraining

- `POST /predict`: classify one uploaded image.
- `POST /feedback`: classify an image and record its verified class for production evaluation. The actual label must be one of the model's classes.
- `GET /metrics/production`: calculate accuracy, precision, recall, weighted and macro F1, per-class report, and confusion matrix from recorded feedback.
- `POST /upload`: upload one or more images labeled with an existing training class for future training.
- `POST /retrain`: merge pending labeled images into training data and start retraining.
- `GET /retrain/status`, `GET /health`, `GET /metrics`, and `GET /data/stats`: report retraining state, API uptime/model status, held-out test results, and image counts.

The dashboard separates held-out offline test metrics from feedback-based production metrics. Production metrics remain empty until users submit verified labels; predictions alone are not treated as ground truth. Feedback and image uploads are persisted under `data/`.

For local use, write endpoints work without a key. Set `ADMIN_API_KEY` and `REQUIRE_ADMIN_API_KEY=true` before exposing the service; these protect feedback, upload, and retraining endpoints. Never commit the `.env` file.

## Docker Compose and load testing

```powershell
docker compose up --build --scale api=1
```

The API is behind nginx on port `8000`; the UI is on `8501`. Compose mounts `data/` and `models/` so feedback, uploaded images, and trained artifacts persist on the host. Scale the API with `docker compose up -d --scale api=2` (or `3`).

Run Locust from the repository root for each scale, saving separate CSV reports:

```powershell
locust -f locust/locustfile.py --host http://localhost:8000 -u 100 -r 10 -t 60s --stop-timeout 60 --headless --csv results_1container
docker compose up -d --scale api=2
locust -f locust/locustfile.py --host http://localhost:8000 -u 100 -r 10 -t 60s --stop-timeout 60 --headless --csv results_2containers
docker compose up -d --scale api=3
locust -f locust/locustfile.py --host http://localhost:8000 -u 100 -r 10 -t 60s --stop-timeout 60 --headless --csv results_3containers
```

### Recorded laptop load-test results

| API containers | Users | Throughput (requests/s) | Median (ms) | P95 (ms) | Failures |
|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 1.74 | 49,000 | 65,000 | 0 |
| 2 | 100 | 2.18 | 34,000 | 51,000 | 1 |
| 3 | 100 | 3.13 | 25,000 | 40,000 | 42 |

These are local laptop measurements, not cloud benchmarks. Throughput rose with container count but not linearly. The three-container run had 36 failed prediction requests and 6 failed health checks; investigate the connection resets before using these results to claim reliable scaling. The raw Locust reports are `results_{1container,2containers,3containers}_stats.csv` and corresponding history/exception/failure CSVs.

## AWS EC2 deployment

This project uses host-mounted directories for the model, datasets, and feedback database, so a single EC2 VM running Docker Compose is the straightforward cloud demonstration target.

1. Launch an Ubuntu 24.04 EC2 instance with enough memory for TensorFlow training. Attach persistent EBS storage if the dataset and feedback must survive instance replacement.
2. Configure the security group: allow SSH (port 22) only from your IP; allow the UI (8501) only from your IP or trusted demo audience. The UI can submit uploads and retraining requests, so do not expose it publicly without adding UI authentication. If you need direct API access or `/docs`, allow port 8000 only from trusted IPs. Do not expose administrative endpoints without setting the API key.
3. SSH to the instance and install Docker Engine with the Compose plugin, then run:

   ```bash
   git clone https://github.com/primitivecodes/Leaf_Disease_ML.git
   cd Leaf_Disease_ML
   sudo apt-get update
   sudo apt-get install -y docker.io docker-compose-v2
   sudo systemctl enable --now docker
   printf 'ADMIN_API_KEY=%s\nREQUIRE_ADMIN_API_KEY=true\n' "$(openssl rand -hex 32)" > .env
   chmod 600 .env
   sudo docker compose up -d --build --scale api=1
   ```

4. Visit `http://<EC2-PUBLIC-IP>:8501`. Verify the dashboard, predict an image, submit its verified class as feedback, and confirm the production feedback count and metrics. For a public internet deployment, put HTTPS/TLS in front of the UI and restrict administrative access.
5. To compare scaling on the VM, run Locust against `http://<EC2-PUBLIC-IP>:8000` from a trusted load-generator machine, scaling the compose API to 1, 2, and 3 containers and saving a new set of results. Clearly label those results with the VM type and region.

The repository does not currently include a live EC2 URL or cloud-run results: those become available only after an AWS account provisions the instance and the steps above are run. Do not present the laptop Locust results as cloud evaluation.

## Model evaluation

The notebook and `models/metrics.json` report held-out results on 283 images. With the 0.30 early-blight probability threshold, accuracy was 86.6%, weighted F1 was 0.863, and macro F1 was 0.865. Compared with argmax, the threshold increased accuracy from 84.1% to 86.6%, macro F1 from 0.837 to 0.865, and early-blight recall from 62.2% to 78.4%; early-blight precision was 84.1%. This operating point can increase false positives, so treat predictions as a screening aid, not a diagnosis. Retrain and re-evaluate before relying on the model after new data is merged.
