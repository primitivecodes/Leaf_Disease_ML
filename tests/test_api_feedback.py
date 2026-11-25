import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import app
from src.monitoring import production_metrics, record_feedback


class ProductionFeedbackApiTests(unittest.TestCase):
    def test_feedback_requires_key_and_updates_production_metrics(self):
        prediction = {
            "label": "blight",
            "confidence": 0.7,
            "probabilities": {"blight": 0.7, "healthy": 0.3},
        }
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.h5"
            class_names_path = Path(directory) / "classes.json"
            database = Path(directory) / "feedback.sqlite3"
            model_path.touch()
            class_names_path.write_text(json.dumps(["blight", "healthy"]), encoding="utf-8")

            def save_feedback(actual, predicted, probabilities):
                return record_feedback(actual, predicted, probabilities, database)

            with (
                patch("src.api.ADMIN_API_KEY", "test-secret"),
                patch("src.api.M.MODEL_PATH", str(model_path)),
                patch("src.api.M.CLASSES_PATH", str(class_names_path)),
                patch("src.api.predict", return_value=prediction),
                patch("src.api.record_feedback", side_effect=save_feedback),
                patch(
                    "src.api.production_metrics",
                    side_effect=lambda: production_metrics(database),
                ),
            ):
                client = TestClient(app)
                files = {"file": ("leaf.jpg", b"test-image", "image/jpeg")}
                data = {"label": "healthy"}

                denied = client.post("/feedback", files=files, data=data)
                accepted = client.post(
                    "/feedback",
                    files=files,
                    data=data,
                    headers={"X-API-Key": "test-secret"},
                )
                metrics = client.get("/metrics/production")

        self.assertEqual(denied.status_code, 401)
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.json()["predicted_label"], "blight")
        self.assertEqual(accepted.json()["production_feedback_count"], 1)
        self.assertEqual(metrics.json()["count"], 1)
        self.assertEqual(metrics.json()["accuracy"], 0.0)

    def test_upload_rejects_classes_missing_from_training_data(self):
        with patch("src.api.count_images", return_value={"healthy": 12}):
            response = TestClient(app).post(
                "/upload",
                data={"label": "new-disease"},
                files={"files": ("leaf.jpg", b"test-image", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
