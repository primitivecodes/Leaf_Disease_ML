import tempfile
import unittest
from pathlib import Path

from src.monitoring import production_metrics, record_feedback


class ProductionMonitoringTests(unittest.TestCase):
    def test_metrics_use_recorded_actual_and_predicted_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "feedback.sqlite3"

            self.assertEqual(production_metrics(database)["count"], 0)
            self.assertEqual(record_feedback("healthy", "healthy", {}, database), 1)
            self.assertEqual(record_feedback("blight", "healthy", {}, database), 2)

            metrics = production_metrics(database)

        self.assertEqual(metrics["count"], 2)
        self.assertEqual(metrics["classes"], ["blight", "healthy"])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["confusion_matrix"], [[0, 1], [0, 1]])
        self.assertIsNotNone(metrics["updated_at"])


if __name__ == "__main__":
    unittest.main()
