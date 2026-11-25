"""Persist labeled production feedback and compute online evaluation metrics."""
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

FEEDBACK_DB_PATH = os.getenv("PRODUCTION_FEEDBACK_DB", "data/production_feedback.sqlite3")


def record_feedback(actual_label, predicted_label, probabilities, db_path=FEEDBACK_DB_PATH):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with closing(sqlite3.connect(db_path, timeout=30)) as connection:
        with connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY,
                    actual_label TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    probabilities TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                "INSERT INTO feedback (actual_label, predicted_label, probabilities, recorded_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    actual_label,
                    predicted_label,
                    json.dumps(probabilities),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]


def production_metrics(db_path=FEEDBACK_DB_PATH):
    if not os.path.exists(db_path):
        return {"count": 0, "classes": [], "updated_at": None}

    with closing(sqlite3.connect(db_path, timeout=30)) as connection:
        rows = connection.execute(
            "SELECT actual_label, predicted_label, recorded_at "
            "FROM feedback ORDER BY id"
        ).fetchall()

    if not rows:
        return {"count": 0, "classes": [], "updated_at": None}

    actual = [row[0] for row in rows]
    predicted = [row[1] for row in rows]
    classes = sorted(set(actual) | set(predicted))
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=classes, average="weighted", zero_division=0
    )
    return {
        "count": len(rows),
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(
            precision_recall_fscore_support(
                actual, predicted, labels=classes, average="macro", zero_division=0
            )[2]
        ),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=classes).tolist(),
        "report": classification_report(
            actual,
            predicted,
            labels=classes,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
        "classes": classes,
        "updated_at": rows[-1][2],
    }
