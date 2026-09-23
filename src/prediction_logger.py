"""
RAEIPHI — Prediction Logger
=============================
Logs every inference request and result to a local SQLite
database for production monitoring. Tracks prediction
confidence, latency, and class distribution over time.

Designed to be imported by inference_server.py:
  from prediction_logger import PredictionLogger
  logger = PredictionLogger()
  logger.log(input_hash, prediction, confidence, latency_ms)

For production, swap SQLite for Supabase by changing the
backend — the interface stays the same.

Database: ml/monitoring/predictions.db
"""

import sqlite3
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

# ── Config ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MONITOR_DIR = os.path.join(BASE_DIR, "monitoring")
DB_PATH = os.path.join(MONITOR_DIR, "predictions.db")

os.makedirs(MONITOR_DIR, exist_ok=True)


class PredictionLogger:
    """
    Logs inference predictions to SQLite for monitoring.

    Each log entry captures:
      - timestamp (UTC)
      - input_hash (SHA256 of the feature vector — no raw data stored)
      - predicted_class (0–10)
      - confidence (model's softmax probability for the predicted class)
      - inference_time_ms
      - top_3_classes (JSON string of top 3 predictions)
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the predictions table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                predicted_class INTEGER NOT NULL,
                confidence REAL NOT NULL,
                inference_time_ms REAL NOT NULL,
                top_3_classes TEXT,
                model_version TEXT DEFAULT '1.0'
            )
        """)

        # Index for time-based queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
            ON predictions(timestamp)
        """)

        # Index for confidence monitoring
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_confidence
            ON predictions(confidence)
        """)

        conn.commit()
        conn.close()

    def log(
        self,
        input_hash: str,
        predicted_class: int,
        confidence: float,
        inference_time_ms: float,
        top_3_classes: Optional[dict] = None,
        model_version: str = "1.0",
    ):
        """Log a single prediction."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO predictions
                (timestamp, input_hash, predicted_class, confidence,
                 inference_time_ms, top_3_classes, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                input_hash,
                predicted_class,
                confidence,
                inference_time_ms,
                json.dumps(top_3_classes) if top_3_classes else None,
                model_version,
            ),
        )
        conn.commit()
        conn.close()

    @staticmethod
    def hash_input(feature_vector) -> str:
        """Generate a SHA256 hash of the input feature vector."""
        data = json.dumps(feature_vector.tolist() if hasattr(feature_vector, 'tolist') else list(feature_vector))
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def get_total_predictions(self) -> int:
        """Total number of logged predictions."""
        conn = sqlite3.connect(self.db_path)
        result = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.close()
        return result

    def get_recent(self, limit: int = 20) -> list:
        """Get the most recent predictions."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]
