from __future__ import annotations

from datetime import datetime, timezone
import os
import pandas as pd


class FeedbackLogger:
    """Stores simple explicit feedback signals for future personalization."""

    def __init__(self, file_path: str = "data/raw/feedback_events.csv"):
        self.file_path = file_path
        self.columns = [
            "timestamp_utc",
            "top",
            "bottom",
            "outerwear",
            "shoes",
            "accessory",
            "target_style",
            "total_score",
            "feedback",
        ]
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        if os.path.exists(self.file_path):
            frame = pd.read_csv(self.file_path)
            for col in self.columns:
                if col not in frame.columns:
                    frame[col] = ""
            frame = frame[self.columns]
            frame["feedback"] = frame["feedback"].astype(str).str.strip().str.lower()
            frame = frame[frame["feedback"].isin(["like", "dislike"])].reset_index(drop=True)
            return frame
        return pd.DataFrame(columns=self.columns)

    def _persist(self) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.data.to_csv(self.file_path, index=False)

    def log_outfit_feedback(
        self,
        top: str,
        bottom: str,
        outerwear: str,
        shoes: str,
        accessory: str,
        target_style: str,
        total_score: float,
        feedback: str,
    ) -> None:
        feedback_norm = str(feedback or "").strip().lower()
        if feedback_norm not in {"like", "dislike"}:
            raise ValueError("feedback must be either 'like' or 'dislike'")

        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "top": top,
            "bottom": bottom,
            "outerwear": outerwear,
            "shoes": shoes,
            "accessory": accessory,
            "target_style": target_style,
            "total_score": float(total_score),
            "feedback": feedback_norm,
        }
        self.data = pd.concat([self.data, pd.DataFrame([event])], ignore_index=True)
        self._persist()

    def summarize_feedback(self) -> dict:
        if self.data.empty:
            return {
                "events_total": 0,
                "positive": 0,
                "negative": 0,
                "positive_rate": 0.0,
            }

        pos = int((self.data["feedback"] == "like").sum())
        neg = int((self.data["feedback"] == "dislike").sum())
        total = len(self.data)
        return {
            "events_total": int(total),
            "positive": pos,
            "negative": neg,
            "positive_rate": round((pos / total) * 100.0, 1),
        }
