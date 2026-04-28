from __future__ import annotations

import json
from pathlib import Path

from data_manager import InventoryManager
from recommender import RecommendationEngine
from feedback import FeedbackLogger

INVENTORY_PATH = "data/raw/inventory.csv"
RULES_PATH = "data/raw/rules.json"
FEEDBACK_PATH = "data/raw/feedback_events.csv"


def _load_color_rules() -> dict:
    rules_file = Path(RULES_PATH)
    if not rules_file.exists():
        return {}
    payload = json.loads(rules_file.read_text(encoding="utf-8"))
    return payload.get("color_matches", {})


def quick_evaluate() -> dict:
    manager = InventoryManager(INVENTORY_PATH)
    rec = RecommendationEngine()
    rec.build_index(manager.data)

    color_rules = _load_color_rules()
    styles = ["Formal", "Casual", "Sport", "Smart Casual"]

    non_empty = 0
    total_scores = []
    confidence_scores = []
    unique_tops = set()

    for style in styles:
        outfits = rec.generate_outfits(
            target_style=style,
            weather_c=18,
            color_rules=color_rules,
            top_k_each=5,
            include_shoes=True,
        )
        if outfits.empty:
            continue

        non_empty += 1
        total_scores.extend(outfits["total_score"].tolist())
        confidence_scores.extend(outfits["confidence"].tolist())
        unique_tops.update(outfits["top"].tolist())

    feedback_summary = FeedbackLogger(FEEDBACK_PATH).summarize_feedback()

    items_total = int(len(manager.data))
    metrics = {
        "inventory_items": items_total,
        "generation_success_rate": round((non_empty / len(styles)) * 100.0, 1),
        "avg_total_score": round(sum(total_scores) / len(total_scores), 2) if total_scores else 0.0,
        "avg_confidence": round(sum(confidence_scores) / len(confidence_scores), 2) if confidence_scores else 0.0,
        "top_diversity_ratio": round(len(unique_tops) / max(1, items_total), 2),
        "feedback": feedback_summary,
    }
    return metrics


if __name__ == "__main__":
    report = quick_evaluate()
    print("Smart Wardrobe quick evaluation")
    for key, value in report.items():
        print(f"- {key}: {value}")
