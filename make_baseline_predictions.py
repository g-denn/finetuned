#!/usr/bin/env python3
"""Create a simple median-return baseline for the held-out test set."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parent
CANONICAL_JSONL = ROOT / "data" / "processed" / "investment_canonical.jsonl"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
PREDICTIONS_PATH = REPORTS_DIR / "majority_baseline_predictions.jsonl"
METRICS_PATH = REPORTS_DIR / "majority_baseline_metrics.json"


def outcome_bucket(multiplier: float | None) -> str | None:
    if multiplier is None or not math.isfinite(multiplier) or multiplier <= 0:
        return None
    if multiplier >= 3.0:
        return "excellent"
    if multiplier >= 1.5:
        return "good"
    if multiplier >= 0.8:
        return "neutral"
    if multiplier >= 0.4:
        return "poor"
    return "failed"


def load_split_ids() -> dict[str, set[str]]:
    split_ids: dict[str, set[str]] = {}
    for split in ("train", "test"):
        ids: set[str] = set()
        path = PROCESSED_DIR / f"investment_{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                idea_id = (payload.get("metadata") or {}).get("idea_id")
                if idea_id:
                    ids.add(idea_id)
        split_ids[split] = ids
    return split_ids


def load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_ids = load_split_ids()
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    with CANONICAL_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            idea_id = row["idea_id"]
            if idea_id in split_ids["train"]:
                train_rows.append(row)
            elif idea_id in split_ids["test"]:
                test_rows.append(row)
    return train_rows, test_rows


def parse_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def score(rows: list[dict[str, Any]], prediction: float) -> dict[str, Any]:
    errors: list[float] = []
    log_errors: list[float] = []
    bucket_correct = 0
    bucket_scored = 0
    predicted_bucket = outcome_bucket(prediction)
    for row in rows:
        truth = parse_float(row.get("directional_perf_3y"))
        truth_bucket = row.get("outcome_3y")
        if truth is None or truth_bucket == "missing":
            continue
        errors.append(abs(prediction - truth))
        log_errors.append(abs(math.log(prediction) - math.log(truth)))
        if predicted_bucket:
            bucket_scored += 1
            bucket_correct += int(predicted_bucket == truth_bucket)
    squared = [value * value for value in errors]
    return {
        "prediction": prediction,
        "predicted_outcome_3y": predicted_bucket,
        "scored": len(errors),
        "mae": sum(errors) / len(errors) if errors else 0.0,
        "rmse": math.sqrt(sum(squared) / len(squared)) if squared else 0.0,
        "mean_abs_log_error": sum(log_errors) / len(log_errors) if log_errors else 0.0,
        "bucket_scored": bucket_scored,
        "bucket_correct": bucket_correct,
        "bucket_accuracy": bucket_correct / bucket_scored if bucket_scored else 0.0,
    }


def main() -> int:
    train_rows, test_rows = load_rows()
    train_values = [
        value
        for row in train_rows
        if (value := parse_float(row.get("directional_perf_3y"))) is not None
    ]
    if not train_values:
        raise SystemExit("No train rows with directional_perf_3y")
    prediction = median(train_values)
    predicted_bucket = outcome_bucket(prediction)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_PATH.open("w", encoding="utf-8") as handle:
        for row in test_rows:
            handle.write(
                json.dumps(
                    {
                        "idea_id": row["idea_id"],
                        "predicted_direction_adjusted_multiplier_3y": prediction,
                        "predicted_outcome_3y": predicted_bucket,
                        "baseline": "train_median_3y_return",
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    metrics = score(test_rows, prediction)
    metrics["test_rows"] = len(test_rows)
    metrics["output"] = str(PREDICTIONS_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
