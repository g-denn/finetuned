#!/usr/bin/env python3
"""Evaluate model-predicted 3-year returns against the held-out test set.

Expected prediction JSONL formats:

1. {"idea_id": "...", "predicted_direction_adjusted_multiplier_3y": 1.7}
2. {"idea_id": "...", "assistant": "{\"direction_adjusted_multiplier_3y\": 1.7}"}
3. {"metadata": {"idea_id": "..."}, "messages": [..., {"role": "assistant", "content": "..."}]}

The primary score is numeric closeness to the actual 3-year direction-adjusted
return. Bucket accuracy is derived from the numeric prediction and kept as a
secondary metric for continuity with the older label-only gate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CANONICAL_JSONL = ROOT / "data" / "processed" / "investment_canonical.jsonl"
PROCESSED_DIR = ROOT / "data" / "processed"
VALID_OUTCOMES = {"excellent", "good", "neutral", "poor", "failed"}
MULTIPLIER_PATTERN = re.compile(
    r"(?:direction[_ -]?adjusted[_ -]?multiplier(?:_3y)?|directional[_ -]?perf(?:_3y)?)"
    r"[^0-9.+-]*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
    re.I,
)
LABEL_PATTERN = re.compile(r"\b(excellent|good|neutral|poor|failed)\b", re.I)


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


def load_test_ids() -> set[str]:
    ids: set[str] = set()
    path = PROCESSED_DIR / "investment_test.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            idea_id = (payload.get("metadata") or {}).get("idea_id")
            if idea_id:
                ids.add(idea_id)
    return ids


def load_test_labels() -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    test_ids = load_test_ids()
    with CANONICAL_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["idea_id"] in test_ids:
                labels[row["idea_id"]] = row
    return labels


def extract_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("assistant"), str):
        return payload["assistant"]
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content") or "")
    return ""


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def extract_prediction(payload: dict[str, Any]) -> tuple[str | None, float | None, str | None]:
    idea_id = payload.get("idea_id")
    if not idea_id and isinstance(payload.get("metadata"), dict):
        idea_id = payload["metadata"].get("idea_id")

    predicted = parse_float(
        payload.get("predicted_direction_adjusted_multiplier_3y")
        or payload.get("predicted_directional_perf_3y")
        or payload.get("direction_adjusted_multiplier_3y")
        or payload.get("directional_perf_3y")
    )
    predicted_outcome = payload.get("predicted_outcome_3y") or payload.get("predicted_outcome")

    text = extract_text(payload)
    if predicted is None and text:
        parsed = parse_json_object(text)
        if parsed:
            predicted = parse_float(
                parsed.get("direction_adjusted_multiplier_3y")
                or parsed.get("directional_perf_3y")
                or parsed.get("predicted_direction_adjusted_multiplier_3y")
            )
            predicted_outcome = predicted_outcome or parsed.get("outcome_3y")
        if predicted is None:
            match = MULTIPLIER_PATTERN.search(text)
            predicted = parse_float(match.group(1)) if match else None
        if not predicted_outcome:
            match = LABEL_PATTERN.search(text)
            predicted_outcome = match.group(1).lower() if match else None

    if isinstance(predicted_outcome, str):
        predicted_outcome = predicted_outcome.strip().lower()
    if predicted_outcome not in VALID_OUTCOMES:
        predicted_outcome = None
    return idea_id, predicted, predicted_outcome


def summarize_errors(errors: list[float], log_errors: list[float]) -> dict[str, float]:
    squared = [value * value for value in errors]
    return {
        "mae": sum(errors) / len(errors) if errors else 0.0,
        "rmse": math.sqrt(sum(squared) / len(squared)) if squared else 0.0,
        "mean_abs_log_error": sum(log_errors) / len(log_errors) if log_errors else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    labels = load_test_labels()
    totals = Counter()
    by_direction: dict[str, Counter] = defaultdict(Counter)
    confusion: dict[str, Counter] = defaultdict(Counter)
    missing_prediction = 0
    missing_gold_3y = 0
    unknown_id = 0
    errors: list[float] = []
    log_errors: list[float] = []
    outcome_matches = 0
    provided_outcome_matches = 0
    provided_outcome_scored = 0

    with args.predictions_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            idea_id, predicted, provided_outcome = extract_prediction(payload)
            if not idea_id or idea_id not in labels:
                unknown_id += 1
                continue
            row = labels[idea_id]
            truth = parse_float(row.get("directional_perf_3y"))
            truth_bucket = row.get("outcome_3y")
            if truth is None or truth_bucket == "missing":
                missing_gold_3y += 1
                continue
            if predicted is None:
                missing_prediction += 1
                continue

            derived_bucket = outcome_bucket(predicted)
            direction = "short" if row["is_short"] else "long"
            error = abs(predicted - truth)
            log_error = abs(math.log(predicted) - math.log(truth))
            totals["scored"] += 1
            by_direction[direction]["scored"] += 1
            by_direction[direction]["abs_error_sum"] += error
            by_direction[direction]["squared_error_sum"] += error * error
            by_direction[direction]["abs_log_error_sum"] += log_error
            errors.append(error)
            log_errors.append(log_error)

            if derived_bucket:
                totals["bucket_scored"] += 1
                by_direction[direction]["bucket_scored"] += 1
                matched = int(derived_bucket == truth_bucket)
                outcome_matches += matched
                by_direction[direction]["bucket_correct"] += matched
                confusion[truth_bucket][derived_bucket] += 1

            if provided_outcome:
                provided_outcome_scored += 1
                provided_outcome_matches += int(provided_outcome == truth_bucket)

    result = {
        "scored": totals["scored"],
        **summarize_errors(errors, log_errors),
        "bucket_scored": totals["bucket_scored"],
        "bucket_correct": outcome_matches,
        "bucket_accuracy": outcome_matches / totals["bucket_scored"] if totals["bucket_scored"] else 0.0,
        "provided_outcome_scored": provided_outcome_scored,
        "provided_outcome_accuracy": (
            provided_outcome_matches / provided_outcome_scored if provided_outcome_scored else 0.0
        ),
        "missing_prediction": missing_prediction,
        "missing_gold_3y": missing_gold_3y,
        "unknown_id": unknown_id,
        "by_direction": {
            key: {
                "scored": value["scored"],
                "mae": value["abs_error_sum"] / value["scored"] if value["scored"] else 0.0,
                "rmse": math.sqrt(value["squared_error_sum"] / value["scored"]) if value["scored"] else 0.0,
                "mean_abs_log_error": (
                    value["abs_log_error_sum"] / value["scored"] if value["scored"] else 0.0
                ),
                "bucket_scored": value["bucket_scored"],
                "bucket_correct": value["bucket_correct"],
                "bucket_accuracy": (
                    value["bucket_correct"] / value["bucket_scored"] if value["bucket_scored"] else 0.0
                ),
            }
            for key, value in sorted(by_direction.items())
        },
        "confusion": {truth: dict(preds) for truth, preds in sorted(confusion.items())},
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
