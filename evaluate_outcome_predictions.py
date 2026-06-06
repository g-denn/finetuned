#!/usr/bin/env python3
"""Evaluate model-predicted outcome labels against the held-out test set.

Expected prediction JSONL formats:

1. {"idea_id": "...", "predicted_outcome": "good"}
2. {"metadata": {"idea_id": "..."}, "assistant": "Primary training label: 3y outcome is good."}
3. {"metadata": {"idea_id": "..."}, "messages": [..., {"role": "assistant", "content": "..."}]}

This intentionally scores only held-out labels. It is not a backtest by itself,
but it is the minimum gate before trusting a fine-tuned model.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CANONICAL_JSONL = ROOT / "data" / "processed" / "investment_canonical.jsonl"
VALID_OUTCOMES = {"excellent", "good", "neutral", "poor", "failed"}
OUTCOME_PATTERN = re.compile(r"primary training label:\s*\S+\s+outcome\s+is\s+(\w+)", re.I)


def load_test_labels() -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    with CANONICAL_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            date = row.get("publication_date") or ""
            if "2021-08-07" <= date <= "2022-11-03":
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


def extract_prediction(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    idea_id = payload.get("idea_id")
    if not idea_id and isinstance(payload.get("metadata"), dict):
        idea_id = payload["metadata"].get("idea_id")

    predicted = payload.get("predicted_outcome")
    if isinstance(predicted, str):
        predicted = predicted.strip().lower()
    else:
        text = extract_text(payload)
        match = OUTCOME_PATTERN.search(text)
        predicted = match.group(1).strip().lower() if match else None

    if predicted not in VALID_OUTCOMES:
        predicted = None
    return idea_id, predicted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    labels = load_test_labels()
    totals = Counter()
    by_direction: dict[str, Counter] = defaultdict(Counter)
    confusion: dict[str, Counter] = defaultdict(Counter)
    missing = 0
    unknown_id = 0

    with args.predictions_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            idea_id, predicted = extract_prediction(payload)
            if not idea_id or idea_id not in labels:
                unknown_id += 1
                continue
            if predicted is None:
                missing += 1
                continue
            row = labels[idea_id]
            truth = row["primary_outcome"]
            direction = "short" if row["is_short"] else "long"
            totals["scored"] += 1
            totals["correct"] += int(predicted == truth)
            by_direction[direction]["scored"] += 1
            by_direction[direction]["correct"] += int(predicted == truth)
            confusion[truth][predicted] += 1

    result = {
        "scored": totals["scored"],
        "correct": totals["correct"],
        "accuracy": totals["correct"] / totals["scored"] if totals["scored"] else 0.0,
        "missing_prediction": missing,
        "unknown_id": unknown_id,
        "by_direction": {
            key: {
                "scored": value["scored"],
                "correct": value["correct"],
                "accuracy": value["correct"] / value["scored"] if value["scored"] else 0.0,
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
