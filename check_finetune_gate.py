#!/usr/bin/env python3
"""Pass/fail gate for the investment fine-tune.

The fine-tune is useful only if it is evaluated on the full held-out test set
and beats the local baselines. A base-model metric file is optional but, when
present, the LoRA must beat that too.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
EXPECTED_TEST_ROWS = 835


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing required metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def accuracy(metrics: dict) -> float:
    value = metrics.get("accuracy")
    if not isinstance(value, (int, float)):
        raise SystemExit("Metrics file is missing numeric accuracy.")
    return float(value)


def scored(metrics: dict) -> int:
    value = metrics.get("scored")
    if not isinstance(value, int):
        raise SystemExit("Metrics file is missing integer scored count.")
    return value


def direction_accuracy(metrics: dict, direction: str) -> float:
    by_direction = metrics.get("by_direction") or {}
    value = by_direction.get(direction, {}).get("accuracy")
    if not isinstance(value, (int, float)):
        raise SystemExit(f"Metrics file is missing {direction} accuracy.")
    return float(value)


def main() -> int:
    majority = load_json(REPORTS / "majority_baseline_metrics.json")
    text = load_json(REPORTS / "text_baseline_metrics.json")["test"]
    finetuned = load_json(REPORTS / "finetuned_test_metrics.json")

    candidates = {
        "majority": accuracy(majority),
        "tfidf_text": accuracy(text),
    }

    base_path = REPORTS / "base_model_test_metrics.json"
    if base_path.exists():
        candidates["base_model"] = accuracy(load_json(base_path))

    best_name, best_accuracy = max(candidates.items(), key=lambda item: item[1])
    fine_accuracy = accuracy(finetuned)
    fine_scored = scored(finetuned)

    result = {
        "expected_test_rows": EXPECTED_TEST_ROWS,
        "finetuned_accuracy": fine_accuracy,
        "finetuned_scored": fine_scored,
        "finetuned_long_accuracy": direction_accuracy(finetuned, "long"),
        "finetuned_short_accuracy": direction_accuracy(finetuned, "short"),
        "best_comparison": best_name,
        "best_comparison_accuracy": best_accuracy,
        "beats_best_comparison": fine_accuracy > best_accuracy,
        "full_test_set": fine_scored == EXPECTED_TEST_ROWS,
        "pass": fine_scored == EXPECTED_TEST_ROWS and fine_accuracy > best_accuracy,
    }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    (REPORTS / "finetune_gate.json").write_text(rendered + "\n", encoding="utf-8")

    if not result["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
