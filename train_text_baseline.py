#!/usr/bin/env python3
"""Train a local text baseline on memo/catalyst text.

This is not the final LLM fine-tune. It is a cheap signal check:

- train on older ideas
- validate on middle-period ideas
- test on newest held-out ideas
- compare against the majority-class baseline

The model intentionally uses only publication-time inputs plus metadata, not
future performance fields.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CANONICAL_JSONL = ROOT / "data" / "processed" / "investment_canonical.jsonl"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"


def load_split_ids() -> dict[str, set[str]]:
    split_ids: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
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


def input_text(row: dict[str, Any]) -> str:
    direction = "short" if row["is_short"] else "long"
    pieces = [
        f"direction={direction}",
        f"symbol={row.get('eodhd_symbol') or ''}",
        f"sector={row.get('fundamentals_sector') or 'unknown'}",
        f"industry={row.get('fundamentals_industry') or 'unknown'}",
        f"market_cap={row.get('fundamentals_market_cap') or 'unknown'}",
        f"revenue_ttm={row.get('fundamentals_revenue_ttm') or 'unknown'}",
        f"profit_margin={row.get('fundamentals_profit_margin') or 'unknown'}",
        "catalyst:",
        row.get("catalyst") or "",
        "memo:",
        row.get("description") or "",
    ]
    return "\n".join(pieces)


def load_rows() -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    split_ids = load_split_ids()
    with CANONICAL_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            idea_id = row["idea_id"]
            for split, ids in split_ids.items():
                if idea_id in ids:
                    splits[split].append(row)
                    break
    return splits


def score(y_true: list[str], y_pred: list[str], directions: list[str]) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    total = len(y_true)
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred, strict=True))
    by_direction: dict[str, Counter] = defaultdict(Counter)
    confusion: dict[str, Counter] = defaultdict(Counter)
    for truth, pred, direction in zip(y_true, y_pred, directions, strict=True):
        by_direction[direction]["scored"] += 1
        by_direction[direction]["correct"] += int(truth == pred)
        confusion[truth][pred] += 1
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "scored": total,
        "labels": labels,
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


def main() -> int:
    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise SystemExit("Install scikit-learn first: pip install scikit-learn") from exc

    splits = load_rows()
    train_rows = splits["train"]
    val_rows = splits["val"]
    test_rows = splits["test"]

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.9,
                    max_features=120_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=2.0,
                    class_weight="balanced",
                    max_iter=1500,
                    solver="saga",
                    random_state=3407,
                ),
            ),
        ]
    )

    x_train = [input_text(row) for row in train_rows]
    y_train = [row["primary_outcome"] for row in train_rows]
    model.fit(x_train, y_train)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "tfidf_logreg_outcome_baseline.joblib")

    output: dict[str, Any] = {}
    for name, rows in {"val": val_rows, "test": test_rows}.items():
        x = [input_text(row) for row in rows]
        y = [row["primary_outcome"] for row in rows]
        directions = ["short" if row["is_short"] else "long" for row in rows]
        pred = list(model.predict(x))
        output[name] = score(y, pred, directions)
        prediction_path = REPORTS_DIR / f"text_baseline_{name}_predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for row, predicted in zip(rows, pred, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "idea_id": row["idea_id"],
                            "predicted_outcome": predicted,
                            "gold_outcome": row["primary_outcome"],
                            "baseline": "tfidf_logreg",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        output[name]["predictions"] = str(prediction_path)

    metrics_path = REPORTS_DIR / "text_baseline_metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
