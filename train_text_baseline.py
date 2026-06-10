#!/usr/bin/env python3
"""Train a dependency-free text baseline for 3-year returns.

The baseline uses sparse TF-IDF nearest-neighbor regression:

- build TF-IDF vectors for train/test memo text
- find the most similar historical train memos
- predict the weighted average log 3-year direction-adjusted return

It is deliberately simple and reproducible without scikit-learn.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CANONICAL_JSONL = ROOT / "data" / "processed" / "investment_canonical.jsonl"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
TOKEN_RE = re.compile(r"[a-z][a-z0-9]{2,}")
MAX_FEATURES = 8_000
TOP_K = 10
MAX_TEXT_CHARS = 6_000


STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "are", "was", "were",
    "has", "have", "had", "but", "not", "our", "its", "will", "would", "can",
    "company", "business", "market", "stock", "shares", "year", "years",
}


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


def parse_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


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
        f"direction {direction}",
        str(row.get("eodhd_symbol") or ""),
        str(row.get("fundamentals_sector") or "unknown"),
        str(row.get("fundamentals_industry") or "unknown"),
        str(row.get("fundamentals_market_cap") or "unknown"),
        str(row.get("fundamentals_revenue_ttm") or "unknown"),
        str(row.get("fundamentals_profit_margin") or "unknown"),
        row.get("catalyst") or "",
        row.get("description") or "",
    ]
    return "\n".join(pieces)[:MAX_TEXT_CHARS]


def load_rows() -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    split_ids = load_split_ids()
    with CANONICAL_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if parse_float(row.get("directional_perf_3y")) is None:
                continue
            idea_id = row["idea_id"]
            for split, ids in split_ids.items():
                if idea_id in ids:
                    splits[split].append(row)
                    break
    return splits


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def build_vocabulary(train_rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, float]]:
    document_frequency: Counter[str] = Counter()
    for row in train_rows:
        document_frequency.update(set(tokenize(input_text(row))))
    max_df = int(len(train_rows) * 0.9)
    terms = [
        (term, df)
        for term, df in document_frequency.items()
        if 3 <= df <= max_df
    ]
    terms.sort(key=lambda item: (-item[1], item[0]))
    vocab_terms = [term for term, _ in terms[:MAX_FEATURES]]
    vocab = {term: index for index, term in enumerate(vocab_terms)}
    idf = {
        term: math.log((1 + len(train_rows)) / (1 + document_frequency[term])) + 1.0
        for term in vocab_terms
    }
    return vocab, idf


def vectorize(row: dict[str, Any], vocab: dict[str, int], idf: dict[str, float]) -> dict[int, float]:
    counts = Counter(token for token in tokenize(input_text(row)) if token in vocab)
    if not counts:
        return {}
    total = sum(counts.values())
    weights = {
        vocab[token]: (count / total) * idf[token]
        for token, count in counts.items()
    }
    norm = math.sqrt(sum(value * value for value in weights.values()))
    if norm == 0:
        return {}
    return {index: value / norm for index, value in weights.items()}


def build_index(vectors: list[dict[int, float]]) -> dict[int, list[tuple[int, float]]]:
    inverted: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for doc_index, vector in enumerate(vectors):
        for term_index, weight in vector.items():
            inverted[term_index].append((doc_index, weight))
    return inverted


def predict_one(
    vector: dict[int, float],
    inverted: dict[int, list[tuple[int, float]]],
    train_log_returns: list[float],
    fallback_log_return: float,
) -> float:
    scores: defaultdict[int, float] = defaultdict(float)
    for term_index, query_weight in vector.items():
        for doc_index, train_weight in inverted.get(term_index, []):
            scores[doc_index] += query_weight * train_weight
    if not scores:
        return math.exp(fallback_log_return)
    neighbors = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:TOP_K]
    weight_sum = sum(max(score, 0.0) for _, score in neighbors)
    if weight_sum <= 0:
        return math.exp(fallback_log_return)
    predicted_log = sum(train_log_returns[index] * score for index, score in neighbors) / weight_sum
    return max(0.001, math.exp(predicted_log))


def score(rows: list[dict[str, Any]], predictions: list[float]) -> dict[str, Any]:
    errors: list[float] = []
    log_errors: list[float] = []
    bucket_correct = 0
    bucket_scored = 0
    by_direction: dict[str, Counter] = defaultdict(Counter)
    confusion: dict[str, Counter] = defaultdict(Counter)
    for row, predicted in zip(rows, predictions, strict=True):
        truth = parse_float(row.get("directional_perf_3y"))
        if truth is None:
            continue
        truth_bucket = row["outcome_3y"]
        predicted_bucket = outcome_bucket(predicted)
        direction = "short" if row["is_short"] else "long"
        error = abs(predicted - truth)
        log_error = abs(math.log(predicted) - math.log(truth))
        errors.append(error)
        log_errors.append(log_error)
        by_direction[direction]["scored"] += 1
        by_direction[direction]["abs_error_sum"] += error
        by_direction[direction]["squared_error_sum"] += error * error
        by_direction[direction]["abs_log_error_sum"] += log_error
        if predicted_bucket:
            bucket_scored += 1
            bucket_correct += int(predicted_bucket == truth_bucket)
            by_direction[direction]["bucket_scored"] += 1
            by_direction[direction]["bucket_correct"] += int(predicted_bucket == truth_bucket)
            confusion[truth_bucket][predicted_bucket] += 1
    squared = [value * value for value in errors]
    return {
        "mae": sum(errors) / len(errors) if errors else 0.0,
        "rmse": math.sqrt(sum(squared) / len(squared)) if squared else 0.0,
        "mean_abs_log_error": sum(log_errors) / len(log_errors) if log_errors else 0.0,
        "scored": len(errors),
        "bucket_scored": bucket_scored,
        "bucket_correct": bucket_correct,
        "bucket_accuracy": bucket_correct / bucket_scored if bucket_scored else 0.0,
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


def main() -> int:
    splits = load_rows()
    train_rows = splits["train"]
    val_rows = splits["val"]
    test_rows = splits["test"]
    vocab, idf = build_vocabulary(train_rows)
    train_vectors = [vectorize(row, vocab, idf) for row in train_rows]
    inverted = build_index(train_vectors)
    train_log_returns = [math.log(parse_float(row["directional_perf_3y"]) or 1.0) for row in train_rows]
    fallback_log_return = sorted(train_log_returns)[len(train_log_returns) // 2]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "tfidf_knn_3y_return_baseline.json").write_text(
        json.dumps(
            {
                "model": "tfidf_knn_log_3y_return",
                "max_features": MAX_FEATURES,
                "top_k": TOP_K,
                "vocab_size": len(vocab),
                "train_rows": len(train_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    output: dict[str, Any] = {}
    for name, rows in {"val": val_rows, "test": test_rows}.items():
        vectors = [vectorize(row, vocab, idf) for row in rows]
        predictions = [predict_one(vector, inverted, train_log_returns, fallback_log_return) for vector in vectors]
        output[name] = score(rows, predictions)
        prediction_path = REPORTS_DIR / f"text_baseline_{name}_predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for row, predicted in zip(rows, predictions, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "idea_id": row["idea_id"],
                            "predicted_direction_adjusted_multiplier_3y": predicted,
                            "predicted_outcome_3y": outcome_bucket(predicted),
                            "gold_direction_adjusted_multiplier_3y": parse_float(row["directional_perf_3y"]),
                            "gold_outcome_3y": row["outcome_3y"],
                            "baseline": "tfidf_knn_log_3y_return",
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
