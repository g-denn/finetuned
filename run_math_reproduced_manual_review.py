#!/usr/bin/env python3
"""Run resumable manual review for math-reproduced performance rows.

This wrapper keeps the expensive Agent A/B/C verifier resumable:

- input rows come from validation_results.csv
- only math_validation_status == math_reproduced is reviewed by default
- each review is appended to JSONL immediately
- pass-only rows are exported to a promotion CSV

No API keys are required. Yahoo cross-checks use the public chart endpoint from
manual_review_validation.py unless --no-yahoo is passed.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from manual_review_validation import (
    DEFAULT_CACHE_DIR,
    DEFAULT_FUNDAMENTALS_CACHE_DIR,
    DEFAULT_QUALITATIVE_EVIDENCE,
    load_qualitative_evidence,
    load_result_rows,
    review_row,
    risk_priority,
)


DEFAULT_RESULTS_CSV = Path("eodhd_output/full_run/validation_results.csv")
DEFAULT_REVIEWS_JSONL = Path("eodhd_output/full_run/math_reproduced_manual_reviews.jsonl")
DEFAULT_PROMOTION_CSV = Path("eodhd_output/full_run/math_reproduced_training_ready.csv")


def review_key(row: dict[str, str]) -> str:
    return "|".join(
        [
            row.get("idea_id") or "",
            row.get("eodhd_symbol") or "",
            row.get("publication_date") or "",
        ]
    )


def load_reviewed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    reviewed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = "|".join(
                [
                    str(payload.get("idea_id") or ""),
                    str(payload.get("eodhd_symbol") or ""),
                    str(payload.get("publication_date") or ""),
                ]
            )
            if key.strip("|"):
                reviewed.add(key)
    return reviewed


def select_math_reproduced_rows(
    rows: list[dict[str, str]],
    include_low_risk: bool,
    include_provider_warnings: bool,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        if row.get("math_validation_status") != "math_reproduced":
            continue
        stage = row.get("review_stage") or ""
        if stage == "math_reproduced_low_risk" and not include_low_risk:
            continue
        if stage == "provider_warning" and not include_provider_warnings:
            continue
        selected.append(row)
    return sorted(selected, key=risk_priority)


def best_verified_returns(review: dict[str, Any]) -> dict[str, Any]:
    returns: dict[str, Any] = {}
    for horizon, result in (review.get("horizon_reviews") or {}).items():
        if result.get("verdict") != "pass":
            continue
        eodhd = result.get("eodhd") or {}
        returns[horizon] = eodhd.get("multiplier")
    return returns


def iter_reviews(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    reviews: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                reviews.append(payload)
    return reviews


def write_promotion_csv(reviews_path: Path, output_path: Path) -> int:
    fieldnames = [
        "idea_id",
        "raw_symbol",
        "eodhd_symbol",
        "publication_date",
        "review_status",
        "verified_perf_1y",
        "verified_perf_3y",
        "verified_perf_5y",
        "verified_perf_10y",
        "verified_perf_20y",
        "agent_c_status",
        "agent_c_reason",
        "agent_c_outcome_type",
        "source_count",
        "promotion_decision",
    ]
    passed = [review for review in iter_reviews(reviews_path) if review.get("review_status") == "pass"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for review in passed:
            returns = best_verified_returns(review)
            agent_c = review.get("agent_c_qualitative") or {}
            writer.writerow(
                {
                    "idea_id": review.get("idea_id"),
                    "raw_symbol": review.get("raw_symbol"),
                    "eodhd_symbol": review.get("eodhd_symbol"),
                    "publication_date": review.get("publication_date"),
                    "review_status": review.get("review_status"),
                    "verified_perf_1y": returns.get("1y"),
                    "verified_perf_3y": returns.get("3y"),
                    "verified_perf_5y": returns.get("5y"),
                    "verified_perf_10y": returns.get("10y"),
                    "verified_perf_20y": returns.get("20y"),
                    "agent_c_status": agent_c.get("reviewer_status"),
                    "agent_c_reason": agent_c.get("reason"),
                    "agent_c_outcome_type": agent_c.get("outcome_type"),
                    "source_count": len(agent_c.get("sources") or []),
                    "promotion_decision": "training_ready_candidate",
                }
            )
    return len(passed)


def count_review_statuses(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for review in iter_reviews(path):
        status = str(review.get("review_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--fundamentals-cache-dir", default=str(DEFAULT_FUNDAMENTALS_CACHE_DIR))
    parser.add_argument("--qualitative-evidence", default=str(DEFAULT_QUALITATIVE_EVIDENCE))
    parser.add_argument("--reviews-jsonl", default=str(DEFAULT_REVIEWS_JSONL))
    parser.add_argument("--promotion-csv", default=str(DEFAULT_PROMOTION_CSV))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-provider-warnings", action="store_true", default=True)
    parser.add_argument("--exclude-provider-warnings", action="store_false", dest="include_provider_warnings")
    parser.add_argument("--include-low-risk", action="store_true", default=True)
    parser.add_argument("--exclude-low-risk", action="store_false", dest="include_low_risk")
    parser.add_argument("--no-yahoo", action="store_true")
    parser.add_argument("--rebuild-promotion-only", action="store_true")
    args = parser.parse_args()

    reviews_path = Path(args.reviews_jsonl)
    promotion_path = Path(args.promotion_csv)
    if args.rebuild_promotion_only:
        passed = write_promotion_csv(reviews_path, promotion_path)
        print(
            json.dumps(
                {
                    "reviewed_counts": count_review_statuses(reviews_path),
                    "promotion_rows": passed,
                    "promotion_csv": str(promotion_path.resolve()),
                },
                indent=2,
            )
        )
        return 0

    rows = select_math_reproduced_rows(
        load_result_rows(Path(args.results_csv)),
        include_low_risk=args.include_low_risk,
        include_provider_warnings=args.include_provider_warnings,
    )
    reviewed_keys = load_reviewed_keys(reviews_path)
    pending = [row for row in rows if review_key(row) not in reviewed_keys]
    if args.limit:
        pending = pending[: args.limit]

    qualitative_evidence = load_qualitative_evidence(Path(args.qualitative_evidence))
    reviews_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with reviews_path.open("a", encoding="utf-8") as handle:
        for row in pending:
            try:
                review = review_row(
                    row,
                    cache_dir=Path(args.cache_dir),
                    fetch_yahoo=not args.no_yahoo,
                    qualitative_evidence=qualitative_evidence,
                    fundamentals_cache_dir=Path(args.fundamentals_cache_dir),
                )
            except Exception as exc:  # noqa: BLE001 - one row must not stop the batch.
                review = {
                    "idea_id": row.get("idea_id"),
                    "raw_symbol": row.get("raw_symbol"),
                    "eodhd_symbol": row.get("eodhd_symbol"),
                    "publication_date": row.get("publication_date"),
                    "review_status": "manual_review",
                    "row_failures": [],
                    "row_warnings": [f"review_runner_exception:{type(exc).__name__}:{exc}"],
                    "reviewed_at": datetime.now(UTC).isoformat(),
                }
            review["reviewed_at"] = datetime.now(UTC).isoformat()
            handle.write(json.dumps(review, sort_keys=True) + "\n")
            handle.flush()
            completed += 1
            if completed % 25 == 0:
                print(f"reviewed_batch_rows={completed}/{len(pending)}")

    passed = write_promotion_csv(reviews_path, promotion_path)
    print(
        json.dumps(
            {
                "selected_math_reproduced_rows": len(rows),
                "already_reviewed": len(reviewed_keys),
                "reviewed_this_run": completed,
                "reviewed_counts": count_review_statuses(reviews_path),
                "promotion_rows": passed,
                "reviews_jsonl": str(reviews_path.resolve()),
                "promotion_csv": str(promotion_path.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
