#!/usr/bin/env python3
"""Salvage manual-review rows with strict EODHD-internal evidence.

This pass handles rows that were held only because the target-horizon Yahoo
cross-check was unavailable. It does not touch rows with provider conflicts,
extreme winners, severe losers, reverse splits, non-common instruments, missing
fundamental financials, or unresolved lineage risk.

Promotion requires:

- common-stock fundamentals with financial statements
- delisted/early-ended history evidence from EODHD
- cached EODHD adjusted return recomputation for the target horizon
- endpoint before delisting/final-trading risk
- no split in the validated window
- stable adjusted/raw ratio and no large single-day adjusted-price jumps

No API keys are read or stored.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from salvage_partial_horizon_labels import (
    BLOCKING_FLAGS,
    DO_NOT_AUTO_PROMOTE,
    cached_prices,
    internal_adjustment_sanity,
    load_symbol_cache,
    looks_non_common_instrument,
    parse_bool,
    parse_float,
    recompute_horizon,
    scalar,
    split_flags,
)


BASE_DIR = Path("eodhd_output/full_run")
VALIDATION_IN = BASE_DIR / "validation_results_with_partial_horizon_salvage.csv"
TRAINING_READY_IN = BASE_DIR / "training_ready_after_partial_horizon_salvage.csv"

ROW_REVIEWS_CSV = BASE_DIR / "manual_internal_salvage_reviews.csv"
ROW_REVIEWS_JSONL = BASE_DIR / "manual_internal_salvage_reviews.jsonl"
VALIDATION_OUT = BASE_DIR / "validation_results_with_internal_manual_salvage.csv"
TRAINING_READY_OUT = BASE_DIR / "training_ready_after_internal_manual_salvage.csv"
SUMMARY_JSON = BASE_DIR / "manual_internal_salvage_summary.json"

MIN_RETURN = 0.05
MAX_RETURN = 10.0


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("idea_id") or "", row.get("eodhd_symbol") or "", row.get("publication_date") or "")


def base_hold_reasons(row: dict[str, str], multiplier: float | None) -> list[str]:
    flags = split_flags(row)
    reasons: list[str] = []
    if row.get("fundamentals_type") != "Common Stock":
        reasons.append("not_common_stock")
    if row.get("fundamentals_status") != "fetched":
        reasons.append("missing_fundamental_identity")
    if looks_non_common_instrument(row):
        reasons.append("instrument_name_or_symbol_not_common_stock")
    for flag in sorted(flags & BLOCKING_FLAGS):
        reasons.append(flag)
    if multiplier is None:
        reasons.append("missing_target_multiplier")
    elif multiplier >= MAX_RETURN:
        reasons.append("large_winner_requires_business_quality_review")
    elif multiplier <= MIN_RETURN:
        reasons.append("severe_loser_requires_bankruptcy_or_delisting_outcome_model")
    symbol = row.get("eodhd_symbol") or ""
    if symbol in DO_NOT_AUTO_PROMOTE:
        reasons.append("do_not_auto_promote_symbol")
    if not (
        parse_bool(row.get("is_in_delisted_cache"))
        or parse_bool(row.get("fundamentals_is_delisted"))
        or "price_history_ends_before_long_horizon" in flags
        or "symbol_in_delisted_cache" in flags
    ):
        reasons.append("not_delisted_or_early_ended_history")
    return reasons


def review_row(row: dict[str, str]) -> dict[str, Any]:
    horizon = row.get("review_target_horizon") or ""
    multiplier = parse_float(row.get("review_target_multiplier"))
    base = {
        "idea_id": row.get("idea_id"),
        "raw_symbol": row.get("raw_symbol"),
        "eodhd_symbol": row.get("eodhd_symbol"),
        "publication_date": row.get("publication_date"),
        "company_name": row.get("fundamentals_name"),
        "target_horizon": horizon,
        "target_multiplier": multiplier,
        "fundamentals_type": row.get("fundamentals_type"),
        "fundamentals_is_delisted": row.get("fundamentals_is_delisted"),
        "fundamentals_delisted_date": row.get("fundamentals_delisted_date"),
        "is_in_delisted_cache": row.get("is_in_delisted_cache"),
        "warning_modes": row.get("warning_modes") or "",
        "failure_modes": row.get("failure_modes") or "",
    }
    reasons = base_hold_reasons(row, multiplier)
    if reasons:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": ";".join(reasons),
            "confidence": 0.45,
            "passed_horizons": {},
        }
    if not horizon:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": "missing_target_horizon",
            "confidence": 0.45,
            "passed_horizons": {},
        }

    payload = load_symbol_cache(row.get("eodhd_symbol") or "")
    prices = cached_prices(payload)
    eodhd_review = recompute_horizon(row, prices, horizon)
    if eodhd_review.get("status") != "pass":
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": str(eodhd_review.get("reason") or "target_horizon_recompute_failed"),
            "eodhd_review": eodhd_review,
            "confidence": 0.5,
            "passed_horizons": {},
        }

    sanity_ok, sanity_reason, sanity_diagnostics = internal_adjustment_sanity(row, payload, {horizon: eodhd_review})
    if not sanity_ok:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": sanity_reason,
            "eodhd_review": eodhd_review,
            "internal_sanity": sanity_diagnostics,
            "confidence": 0.55,
            "passed_horizons": {},
        }

    return {
        **base,
        "review_status": "pass",
        "training_action": "add_to_training_ready",
        "reason": "manual_target_horizon_eodhd_delisted_fundamentals_internal_sanity_passed",
        "eodhd_review": eodhd_review,
        "internal_sanity": sanity_diagnostics,
        "confidence": 0.64,
        "passed_horizons": {horizon: eodhd_review["multiplier"]},
    }


def select_candidates(rows: list[dict[str, str]], training_keys: set[tuple[str, str, str]]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for row in rows:
        if row_key(row) in training_keys:
            continue
        if row.get("remaining_manual_training_action") != "hold":
            continue
        if row.get("remaining_manual_reason") != "missing_yahoo_target_horizon_cross_check":
            continue
        candidates.append(row)
    return candidates


def write_reviews(reviews: list[dict[str, Any]]) -> None:
    fieldnames = [
        "idea_id",
        "raw_symbol",
        "eodhd_symbol",
        "publication_date",
        "company_name",
        "target_horizon",
        "target_multiplier",
        "review_status",
        "training_action",
        "reason",
        "passed_horizons",
        "confidence",
        "fundamentals_type",
        "fundamentals_is_delisted",
        "fundamentals_delisted_date",
        "is_in_delisted_cache",
        "warning_modes",
        "failure_modes",
    ]
    with ROW_REVIEWS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for review in reviews:
            writer.writerow({field: scalar(review.get(field)) for field in fieldnames})
    with ROW_REVIEWS_JSONL.open("w", encoding="utf-8") as handle:
        for review in reviews:
            handle.write(json.dumps(review, sort_keys=True) + "\n")


def write_validation(rows: list[dict[str, str]], reviews_by_key: dict[tuple[str, str, str], dict[str, Any]]) -> None:
    extra = [
        "manual_internal_review_status",
        "manual_internal_training_action",
        "manual_internal_reason",
        "manual_internal_passed_horizons",
        "manual_internal_confidence",
    ]
    fieldnames = list(rows[0].keys()) + [field for field in extra if field not in rows[0]]
    with VALIDATION_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            review = reviews_by_key.get(row_key(row))
            if review:
                output.update(
                    {
                        "manual_internal_review_status": review.get("review_status"),
                        "manual_internal_training_action": review.get("training_action"),
                        "manual_internal_reason": review.get("reason"),
                        "manual_internal_passed_horizons": scalar(review.get("passed_horizons")),
                        "manual_internal_confidence": scalar(review.get("confidence")),
                    }
                )
            writer.writerow(output)


def write_training(existing_training: list[dict[str, str]], pass_reviews: list[dict[str, Any]]) -> int:
    fieldnames = list(existing_training[0].keys())
    now = datetime.now(UTC).isoformat()
    additions: list[dict[str, str]] = []
    for review in pass_reviews:
        output = {field: "" for field in fieldnames}
        output.update(
            {
                "idea_id": scalar(review.get("idea_id")),
                "raw_symbol": scalar(review.get("raw_symbol")),
                "eodhd_symbol": scalar(review.get("eodhd_symbol")),
                "publication_date": scalar(review.get("publication_date")),
                "include_in_training": "true",
                "math_validation_status": "manual_target_horizon_verified_eodhd_internal",
                "review_stage": "eodhd_internal_manual_target_horizon",
                "training_readiness": "training_ready",
                "review_status": "pass",
                "reviewed_at": now,
                "agent_b_yahoo_symbol": "",
                "agent_b_yahoo_rows": "0",
                "agent_c_status": "pass",
                "agent_c_reason": scalar(review.get("reason")),
                "agent_c_outcome_type": "eodhd_internal_delisted_manual_target_horizon",
                "source_count": "5",
                "fundamentals_name": scalar(review.get("company_name")),
                "fundamentals_type": scalar(review.get("fundamentals_type")),
                "original_validation_status": "manual_review_required",
                "original_review_stage": "manual_target_horizon_eodhd_internal_salvage",
                "original_warning_modes": scalar(review.get("warning_modes")),
                "original_failure_modes": scalar(review.get("failure_modes")),
            }
        )
        for horizon, value in (review.get("passed_horizons") or {}).items():
            field = f"validated_perf_{horizon}"
            if field in output:
                output[field] = scalar(value)
        additions.append(output)

    with TRAINING_READY_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_training)
        writer.writerows(additions)
    return len(additions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = load_csv(VALIDATION_IN)
    existing_training = load_csv(TRAINING_READY_IN)
    training_keys = {row_key(row) for row in existing_training}
    candidates = select_candidates(rows, training_keys)
    if args.limit:
        candidates = candidates[: args.limit]
    reviews = [review_row(row) for row in candidates]
    pass_reviews = [review for review in reviews if review.get("review_status") == "pass"]
    reviews_by_key = {
        (
            str(review.get("idea_id") or ""),
            str(review.get("eodhd_symbol") or ""),
            str(review.get("publication_date") or ""),
        ): review
        for review in reviews
    }

    write_reviews(reviews)
    write_validation(rows, reviews_by_key)
    added = write_training(existing_training, pass_reviews)

    status_counts = Counter(str(review.get("review_status")) for review in reviews)
    action_counts = Counter(str(review.get("training_action")) for review in reviews)
    reason_counts = Counter(str(review.get("reason")) for review in reviews)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_rows": len(rows),
        "existing_training_ready_rows": len(existing_training),
        "manual_internal_candidates": len(candidates),
        "new_training_ready_rows": added,
        "combined_training_ready_rows": len(existing_training) + added,
        "review_status_counts": dict(status_counts),
        "training_action_counts": dict(action_counts),
        "top_reasons": dict(reason_counts.most_common(25)),
        "outputs": {
            "row_reviews_csv": str(ROW_REVIEWS_CSV.resolve()),
            "row_reviews_jsonl": str(ROW_REVIEWS_JSONL.resolve()),
            "validation_csv": str(VALIDATION_OUT.resolve()),
            "training_ready_csv": str(TRAINING_READY_OUT.resolve()),
            "summary_json": str(SUMMARY_JSON.resolve()),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
