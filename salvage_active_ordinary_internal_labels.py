#!/usr/bin/env python3
"""Salvage ordinary active rows held by the delisted-only internal verifier.

The manual-internal pass intentionally focused on delisted/early-ended rows.
This pass handles the opposite: ordinary common-stock rows that were held only
because they were *not* delisted/early-ended, while the target return is already
recomputed from cached EODHD prices and passes internal adjustment sanity.

No API keys are read or stored.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from salvage_partial_horizon_labels import (
    BLOCKING_FLAGS,
    cached_prices,
    internal_adjustment_sanity,
    looks_non_common_instrument,
    parse_bool,
    parse_float,
    recompute_horizon,
    safe_symbol_filename,
    scalar,
    split_flags,
)


BASE_DIR = Path("eodhd_output/full_run")
VALIDATION_IN = BASE_DIR / "validation_results_with_sec_yahoo_salvage.csv"
TRAINING_READY_IN = BASE_DIR / "training_ready_after_sec_yahoo_salvage.csv"
SYMBOL_CACHE = BASE_DIR / "symbol_cache"

VALIDATION_OUT = BASE_DIR / "validation_results_with_active_ordinary_salvage.csv"
TRAINING_READY_OUT = BASE_DIR / "training_ready_after_active_ordinary_salvage.csv"
ROW_REVIEWS_CSV = BASE_DIR / "active_ordinary_salvage_reviews.csv"
SUMMARY_JSON = BASE_DIR / "active_ordinary_salvage_summary.json"

MIN_RETURN = 0.05
MAX_RETURN = 10.0


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("idea_id") or "", row.get("eodhd_symbol") or "", row.get("publication_date") or "")


def load_payload(symbol: str) -> dict[str, Any]:
    path = SYMBOL_CACHE / safe_symbol_filename(symbol)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def current_status(row: dict[str, str], training_keys: set[tuple[str, str, str]]) -> str:
    if row_key(row) in training_keys:
        return "training_ready"
    if (
        row.get("manual_review_status") == "reject"
        or row.get("training_readiness") == "rejected"
        or row.get("extreme_15x_review_status") == "reject"
        or row.get("remaining_manual_review_status") == "reject"
    ):
        return "rejected"
    if row.get("math_validation_status") in {"provider_error", "math_incomplete"}:
        return row.get("math_validation_status") or ""
    return "manual_review_remaining"


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
        "warning_modes": row.get("warning_modes") or "",
        "failure_modes": row.get("failure_modes") or "",
    }
    if row.get("fundamentals_status") != "fetched":
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": "missing_fundamentals_identity", "passed_horizons": {}}
    if row.get("fundamentals_type") != "Common Stock":
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": "not_common_stock", "passed_horizons": {}}
    if looks_non_common_instrument(row):
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": "instrument_name_or_symbol_not_common_stock", "passed_horizons": {}}
    if not horizon or multiplier is None or not (MIN_RETURN < multiplier < MAX_RETURN):
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": "target_return_outside_ordinary_band", "passed_horizons": {}}
    flags = split_flags(row)
    if parse_bool(row.get("is_in_delisted_cache")) or parse_bool(row.get("fundamentals_is_delisted")) or "symbol_in_delisted_cache" in flags:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": "delisted_or_archive_flag_not_active_ordinary", "passed_horizons": {}}
    blocking = sorted(flags & BLOCKING_FLAGS)
    if blocking:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": ";".join(blocking), "passed_horizons": {}}

    payload = load_payload(row.get("eodhd_symbol") or "")
    prices = cached_prices(payload)
    review = recompute_horizon(row, prices, horizon)
    if review.get("status") != "pass":
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": str(review.get("reason") or "target_horizon_recompute_failed"), "passed_horizons": {}}
    sanity_ok, sanity_reason, sanity_diagnostics = internal_adjustment_sanity(row, payload, {horizon: review})
    if not sanity_ok:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": sanity_reason, "passed_horizons": {}, "internal_sanity": sanity_diagnostics}
    return {
        **base,
        "review_status": "pass",
        "training_action": "add_to_training_ready",
        "reason": "active_ordinary_eodhd_internal_sanity_passed",
        "passed_horizons": {horizon: review["multiplier"]},
        "internal_sanity": sanity_diagnostics,
        "confidence": 0.6,
    }


def select_candidates(rows: list[dict[str, str]], training_keys: set[tuple[str, str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if current_status(row, training_keys) == "manual_review_remaining"
        and row.get("manual_internal_reason") == "not_delisted_or_early_ended_history"
    ]


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
        "warning_modes",
        "failure_modes",
    ]
    with ROW_REVIEWS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for review in reviews:
            writer.writerow({field: scalar(review.get(field)) for field in fieldnames})


def write_validation(rows: list[dict[str, str]], reviews_by_key: dict[tuple[str, str, str], dict[str, Any]]) -> None:
    extra = [
        "active_ordinary_salvage_status",
        "active_ordinary_salvage_action",
        "active_ordinary_salvage_reason",
        "active_ordinary_salvage_passed_horizons",
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
                        "active_ordinary_salvage_status": review.get("review_status"),
                        "active_ordinary_salvage_action": review.get("training_action"),
                        "active_ordinary_salvage_reason": review.get("reason"),
                        "active_ordinary_salvage_passed_horizons": scalar(review.get("passed_horizons")),
                    }
                )
            writer.writerow(output)


def write_training(existing_training: list[dict[str, str]], pass_reviews: list[dict[str, Any]]) -> int:
    fieldnames = list(existing_training[0].keys())
    existing_idea_ids = {row.get("idea_id") for row in existing_training}
    now = datetime.now(UTC).isoformat()
    additions: list[dict[str, str]] = []
    for review in pass_reviews:
        idea_id = str(review.get("idea_id") or "")
        if idea_id in existing_idea_ids:
            continue
        output = {field: "" for field in fieldnames}
        output.update(
            {
                "idea_id": idea_id,
                "raw_symbol": scalar(review.get("raw_symbol")),
                "eodhd_symbol": scalar(review.get("eodhd_symbol")),
                "publication_date": scalar(review.get("publication_date")),
                "include_in_training": "true",
                "math_validation_status": "active_ordinary_internal_verified",
                "review_stage": "active_ordinary_eodhd_internal_sanity",
                "training_readiness": "training_ready",
                "review_status": "pass",
                "reviewed_at": now,
                "agent_c_status": "pass",
                "agent_c_reason": scalar(review.get("reason")),
                "agent_c_outcome_type": "active_ordinary_internal_verified",
                "source_count": "4",
                "fundamentals_name": scalar(review.get("company_name")),
                "fundamentals_type": scalar(review.get("fundamentals_type")),
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
    rows = load_csv(VALIDATION_IN)
    existing_training = load_csv(TRAINING_READY_IN)
    training_keys = {row_key(row) for row in existing_training}
    candidates = select_candidates(rows, training_keys)
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
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_rows": len(rows),
        "active_ordinary_candidates": len(candidates),
        "existing_training_ready_rows": len(existing_training),
        "new_training_ready_rows": added,
        "combined_training_ready_rows": len(existing_training) + added,
        "review_status_counts": dict(Counter(review.get("review_status") for review in reviews)),
        "training_action_counts": dict(Counter(review.get("training_action") for review in reviews)),
        "top_reasons": dict(Counter(str(review.get("reason")) for review in reviews).most_common(25)),
        "outputs": {
            "row_reviews_csv": str(ROW_REVIEWS_CSV.resolve()),
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
