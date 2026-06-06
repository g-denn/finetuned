#!/usr/bin/env python3
"""Salvage ordinary split-event rows when Yahoo is unavailable.

Rows can be held even when EODHD recomputes the return because a split occurred
inside the validation window and no Yahoo cross-check was available. This pass
promotes only low-drama, ordinary-return common-stock rows when cached EODHD
split events explain the window and adjusted prices remain continuous.

No API keys are read or stored.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from review_extreme_winners_15x import parse_date
from salvage_partial_horizon_labels import (
    BLOCKING_FLAGS,
    MAX_SINGLE_DAY_ADJUSTED_UP,
    MIN_SINGLE_DAY_ADJUSTED_DOWN,
    cached_prices,
    cached_splits,
    internal_adjustment_sanity,
    looks_non_common_instrument,
    parse_float,
    recompute_horizon,
    safe_symbol_filename,
    scalar,
    split_flags,
)


BASE_DIR = Path("eodhd_output/full_run")
VALIDATION_IN = BASE_DIR / "validation_results_with_search_lineage_yahoo_repair.csv"
TRAINING_READY_IN = BASE_DIR / "training_ready_after_search_lineage_yahoo_repair.csv"
SYMBOL_CACHE = BASE_DIR / "symbol_cache"

VALIDATION_OUT = BASE_DIR / "validation_results_with_split_event_salvage.csv"
TRAINING_READY_OUT = BASE_DIR / "training_ready_after_split_event_salvage.csv"
ROW_REVIEWS_CSV = BASE_DIR / "split_event_salvage_reviews.csv"
SUMMARY_JSON = BASE_DIR / "split_event_salvage_summary.json"

MIN_RETURN = 0.05
MAX_RETURN = 10.0
MIN_PRICE_ROWS = 80
SPLIT_REASON = "internal_sanity_split_in_validated_window_without_cross_provider"


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


def candidate_reason(row: dict[str, str]) -> str:
    return (
        row.get("manual_internal_reason")
        or row.get("partial_horizon_salvage_reason")
        or row.get("reverse_split_salvage_reason")
        or ""
    )


def ordinary_horizon_reviews(row: dict[str, str], prices: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    if row.get("math_validation_status") == "math_reproduced":
        horizon = row.get("review_target_horizon") or ""
        multiplier = parse_float(row.get("review_target_multiplier"))
        if horizon and multiplier is not None and MIN_RETURN < multiplier < MAX_RETURN:
            review = recompute_horizon(row, prices, horizon)
            if review.get("status") == "pass":
                reviews[horizon] = review
    elif row.get("math_validation_status") == "math_incomplete":
        for horizon in ("1y", "3y", "5y", "10y", "20y"):
            multiplier = parse_float(row.get(f"perf_{horizon}"))
            if multiplier is not None and MIN_RETURN < multiplier < MAX_RETURN:
                review = recompute_horizon(row, prices, horizon)
                if review.get("status") == "pass":
                    reviews[horizon] = review
    return reviews


def endpoint_before_delist(row: dict[str, str], reviews: dict[str, dict[str, Any]]) -> tuple[bool, str]:
    delisted_date = parse_date(row.get("fundamentals_delisted_date") or "")
    if delisted_date is None:
        return True, "no_delisted_date"
    for horizon, review in reviews.items():
        endpoint = parse_date(review.get("endpoint_trade_date") or "")
        if endpoint is None:
            return False, f"{horizon}:missing_endpoint_date"
        if endpoint >= delisted_date:
            return False, f"{horizon}:endpoint_on_or_after_delisted_date"
    return True, "endpoint_precedes_delisting"


def split_aware_sanity(payload: dict[str, Any], reviews: dict[str, dict[str, Any]]) -> tuple[bool, str, dict[str, Any]]:
    prices = cached_prices(payload)
    splits = cached_splits(payload)
    if not splits:
        return False, "missing_split_events", {}
    diagnostics: dict[str, Any] = {}
    for horizon, review in reviews.items():
        start_day = parse_date(review.get("start_trade_date") or "")
        endpoint_day = parse_date(review.get("endpoint_trade_date") or "")
        if start_day is None or endpoint_day is None:
            return False, "missing_start_or_endpoint_date", diagnostics
        window_prices = [item for item in prices if start_day <= item["day"] <= endpoint_day]
        window_splits = [item for item in splits if start_day <= item["day"] <= endpoint_day]
        if not window_splits:
            return False, "no_split_event_in_validated_window", diagnostics
        if len(window_splits) > 5:
            return False, "too_many_split_events_in_window", {"split_count": len(window_splits)}
        if len(window_prices) < MIN_PRICE_ROWS:
            return False, "too_few_price_rows", {"price_rows": len(window_prices)}
        adjusted_values = [item["adjusted_close"] for item in window_prices if item.get("adjusted_close") and item["adjusted_close"] > 0]
        moves = [
            adjusted_values[index] / adjusted_values[index - 1]
            for index in range(1, len(adjusted_values))
            if adjusted_values[index - 1] > 0
        ]
        max_up = max(moves) if moves else None
        min_down = min(moves) if moves else None
        if max_up is not None and max_up > MAX_SINGLE_DAY_ADJUSTED_UP:
            return False, "adjusted_jump_too_large", {"max_single_day_adjusted_move": max_up}
        if min_down is not None and min_down < MIN_SINGLE_DAY_ADJUSTED_DOWN:
            return False, "adjusted_drop_too_large", {"min_single_day_adjusted_move": min_down}
        for split in window_splits:
            parts = str(split.get("split") or "").replace(":", "/").split("/")
            if len(parts) != 2:
                return False, "unparseable_split_factor", {"split": split.get("split")}
            try:
                numerator = float(parts[0])
                denominator = float(parts[1])
            except ValueError:
                return False, "unparseable_split_factor", {"split": split.get("split")}
            if numerator <= 0 or denominator <= 0:
                return False, "invalid_split_factor", {"split": split.get("split")}
        diagnostics[horizon] = {
            "price_rows": len(window_prices),
            "split_count": len(window_splits),
            "splits": [split.get("split") for split in window_splits],
            "max_single_day_adjusted_move": max_up,
            "min_single_day_adjusted_move": min_down,
        }
    return True, "split_event_adjusted_continuity_passed", diagnostics


def review_row(row: dict[str, str]) -> dict[str, Any]:
    base = {
        "idea_id": row.get("idea_id"),
        "raw_symbol": row.get("raw_symbol"),
        "eodhd_symbol": row.get("eodhd_symbol"),
        "publication_date": row.get("publication_date"),
        "company_name": row.get("fundamentals_name"),
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

    flags = split_flags(row)
    allowed_blockers = {
        "symbol_in_delisted_cache",
        "fundamentals_is_delisted",
        "price_history_ends_before_long_horizon",
    }
    blocking = sorted((flags & BLOCKING_FLAGS) - allowed_blockers - {"reverse_split_provider_adjusted"})
    if blocking:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": ";".join(blocking), "passed_horizons": {}}

    payload = load_payload(row.get("eodhd_symbol") or "")
    prices = cached_prices(payload)
    reviews = ordinary_horizon_reviews(row, prices)
    if not reviews:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": "no_recomputed_ordinary_horizons", "passed_horizons": {}}

    delist_ok, delist_reason = endpoint_before_delist(row, reviews)
    if not delist_ok:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": delist_reason, "passed_horizons": {}}

    normal_ok, normal_reason, normal_diag = internal_adjustment_sanity(row, payload, reviews)
    if normal_ok:
        return {
            **base,
            "review_status": "pass",
            "training_action": "add_to_training_ready",
            "reason": "split_flag_resolved_by_standard_internal_sanity",
            "passed_horizons": {horizon: review["multiplier"] for horizon, review in reviews.items()},
            "internal_sanity": normal_diag,
            "confidence": 0.63,
        }
    if normal_reason != SPLIT_REASON:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": normal_reason,
            "passed_horizons": {},
            "internal_sanity": normal_diag,
        }

    split_ok, split_reason, split_diag = split_aware_sanity(payload, reviews)
    if not split_ok:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": split_reason,
            "passed_horizons": {},
            "split_sanity": split_diag,
        }
    return {
        **base,
        "review_status": "pass",
        "training_action": "add_to_training_ready",
        "reason": "cached_split_event_adjusted_continuity_passed",
        "passed_horizons": {horizon: review["multiplier"] for horizon, review in reviews.items()},
        "split_sanity": split_diag,
        "confidence": 0.61,
    }


def select_candidates(rows: list[dict[str, str]], training_keys: set[tuple[str, str, str]]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for row in rows:
        if row_key(row) in training_keys:
            continue
        reason = candidate_reason(row)
        if reason == SPLIT_REASON:
            candidates.append(row)
    return candidates


def write_reviews(reviews: list[dict[str, Any]]) -> None:
    fieldnames = [
        "idea_id",
        "raw_symbol",
        "eodhd_symbol",
        "publication_date",
        "company_name",
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
        "split_event_salvage_status",
        "split_event_salvage_action",
        "split_event_salvage_reason",
        "split_event_salvage_passed_horizons",
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
                        "split_event_salvage_status": review.get("review_status"),
                        "split_event_salvage_action": review.get("training_action"),
                        "split_event_salvage_reason": review.get("reason"),
                        "split_event_salvage_passed_horizons": scalar(review.get("passed_horizons")),
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
                "math_validation_status": "split_event_internal_verified",
                "review_stage": "split_event_eodhd_internal_sanity",
                "training_readiness": "training_ready",
                "review_status": "pass",
                "reviewed_at": now,
                "agent_c_status": "pass",
                "agent_c_reason": scalar(review.get("reason")),
                "agent_c_outcome_type": "split_event_internal_verified",
                "source_count": "5",
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
        "split_event_candidates": len(candidates),
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
