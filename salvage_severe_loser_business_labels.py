#!/usr/bin/env python3
"""Salvage non-zero severe losers with price, action, and business evidence.

This pass does not turn missing prices into zeroes. It only promotes rows where
the severe loss is an observed non-zero adjusted-price return, the EODHD cached
series recomputes cleanly, corporate-action adjustment sanity passes, and cached
fundamentals show a plausible business deterioration.

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
from review_remaining_manual_cases import financial_evidence, load_fundamentals, loser_supported, summary_text
from salvage_partial_horizon_labels import (
    BLOCKING_FLAGS,
    cached_prices,
    internal_adjustment_sanity,
    looks_non_common_instrument,
    parse_float,
    recompute_horizon,
    safe_symbol_filename,
    scalar,
    split_flags,
)
from salvage_split_event_internal_labels import split_aware_sanity


BASE_DIR = Path("eodhd_output/full_run")
VALIDATION_IN = BASE_DIR / "validation_results_with_split_event_salvage.csv"
TRAINING_READY_IN = BASE_DIR / "training_ready_after_split_event_salvage.csv"
SYMBOL_CACHE = BASE_DIR / "symbol_cache"

VALIDATION_OUT = BASE_DIR / "validation_results_with_severe_loser_salvage.csv"
TRAINING_READY_OUT = BASE_DIR / "training_ready_after_severe_loser_salvage.csv"
ROW_REVIEWS_CSV = BASE_DIR / "severe_loser_salvage_reviews.csv"
SUMMARY_JSON = BASE_DIR / "severe_loser_salvage_summary.json"

MIN_NONZERO_RETURN = 0.0
SEVERE_RETURN = 0.05


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


def severe_horizon_reviews(row: dict[str, str], prices: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    if row.get("math_validation_status") == "math_reproduced":
        horizon = row.get("review_target_horizon") or ""
        multiplier = parse_float(row.get("review_target_multiplier"))
        if horizon and multiplier is not None and MIN_NONZERO_RETURN < multiplier <= SEVERE_RETURN:
            review = recompute_horizon(row, prices, horizon)
            if review.get("status") == "pass":
                reviews[horizon] = review
    elif row.get("math_validation_status") == "math_incomplete":
        for horizon in ("1y", "3y", "5y", "10y", "20y"):
            multiplier = parse_float(row.get(f"perf_{horizon}"))
            if multiplier is not None and MIN_NONZERO_RETURN < multiplier <= SEVERE_RETURN:
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
        "reverse_split_provider_adjusted",
    }
    blocking = sorted((flags & BLOCKING_FLAGS) - allowed_blockers)
    if blocking:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": ";".join(blocking), "passed_horizons": {}}

    payload = load_payload(row.get("eodhd_symbol") or "")
    prices = cached_prices(payload)
    reviews = severe_horizon_reviews(row, prices)
    if not reviews:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": "no_recomputed_nonzero_severe_horizon", "passed_horizons": {}}
    delist_ok, delist_reason = endpoint_before_delist(row, reviews)
    if not delist_ok:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": delist_reason, "passed_horizons": {}}

    fundamentals = load_fundamentals(row.get("eodhd_symbol") or "")
    evidence = financial_evidence(row, fundamentals["_payload"])
    if not loser_supported(evidence):
        reason = "severe_loser_business_deterioration_not_supported_by_fundamentals"
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": reason,
            "passed_horizons": {},
            "financial_evidence": evidence,
        }

    normal_ok, normal_reason, normal_diag = internal_adjustment_sanity(row, payload, reviews)
    if normal_ok:
        sanity_reason = "severe_loser_eodhd_internal_sanity_and_business_deterioration_passed"
        sanity_diag = normal_diag
    elif normal_reason == "internal_sanity_split_in_validated_window_without_cross_provider":
        split_ok, split_reason, split_diag = split_aware_sanity(payload, reviews)
        if not split_ok:
            return {
                **base,
                "review_status": "manual_review",
                "training_action": "hold",
                "reason": split_reason,
                "passed_horizons": {},
                "split_sanity": split_diag,
                "financial_evidence": evidence,
            }
        sanity_reason = "severe_loser_split_event_continuity_and_business_deterioration_passed"
        sanity_diag = split_diag
    else:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": normal_reason,
            "passed_horizons": {},
            "internal_sanity": normal_diag,
            "financial_evidence": evidence,
        }

    return {
        **base,
        "review_status": "pass",
        "training_action": "add_to_training_ready",
        "reason": sanity_reason,
        "passed_horizons": {horizon: review["multiplier"] for horizon, review in reviews.items()},
        "internal_sanity": sanity_diag,
        "financial_evidence": evidence,
        "qualitative_summary": summary_text(row, evidence, sanity_reason),
        "confidence": 0.6,
    }


def select_candidates(rows: list[dict[str, str]], training_keys: set[tuple[str, str, str]]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for row in rows:
        if current_status(row, training_keys) in {"training_ready", "rejected", "provider_error"}:
            continue
        multiplier = parse_float(row.get("review_target_multiplier"))
        if multiplier is not None and MIN_NONZERO_RETURN < multiplier <= SEVERE_RETURN:
            candidates.append(row)
            continue
        if row.get("math_validation_status") == "math_incomplete":
            for horizon in ("1y", "3y", "5y", "10y", "20y"):
                value = parse_float(row.get(f"perf_{horizon}"))
                if value is not None and MIN_NONZERO_RETURN < value <= SEVERE_RETURN:
                    candidates.append(row)
                    break
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
        "qualitative_summary",
    ]
    with ROW_REVIEWS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for review in reviews:
            writer.writerow({field: scalar(review.get(field)) for field in fieldnames})


def write_validation(rows: list[dict[str, str]], reviews_by_key: dict[tuple[str, str, str], dict[str, Any]]) -> None:
    extra = [
        "severe_loser_salvage_status",
        "severe_loser_salvage_action",
        "severe_loser_salvage_reason",
        "severe_loser_salvage_passed_horizons",
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
                        "severe_loser_salvage_status": review.get("review_status"),
                        "severe_loser_salvage_action": review.get("training_action"),
                        "severe_loser_salvage_reason": review.get("reason"),
                        "severe_loser_salvage_passed_horizons": scalar(review.get("passed_horizons")),
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
                "math_validation_status": "severe_loser_business_verified",
                "review_stage": "severe_loser_business_eodhd_internal_sanity",
                "training_readiness": "training_ready",
                "review_status": "pass",
                "reviewed_at": now,
                "agent_c_status": "pass",
                "agent_c_reason": scalar(review.get("reason")),
                "agent_c_outcome_type": "severe_loser_business_deterioration",
                "source_count": "5",
                "fundamentals_name": scalar(review.get("company_name")),
                "fundamentals_type": scalar(review.get("fundamentals_type")),
                "original_warning_modes": scalar(review.get("warning_modes")),
                "original_failure_modes": scalar(review.get("failure_modes")),
            }
        )
        if "agent_c_qualitative_summary" in output:
            output["agent_c_qualitative_summary"] = scalar(review.get("qualitative_summary"))
        if "qualitative_summary" in output:
            output["qualitative_summary"] = scalar(review.get("qualitative_summary"))
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
        "severe_loser_candidates": len(candidates),
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
