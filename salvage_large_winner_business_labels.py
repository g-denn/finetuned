#!/usr/bin/env python3
"""Salvage large winners with internal price sanity and business evidence.

This is for rows whose math is reproduced but were held because Yahoo was not
available or the first extreme-winner pass required a narrow revenue-growth
shape. It keeps special situations held and promotes only common-stock rows with
clean cached price/corporate-action evidence plus strong fundamentals.

No API keys are read or stored.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from review_extreme_winners_15x import (
    DO_NOT_AUTO_PROMOTE,
    business_summary,
    financial_quality,
    load_fundamentals,
    parse_date,
)
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
VALIDATION_IN = BASE_DIR / "validation_results_with_severe_loser_salvage.csv"
TRAINING_READY_IN = BASE_DIR / "training_ready_after_severe_loser_salvage.csv"
SYMBOL_CACHE = BASE_DIR / "symbol_cache"

VALIDATION_OUT = BASE_DIR / "validation_results_with_large_winner_salvage.csv"
TRAINING_READY_OUT = BASE_DIR / "training_ready_after_large_winner_salvage.csv"
ROW_REVIEWS_CSV = BASE_DIR / "large_winner_salvage_reviews.csv"
SUMMARY_JSON = BASE_DIR / "large_winner_salvage_summary.json"

MIN_LARGE_WINNER = 10.0
MAX_AUTO_WINNER = 50.0


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


def endpoint_before_delist(row: dict[str, str], review: dict[str, Any]) -> tuple[bool, str]:
    delisted_date = parse_date(row.get("fundamentals_delisted_date") or "")
    if delisted_date is None:
        return True, "no_delisted_date"
    endpoint = parse_date(review.get("endpoint_trade_date") or "")
    if endpoint is None:
        return False, "missing_endpoint_date"
    if endpoint >= delisted_date:
        return False, "endpoint_on_or_after_delisted_date"
    return True, "endpoint_precedes_delisting"


def supplemental_quality(row: dict[str, str], quality: dict[str, Any]) -> tuple[bool, str]:
    revenue_growth = quality.get("revenue_growth")
    net_income_growth = quality.get("net_income_growth")
    end_margin = quality.get("end_net_margin")
    operating_margin = quality.get("end_operating_margin")
    start_net_income = quality.get("start_net_income")
    end_net_income = quality.get("end_net_income")
    if not end_net_income or end_net_income <= 0:
        return False, "large_winner_end_profit_missing"
    margin_ok = (end_margin is not None and end_margin >= 0.08) or (operating_margin is not None and operating_margin >= 0.10)
    revenue_ok = revenue_growth is not None and revenue_growth >= 1.5
    strong_revenue = revenue_growth is not None and revenue_growth >= 5.0
    strong_profit = net_income_growth is not None and net_income_growth >= 5.0
    exceptional_profit = net_income_growth is not None and net_income_growth >= 8.0
    loss_to_profit = start_net_income is not None and start_net_income <= 0 and end_net_income > 0
    if margin_ok and revenue_ok and strong_profit:
        return True, "profit_compounder_fundamentals_support_large_winner"
    if strong_revenue and (exceptional_profit or loss_to_profit):
        return True, "scale_growth_fundamentals_support_large_winner"
    return False, "large_winner_business_quality_not_supported_by_fundamentals"


def review_row(row: dict[str, str]) -> dict[str, Any]:
    symbol = row.get("eodhd_symbol") or ""
    multiplier = parse_float(row.get("review_target_multiplier"))
    base = {
        "idea_id": row.get("idea_id"),
        "raw_symbol": row.get("raw_symbol"),
        "eodhd_symbol": symbol,
        "publication_date": row.get("publication_date"),
        "company_name": row.get("fundamentals_name"),
        "target_horizon": row.get("review_target_horizon"),
        "target_multiplier": multiplier,
        "fundamentals_type": row.get("fundamentals_type"),
        "warning_modes": row.get("warning_modes") or "",
        "failure_modes": row.get("failure_modes") or "",
    }
    if multiplier is None or multiplier < MIN_LARGE_WINNER or multiplier > MAX_AUTO_WINNER:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": "winner_multiplier_outside_auto_band", "passed_horizons": {}}
    if symbol in DO_NOT_AUTO_PROMOTE:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": DO_NOT_AUTO_PROMOTE[symbol], "passed_horizons": {}}
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
        "extreme_return_requires_stronger_evidence",
    }
    blocking = sorted((flags & BLOCKING_FLAGS) - allowed_blockers)
    if blocking:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": ";".join(blocking), "passed_horizons": {}}

    payload = load_payload(symbol)
    prices = cached_prices(payload)
    horizon = row.get("review_target_horizon") or ""
    review = recompute_horizon(row, prices, horizon)
    if review.get("status") != "pass":
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": str(review.get("reason") or "target_horizon_recompute_failed"), "passed_horizons": {}}
    delist_ok, delist_reason = endpoint_before_delist(row, review)
    if not delist_ok:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": delist_reason, "passed_horizons": {}}

    normal_ok, normal_reason, normal_diag = internal_adjustment_sanity(row, payload, {horizon: review})
    if normal_ok:
        sanity_reason = "large_winner_eodhd_internal_sanity_passed"
        sanity_diag = normal_diag
    elif normal_reason == "internal_sanity_split_in_validated_window_without_cross_provider":
        split_ok, split_reason, split_diag = split_aware_sanity(payload, {horizon: review})
        if not split_ok:
            return {**base, "review_status": "manual_review", "training_action": "hold", "reason": split_reason, "passed_horizons": {}, "split_sanity": split_diag}
        sanity_reason = "large_winner_split_event_continuity_passed"
        sanity_diag = split_diag
    else:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": normal_reason, "passed_horizons": {}, "internal_sanity": normal_diag}

    fundamentals = load_fundamentals(symbol)
    fund_payload = fundamentals["_payload"]
    quality_ok, quality_reason, quality = financial_quality(row, fund_payload)
    if not quality_ok:
        quality_ok, quality_reason = supplemental_quality(row, quality)
    if not quality_ok:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": quality_reason,
            "passed_horizons": {},
            "quality_evidence": quality,
        }

    reason = f"{sanity_reason};{quality_reason}"
    return {
        **base,
        "review_status": "pass",
        "training_action": "add_to_training_ready",
        "reason": reason,
        "passed_horizons": {horizon: review["multiplier"]},
        "internal_sanity": sanity_diag,
        "quality_evidence": quality,
        "qualitative_summary": business_summary(row, quality, reason),
        "confidence": 0.62,
    }


def select_candidates(rows: list[dict[str, str]], training_keys: set[tuple[str, str, str]]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for row in rows:
        if current_status(row, training_keys) in {"training_ready", "rejected", "provider_error", "math_incomplete"}:
            continue
        multiplier = parse_float(row.get("review_target_multiplier"))
        if multiplier is not None and multiplier >= MIN_LARGE_WINNER:
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
        "large_winner_salvage_status",
        "large_winner_salvage_action",
        "large_winner_salvage_reason",
        "large_winner_salvage_passed_horizons",
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
                        "large_winner_salvage_status": review.get("review_status"),
                        "large_winner_salvage_action": review.get("training_action"),
                        "large_winner_salvage_reason": review.get("reason"),
                        "large_winner_salvage_passed_horizons": scalar(review.get("passed_horizons")),
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
                "math_validation_status": "large_winner_business_verified",
                "review_stage": "large_winner_business_eodhd_internal_sanity",
                "training_readiness": "training_ready",
                "review_status": "pass",
                "reviewed_at": now,
                "agent_c_status": "pass",
                "agent_c_reason": scalar(review.get("reason")),
                "agent_c_outcome_type": "large_winner_business_quality",
                "source_count": "5",
                "fundamentals_name": scalar(review.get("company_name")),
                "fundamentals_type": scalar(review.get("fundamentals_type")),
                "original_warning_modes": scalar(review.get("warning_modes")),
                "original_failure_modes": scalar(review.get("failure_modes")),
            }
        )
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
        "large_winner_candidates": len(candidates),
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
