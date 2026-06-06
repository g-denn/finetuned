#!/usr/bin/env python3
"""Merge high-confidence delisted-archive symbol repairs into training output."""

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
    parse_float,
    recompute_horizon,
    safe_symbol_filename,
    scalar,
    split_flags,
)


BASE_DIR = Path("eodhd_output/full_run")
REPAIR_DIR = Path("eodhd_output/provider_error_delisted_archive_repair")

VALIDATION_IN = BASE_DIR / "validation_results_with_provider_error_yahoo_identity.csv"
TRAINING_READY_IN = BASE_DIR / "training_ready_after_provider_error_yahoo_identity.csv"
REPAIR_VALIDATION = REPAIR_DIR / "validation_results.csv"
REPAIR_IDEAS = Path("eodhd_output/provider_error_delisted_archive_repair_ideas.json")

VALIDATION_OUT = BASE_DIR / "validation_results_with_delisted_archive_symbol_repair.csv"
TRAINING_READY_OUT = BASE_DIR / "training_ready_after_delisted_archive_symbol_repair.csv"
SUMMARY_JSON = BASE_DIR / "delisted_archive_symbol_repair_summary.json"
ROW_REVIEWS_CSV = BASE_DIR / "delisted_archive_symbol_repair_reviews.csv"

MIN_RETURN = 0.05
MAX_RETURN = 10.0


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("idea_id") or "", row.get("eodhd_symbol") or "", row.get("publication_date") or "")


def load_payload(symbol: str) -> dict[str, Any]:
    path = REPAIR_DIR / "symbol_cache" / safe_symbol_filename(symbol)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def ordinary_available_horizons(row: dict[str, str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for horizon in ("1y", "3y", "5y", "10y", "20y"):
        value = parse_float(row.get(f"perf_{horizon}"))
        if value is not None and MIN_RETURN < value < MAX_RETURN:
            values[horizon] = value
    return values


def review_repair(row: dict[str, str]) -> dict[str, Any]:
    base = {
        "idea_id": row.get("idea_id"),
        "raw_symbol": row.get("raw_symbol"),
        "eodhd_symbol": row.get("eodhd_symbol"),
        "publication_date": row.get("publication_date"),
        "company_name": row.get("fundamentals_name"),
        "math_validation_status": row.get("math_validation_status"),
        "fundamentals_type": row.get("fundamentals_type"),
        "fundamentals_status": row.get("fundamentals_status"),
        "warning_modes": row.get("warning_modes") or "",
        "failure_modes": row.get("failure_modes") or "",
    }
    flags = split_flags(row)
    if row.get("fundamentals_status") != "fetched":
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": "missing_fundamentals_identity", "passed_horizons": {}}
    if row.get("fundamentals_type") != "Common Stock":
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": "not_common_stock", "passed_horizons": {}}
    if looks_non_common_instrument(row):
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": "instrument_name_or_symbol_not_common_stock",
            "passed_horizons": {},
        }
    blocking = sorted(flags & BLOCKING_FLAGS)
    if blocking:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": ";".join(blocking),
            "passed_horizons": {},
        }
    payload = load_payload(row.get("eodhd_symbol") or "")
    prices = cached_prices(payload)
    reviews: dict[str, dict[str, Any]] = {}
    if row.get("math_validation_status") == "math_reproduced":
        horizon = row.get("review_target_horizon") or ""
        multiplier = parse_float(row.get("review_target_multiplier"))
        if not horizon or multiplier is None or not (MIN_RETURN < multiplier < MAX_RETURN):
            return {
                **base,
                "review_status": "manual_review",
                "training_action": "hold",
                "reason": "target_return_outside_ordinary_band",
                "passed_horizons": {},
            }
        review = recompute_horizon(row, prices, horizon)
        if review.get("status") != "pass":
            return {
                **base,
                "review_status": "manual_review",
                "training_action": "hold",
                "reason": str(review.get("reason") or "target_recompute_failed"),
                "passed_horizons": {},
                "eodhd_reviews": {horizon: review},
            }
        reviews[horizon] = review
    elif row.get("math_validation_status") == "math_incomplete":
        for horizon in ordinary_available_horizons(row):
            review = recompute_horizon(row, prices, horizon)
            if review.get("status") == "pass":
                reviews[horizon] = review
        if not reviews:
            return {
                **base,
                "review_status": "manual_review",
                "training_action": "hold",
                "reason": "no_recomputed_ordinary_horizons",
                "passed_horizons": {},
            }
    else:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": "still_provider_error", "passed_horizons": {}}

    sanity_ok, sanity_reason, sanity_diagnostics = internal_adjustment_sanity(row, payload, reviews)
    if not sanity_ok:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": sanity_reason,
            "passed_horizons": {},
            "eodhd_reviews": reviews,
            "internal_sanity": sanity_diagnostics,
        }
    return {
        **base,
        "review_status": "pass",
        "training_action": "add_to_training_ready",
        "reason": "delisted_archive_symbol_repair_eodhd_internal_sanity_passed",
        "passed_horizons": {horizon: review["multiplier"] for horizon, review in reviews.items()},
        "eodhd_reviews": reviews,
        "internal_sanity": sanity_diagnostics,
        "confidence": 0.66,
    }


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
        "math_validation_status",
        "fundamentals_type",
        "warning_modes",
        "failure_modes",
    ]
    with ROW_REVIEWS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for review in reviews:
            writer.writerow({field: scalar(review.get(field)) for field in fieldnames})


def write_validation(
    main_rows: list[dict[str, str]],
    repairs_by_idea: dict[str, dict[str, str]],
    reviews_by_idea: dict[str, dict[str, Any]],
    repair_ideas_by_idea: dict[str, dict[str, Any]],
) -> None:
    extra = [
        "delisted_archive_symbol_repair_status",
        "delisted_archive_symbol_repair_original_eodhd_symbol",
        "delisted_archive_symbol_repair_reason",
        "delisted_archive_symbol_repair_passed_horizons",
    ]
    fieldnames = list(main_rows[0].keys()) + [field for field in extra if field not in main_rows[0]]
    with VALIDATION_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in main_rows:
            review = reviews_by_idea.get(row.get("idea_id") or "")
            repair = repairs_by_idea.get(row.get("idea_id") or "")
            if review and repair and review.get("review_status") == "pass":
                idea = repair_ideas_by_idea.get(row.get("idea_id") or "") or {}
                output = {field: repair.get(field, row.get(field, "")) for field in fieldnames}
                output.update(
                    {
                        "delisted_archive_symbol_repair_status": "pass",
                        "delisted_archive_symbol_repair_original_eodhd_symbol": idea.get("original_eodhd_symbol") or row.get("eodhd_symbol") or "",
                        "delisted_archive_symbol_repair_reason": review.get("reason") or "",
                        "delisted_archive_symbol_repair_passed_horizons": scalar(review.get("passed_horizons")),
                    }
                )
                writer.writerow(output)
            else:
                output = dict(row)
                output.setdefault("delisted_archive_symbol_repair_status", "")
                output.setdefault("delisted_archive_symbol_repair_original_eodhd_symbol", "")
                output.setdefault("delisted_archive_symbol_repair_reason", "")
                output.setdefault("delisted_archive_symbol_repair_passed_horizons", "")
                writer.writerow(output)


def write_training(existing_training: list[dict[str, str]], pass_reviews: list[dict[str, Any]], repair_rows_by_idea: dict[str, dict[str, str]]) -> int:
    fieldnames = list(existing_training[0].keys())
    existing_idea_ids = {row.get("idea_id") for row in existing_training}
    now = datetime.now(UTC).isoformat()
    additions: list[dict[str, str]] = []
    for review in pass_reviews:
        idea_id = str(review.get("idea_id") or "")
        if idea_id in existing_idea_ids:
            continue
        repair = repair_rows_by_idea.get(idea_id) or {}
        output = {field: "" for field in fieldnames}
        output.update(
            {
                "idea_id": idea_id,
                "raw_symbol": scalar(review.get("raw_symbol")),
                "eodhd_symbol": scalar(review.get("eodhd_symbol")),
                "publication_date": scalar(review.get("publication_date")),
                "include_in_training": "true",
                "math_validation_status": "delisted_archive_symbol_repaired",
                "review_stage": "delisted_archive_symbol_repair_eodhd_internal",
                "training_readiness": "training_ready",
                "review_status": "pass",
                "reviewed_at": now,
                "agent_c_status": "pass",
                "agent_c_reason": scalar(review.get("reason")),
                "agent_c_outcome_type": "delisted_archive_symbol_repair",
                "source_count": "5",
                "fundamentals_name": repair.get("fundamentals_name") or scalar(review.get("company_name")),
                "fundamentals_type": repair.get("fundamentals_type") or scalar(review.get("fundamentals_type")),
                "fundamentals_sector": repair.get("fundamentals_sector") or "",
                "fundamentals_industry": repair.get("fundamentals_industry") or "",
                "fundamentals_market_cap": repair.get("fundamentals_market_cap") or "",
                "fundamentals_revenue_ttm": repair.get("fundamentals_revenue_ttm") or "",
                "fundamentals_profit_margin": repair.get("fundamentals_profit_margin") or "",
                "original_validation_status": "provider_error",
                "original_review_stage": "delisted_archive_symbol_repair",
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
    main_rows = load_csv(VALIDATION_IN)
    existing_training = load_csv(TRAINING_READY_IN)
    repair_rows = load_csv(REPAIR_VALIDATION)
    repair_ideas = json.loads(REPAIR_IDEAS.read_text(encoding="utf-8")) if REPAIR_IDEAS.exists() else []
    repair_ideas_by_idea = {str(row.get("idea_id") or ""): row for row in repair_ideas if isinstance(row, dict)}
    reviews = [review_repair(row) for row in repair_rows]
    pass_reviews = [review for review in reviews if review.get("review_status") == "pass"]
    repair_rows_by_idea = {row.get("idea_id") or "": row for row in repair_rows}
    reviews_by_idea = {str(review.get("idea_id") or ""): review for review in reviews}

    write_reviews(reviews)
    write_validation(main_rows, repair_rows_by_idea, reviews_by_idea, repair_ideas_by_idea)
    added = write_training(existing_training, pass_reviews, repair_rows_by_idea)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_rows": len(main_rows),
        "repair_rows": len(repair_rows),
        "safe_repair_rows": len(pass_reviews),
        "existing_training_ready_rows": len(existing_training),
        "new_training_ready_rows": added,
        "combined_training_ready_rows": len(existing_training) + added,
        "review_status_counts": dict(Counter(review.get("review_status") for review in reviews)),
        "training_action_counts": dict(Counter(review.get("training_action") for review in reviews)),
        "top_reasons": dict(Counter(str(review.get("reason")) for review in reviews).most_common(25)),
        "outputs": {
            "review_csv": str(ROW_REVIEWS_CSV.resolve()),
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
