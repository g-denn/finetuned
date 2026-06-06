#!/usr/bin/env python3
"""Salvage safe shorter-horizon labels from math-incomplete rows.

Some rows are marked ``math_incomplete`` only because the primary/long horizon
is missing after a delisting or early-ended price history. If a shorter horizon
has a real endpoint before that break, it can still be useful as a training
label.

This pass is intentionally conservative:

- common-stock identity and cached fundamentals are required
- reverse splits, non-common instruments, severe losers, and >=10x winners stay
  held for manual review
- every promoted horizon is recomputed from cached EODHD adjusted prices
- the endpoint must be before the delisting date when known and within the
  cached price history
- Yahoo must independently reproduce all promoted horizons unless
  ``--no-yahoo`` is explicitly passed. A separate opt-in tier can promote rows
  with unavailable Yahoo history only when EODHD-internal delisted/fundamental
  evidence passes stricter adjustment and endpoint sanity checks.

No API keys are read or stored.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from manual_review_validation import (
    ProviderSeries,
    add_years,
    fetch_yahoo_series,
    first_on_or_after,
    provider_returns,
    relative_diff,
    yahoo_symbol_from_eodhd,
)
from review_extreme_winners_15x import DO_NOT_AUTO_PROMOTE, parse_date


BASE_DIR = Path("eodhd_output/full_run")
VALIDATION_IN = BASE_DIR / "validation_results_with_all_manual_review.csv"
TRAINING_READY_IN = BASE_DIR / "training_ready_after_manual_review.csv"
SYMBOL_CACHE = BASE_DIR / "symbol_cache"

ROW_REVIEWS_CSV = BASE_DIR / "partial_horizon_salvage_reviews.csv"
ROW_REVIEWS_JSONL = BASE_DIR / "partial_horizon_salvage_reviews.jsonl"
VALIDATION_OUT = BASE_DIR / "validation_results_with_partial_horizon_salvage.csv"
TRAINING_READY_OUT = BASE_DIR / "training_ready_after_partial_horizon_salvage.csv"
SUMMARY_JSON = BASE_DIR / "partial_horizon_salvage_summary.json"

HORIZON_YEARS = {"1y": 1, "3y": 3, "5y": 5, "10y": 10, "20y": 20}
YAHOO_DIFF_LIMIT = 0.15
LARGE_WINNER_REQUIRES_MANUAL_REVIEW = 10.0
SEVERE_LOSER_REQUIRES_OUTCOME_MODEL = 0.05
RECOMPUTE_TOLERANCE = 0.001
MAX_START_GAP_DAYS = 14
MAX_ENDPOINT_GAP_DAYS = 14
FINAL_TRADING_CUSHION_DAYS = 14
MIN_PRICE_ROWS_PER_HORIZON_YEAR = 80
MAX_ADJUSTMENT_RATIO_DRIFT = 4.0
MAX_SINGLE_DAY_ADJUSTED_UP = 3.0
MIN_SINGLE_DAY_ADJUSTED_DOWN = 0.25

BLOCKING_FLAGS = {
    "reverse_split_provider_adjusted",
    "fundamentals_non_common_instrument",
    "provider_adjustment_factor_conflict",
    "provider_return_conflict",
    "lineage_override_requires_agent_review",
}

NON_COMMON_INSTRUMENT_MARKERS = (
    " warrant",
    " wt ",
    " wt.",
    " right",
    " unit",
    " note",
    " notes",
    " bond",
    " debenture",
    " preferred",
    " pfd",
    " fund",
    " etf",
)
NON_COMMON_SYMBOL_MARKERS = ("-WS", ".WS", "/WS", " WS", "-WT", ".WT", "/WT", " WT", "-U", ".U", "/U")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("idea_id") or "", row.get("eodhd_symbol") or "", row.get("publication_date") or "")


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "t", "yes", "y"}


def split_flags(row: dict[str, str]) -> set[str]:
    flags: set[str] = set()
    for column in ("failure_modes", "warning_modes", "manual_review_row_failures", "manual_review_row_warnings"):
        flags.update(part.strip() for part in (row.get(column) or "").replace("|", ";").split(";") if part.strip())
    return flags


def looks_non_common_instrument(row: dict[str, Any]) -> bool:
    name = f" {str(row.get('fundamentals_name') or row.get('company_name') or '').lower()} "
    raw_symbol = str(row.get("raw_symbol") or "").upper()
    eodhd_symbol = str(row.get("eodhd_symbol") or "").upper().removesuffix(".US")
    if any(marker in name for marker in NON_COMMON_INSTRUMENT_MARKERS):
        return True
    return any(marker in raw_symbol or marker in eodhd_symbol for marker in NON_COMMON_SYMBOL_MARKERS)


def safe_symbol_filename(symbol: str) -> str:
    return "".join(char if char.isalnum() or char in ".-" else "_" for char in symbol) + ".json"


def load_symbol_cache(symbol: str) -> dict[str, Any]:
    path = SYMBOL_CACHE / safe_symbol_filename(symbol)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def cached_prices(payload: dict[str, Any]) -> list[dict[str, Any]]:
    prices: list[dict[str, Any]] = []
    for item in payload.get("prices") or []:
        if not isinstance(item, dict):
            continue
        day = parse_date(str(item.get("date") or ""))
        adjusted = parse_float(item.get("adjusted_close"))
        close = parse_float(item.get("close"))
        if day is None or adjusted is None or adjusted <= 0:
            continue
        prices.append({"day": day, "adjusted_close": adjusted, "close": close})
    return sorted(prices, key=lambda item: item["day"])


def cached_splits(payload: dict[str, Any]) -> list[dict[str, Any]]:
    splits: list[dict[str, Any]] = []
    for item in payload.get("splits") or []:
        if not isinstance(item, dict):
            continue
        day = parse_date(str(item.get("date") or ""))
        if day is None:
            continue
        splits.append({"day": day, **item})
    return sorted(splits, key=lambda item: item["day"])


def first_price_on_or_after(prices: list[dict[str, Any]], target: date) -> dict[str, Any] | None:
    for item in prices:
        if item["day"] >= target:
            return item
    return None


def available_horizons(row: dict[str, str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for horizon in HORIZON_YEARS:
        value = parse_float(row.get(f"perf_{horizon}"))
        if value is not None and value > 0:
            values[horizon] = value
    return values


def base_hold_reasons(row: dict[str, str], horizon_values: dict[str, float]) -> list[str]:
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
    if any(value >= LARGE_WINNER_REQUIRES_MANUAL_REVIEW for value in horizon_values.values()):
        reasons.append("large_winner_requires_business_quality_review")
    if any(value <= SEVERE_LOSER_REQUIRES_OUTCOME_MODEL for value in horizon_values.values()):
        reasons.append("severe_loser_requires_bankruptcy_or_delisting_outcome_model")
    symbol = row.get("eodhd_symbol") or ""
    if symbol in DO_NOT_AUTO_PROMOTE:
        reasons.append("do_not_auto_promote_symbol")
    return reasons


def recompute_horizon(row: dict[str, str], prices: list[dict[str, Any]], horizon: str) -> dict[str, Any]:
    pub_date = parse_date(row.get("publication_date") or "")
    if pub_date is None:
        return {"status": "fail", "reason": "missing_publication_date"}
    start = first_price_on_or_after(prices, pub_date)
    if start is None:
        return {"status": "fail", "reason": "missing_cached_start_price"}
    start_gap_days = (start["day"] - pub_date).days
    if start_gap_days < 0 or start_gap_days > MAX_START_GAP_DAYS:
        return {
            "status": "hold",
            "reason": "start_price_too_far_from_publication_date",
            "publication_date": pub_date.isoformat(),
            "start_trade_date": start["day"].isoformat(),
            "start_gap_days": start_gap_days,
        }
    expected_start_date = parse_date(row.get("start_trade_date") or "")
    if expected_start_date and start["day"] != expected_start_date:
        return {
            "status": "fail",
            "reason": "cached_start_date_mismatch",
            "cached_start_trade_date": start["day"].isoformat(),
            "row_start_trade_date": expected_start_date.isoformat(),
        }
    target = add_years(pub_date, HORIZON_YEARS[horizon])
    endpoint = first_price_on_or_after(prices, target)
    if endpoint is None:
        return {"status": "fail", "reason": "missing_cached_endpoint_price", "target_date": target.isoformat()}
    endpoint_gap_days = (endpoint["day"] - target).days
    if endpoint_gap_days < 0 or endpoint_gap_days > MAX_ENDPOINT_GAP_DAYS:
        return {
            "status": "hold",
            "reason": "endpoint_price_too_far_from_target_date",
            "target_date": target.isoformat(),
            "endpoint_trade_date": endpoint["day"].isoformat(),
            "endpoint_gap_days": endpoint_gap_days,
        }
    multiplier = endpoint["adjusted_close"] / start["adjusted_close"]
    row_multiplier = parse_float(row.get(f"perf_{horizon}"))
    diff = relative_diff(multiplier, row_multiplier)
    if diff is None or diff > RECOMPUTE_TOLERANCE:
        return {
            "status": "fail",
            "reason": "cached_return_recompute_mismatch",
            "target_date": target.isoformat(),
            "endpoint_trade_date": endpoint["day"].isoformat(),
            "recomputed_multiplier": multiplier,
            "row_multiplier": row_multiplier,
            "relative_diff": diff,
        }
    last_price_date = parse_date(row.get("last_price_date") or "")
    if last_price_date and endpoint["day"] > last_price_date:
        return {
            "status": "hold",
            "reason": "endpoint_after_cached_last_price_date",
            "endpoint_trade_date": endpoint["day"].isoformat(),
            "last_price_date": last_price_date.isoformat(),
        }
    delisted_date = parse_date(row.get("fundamentals_delisted_date") or "")
    if delisted_date and endpoint["day"] >= delisted_date:
        return {
            "status": "hold",
            "reason": "endpoint_on_or_after_delisted_date",
            "endpoint_trade_date": endpoint["day"].isoformat(),
            "delisted_date": delisted_date.isoformat(),
        }
    if not delisted_date and parse_bool(row.get("is_in_delisted_cache")) and last_price_date:
        if endpoint["day"] > last_price_date - timedelta(days=FINAL_TRADING_CUSHION_DAYS):
            return {
                "status": "hold",
                "reason": "endpoint_too_close_to_final_trading_date_without_delisted_date",
                "endpoint_trade_date": endpoint["day"].isoformat(),
                "last_price_date": last_price_date.isoformat(),
            }
    return {
        "status": "pass",
        "reason": "eodhd_cached_adjusted_return_recomputed",
        "target_date": target.isoformat(),
        "start_trade_date": start["day"].isoformat(),
        "endpoint_trade_date": endpoint["day"].isoformat(),
        "start_gap_days": start_gap_days,
        "endpoint_gap_days": endpoint_gap_days,
        "start_adjusted_close": start["adjusted_close"],
        "endpoint_adjusted_close": endpoint["adjusted_close"],
        "multiplier": multiplier,
        "relative_diff": diff,
    }


def internal_adjustment_sanity(
    row: dict[str, str],
    payload: dict[str, Any],
    passed_eodhd: dict[str, dict[str, Any]],
) -> tuple[bool, str, dict[str, Any]]:
    """Check EODHD-only rows for adjustment discontinuities before promotion."""
    prices = cached_prices(payload)
    splits = cached_splits(payload)
    diagnostics: dict[str, Any] = {}
    for horizon, review in passed_eodhd.items():
        start_day = parse_date(review.get("start_trade_date") or "")
        endpoint_day = parse_date(review.get("endpoint_trade_date") or "")
        if start_day is None or endpoint_day is None:
            return False, "internal_sanity_missing_start_or_endpoint_date", diagnostics
        years = HORIZON_YEARS[horizon]
        window_prices = [item for item in prices if start_day <= item["day"] <= endpoint_day]
        minimum_rows = MIN_PRICE_ROWS_PER_HORIZON_YEAR * years
        if len(window_prices) < minimum_rows:
            return False, "internal_sanity_too_few_price_rows", {"horizon": horizon, "price_rows": len(window_prices)}
        window_splits = [item for item in splits if start_day <= item["day"] <= endpoint_day]
        if window_splits:
            return False, "internal_sanity_split_in_validated_window_without_cross_provider", {
                "horizon": horizon,
                "split_count": len(window_splits),
            }

        ratios: list[float] = []
        adjusted_values: list[float] = []
        for item in window_prices:
            close = item.get("close")
            adjusted = item.get("adjusted_close")
            if close is not None and close > 0 and adjusted is not None and adjusted > 0:
                ratios.append(adjusted / close)
            if adjusted is not None and adjusted > 0:
                adjusted_values.append(adjusted)
        if not ratios or not adjusted_values:
            return False, "internal_sanity_missing_adjustment_ratio_values", {"horizon": horizon}
        ratio_min = min(ratios)
        ratio_max = max(ratios)
        ratio_drift = ratio_max / ratio_min if ratio_min > 0 else math.inf
        if ratio_drift > MAX_ADJUSTMENT_RATIO_DRIFT:
            return False, "internal_sanity_adjustment_ratio_drift_too_large", {
                "horizon": horizon,
                "adjustment_ratio_min": ratio_min,
                "adjustment_ratio_max": ratio_max,
                "adjustment_ratio_drift": ratio_drift,
            }
        daily_moves = [
            adjusted_values[index] / adjusted_values[index - 1]
            for index in range(1, len(adjusted_values))
            if adjusted_values[index - 1] > 0
        ]
        max_up = max(daily_moves) if daily_moves else None
        min_down = min(daily_moves) if daily_moves else None
        if max_up is not None and max_up > MAX_SINGLE_DAY_ADJUSTED_UP:
            return False, "internal_sanity_single_day_adjusted_jump_too_large", {
                "horizon": horizon,
                "max_single_day_adjusted_move": max_up,
            }
        if min_down is not None and min_down < MIN_SINGLE_DAY_ADJUSTED_DOWN:
            return False, "internal_sanity_single_day_adjusted_drop_too_large", {
                "horizon": horizon,
                "min_single_day_adjusted_move": min_down,
            }
        diagnostics[horizon] = {
            "price_rows": len(window_prices),
            "split_count": len(window_splits),
            "adjustment_ratio_min": ratio_min,
            "adjustment_ratio_max": ratio_max,
            "adjustment_ratio_drift": ratio_drift,
            "max_single_day_adjusted_move": max_up,
            "min_single_day_adjusted_move": min_down,
        }
    return True, "eodhd_internal_adjustment_sanity_passed", diagnostics


def yahoo_check(
    row: dict[str, str],
    horizons: dict[str, dict[str, Any]],
    yahoo_cache: dict[tuple[str, str, str], ProviderSeries],
    require_yahoo: bool,
) -> tuple[dict[str, dict[str, Any]], str, int]:
    if not require_yahoo:
        return {}, yahoo_symbol_from_eodhd(row.get("eodhd_symbol") or ""), 0
    pub_date = parse_date(row.get("publication_date") or "")
    if pub_date is None:
        return {}, yahoo_symbol_from_eodhd(row.get("eodhd_symbol") or ""), 0
    max_years = max(HORIZON_YEARS[horizon] for horizon in horizons)
    end = min(add_years(pub_date, max_years), date.today())
    symbol = row.get("eodhd_symbol") or ""
    cache_key = (symbol, pub_date.isoformat(), end.isoformat())
    if cache_key not in yahoo_cache:
        yahoo_cache[cache_key] = fetch_yahoo_series(symbol, pub_date, end, retries=0)
    yahoo = yahoo_cache[cache_key]
    if not yahoo.prices:
        return {
            horizon: {
                "status": "hold",
                "reason": "yahoo_cross_check_unavailable",
                "warnings": yahoo.warnings,
            }
            for horizon in horizons
        }, yahoo.symbol, 0
    returns = provider_returns(yahoo, pub_date)
    checks: dict[str, dict[str, Any]] = {}
    for horizon, eodhd_review in horizons.items():
        yahoo_review = returns.get(horizon) or {}
        yahoo_multiplier = yahoo_review.get("multiplier")
        diff = relative_diff(eodhd_review.get("multiplier"), yahoo_multiplier)
        if yahoo_multiplier is None:
            checks[horizon] = {
                "status": "hold",
                "reason": "missing_yahoo_horizon_multiplier",
                "yahoo": yahoo_review,
            }
        elif diff is not None and diff > YAHOO_DIFF_LIMIT:
            checks[horizon] = {
                "status": "reject",
                "reason": "provider_return_conflict",
                "relative_diff": diff,
                "yahoo": yahoo_review,
            }
        else:
            checks[horizon] = {
                "status": "pass",
                "reason": "yahoo_cross_check_reproduced",
                "relative_diff": diff,
                "yahoo": yahoo_review,
            }
    return checks, yahoo.symbol, len(yahoo.prices)


def review_row(
    row: dict[str, str],
    yahoo_cache: dict[tuple[str, str, str], ProviderSeries],
    require_yahoo: bool,
    allow_eodhd_internal: bool,
) -> dict[str, Any]:
    horizon_values = available_horizons(row)
    base = {
        "idea_id": row.get("idea_id"),
        "raw_symbol": row.get("raw_symbol"),
        "eodhd_symbol": row.get("eodhd_symbol"),
        "publication_date": row.get("publication_date"),
        "company_name": row.get("fundamentals_name"),
        "fundamentals_type": row.get("fundamentals_type"),
        "fundamentals_is_delisted": row.get("fundamentals_is_delisted"),
        "fundamentals_delisted_date": row.get("fundamentals_delisted_date"),
        "is_in_delisted_cache": row.get("is_in_delisted_cache"),
        "candidate_horizons": sorted(horizon_values),
        "warning_modes": row.get("warning_modes") or "",
        "failure_modes": row.get("failure_modes") or "",
    }
    reasons = base_hold_reasons(row, horizon_values)
    if reasons:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": ";".join(reasons),
            "passed_horizons": {},
            "confidence": 0.45,
            "validation_tier": "held",
        }

    payload = load_symbol_cache(row.get("eodhd_symbol") or "")
    prices = cached_prices(payload)
    if not prices:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": "missing_or_invalid_cached_eodhd_prices",
            "passed_horizons": {},
            "confidence": 0.45,
            "validation_tier": "held",
        }

    eodhd_reviews = {horizon: recompute_horizon(row, prices, horizon) for horizon in horizon_values}
    failed = [value for value in eodhd_reviews.values() if value.get("status") == "fail"]
    held = [value for value in eodhd_reviews.values() if value.get("status") == "hold"]
    passed_eodhd = {horizon: value for horizon, value in eodhd_reviews.items() if value.get("status") == "pass"}
    if failed or held or not passed_eodhd:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": ";".join(sorted({str(item.get("reason")) for item in failed + held})) or "no_passed_horizons",
            "eodhd_reviews": eodhd_reviews,
            "passed_horizons": {},
            "confidence": 0.5,
            "validation_tier": "held",
        }

    yahoo_reviews, yahoo_symbol, yahoo_rows = yahoo_check(row, passed_eodhd, yahoo_cache, require_yahoo)
    if require_yahoo:
        yahoo_statuses = {review.get("status") for review in yahoo_reviews.values()}
        if "reject" in yahoo_statuses:
            return {
                **base,
                "review_status": "reject",
                "training_action": "exclude",
                "reason": "provider_return_conflict",
                "eodhd_reviews": eodhd_reviews,
                "yahoo_reviews": yahoo_reviews,
                "passed_horizons": {},
                "yahoo_symbol": yahoo_symbol,
                "yahoo_rows": yahoo_rows,
                "confidence": 0.75,
                "validation_tier": "provider_conflict_reject",
            }
        if yahoo_statuses != {"pass"}:
            yahoo_reasons = {str(review.get("reason")) for review in yahoo_reviews.values()}
            if allow_eodhd_internal and yahoo_reasons == {"yahoo_cross_check_unavailable"}:
                sanity_ok, sanity_reason, sanity_diagnostics = internal_adjustment_sanity(row, payload, passed_eodhd)
                if sanity_ok:
                    return {
                        **base,
                        "review_status": "pass",
                        "training_action": "add_to_training_ready",
                        "reason": "eodhd_delisted_fundamentals_internal_sanity_passed_yahoo_unavailable",
                        "eodhd_reviews": eodhd_reviews,
                        "yahoo_reviews": yahoo_reviews,
                        "internal_sanity": sanity_diagnostics,
                        "passed_horizons": {horizon: review["multiplier"] for horizon, review in passed_eodhd.items()},
                        "yahoo_symbol": yahoo_symbol,
                        "yahoo_rows": yahoo_rows,
                        "confidence": 0.64,
                        "validation_tier": "eodhd_internal_delisted_partial_horizon",
                    }
                return {
                    **base,
                    "review_status": "manual_review",
                    "training_action": "hold",
                    "reason": sanity_reason,
                    "eodhd_reviews": eodhd_reviews,
                    "yahoo_reviews": yahoo_reviews,
                    "internal_sanity": sanity_diagnostics,
                    "passed_horizons": {},
                    "yahoo_symbol": yahoo_symbol,
                    "yahoo_rows": yahoo_rows,
                    "confidence": 0.55,
                    "validation_tier": "held",
                }
            return {
                **base,
                "review_status": "manual_review",
                "training_action": "hold",
                "reason": ";".join(sorted({str(review.get("reason")) for review in yahoo_reviews.values()})),
                "eodhd_reviews": eodhd_reviews,
                "yahoo_reviews": yahoo_reviews,
                "passed_horizons": {},
                "yahoo_symbol": yahoo_symbol,
                "yahoo_rows": yahoo_rows,
                "confidence": 0.55,
                "validation_tier": "held",
            }

    return {
        **base,
        "review_status": "pass",
        "training_action": "add_to_training_ready",
        "reason": "partial_horizons_recomputed_identity_checked_endpoint_pre_delist_and_cross_provider_reproduced"
        if require_yahoo
        else "partial_horizons_recomputed_identity_checked_and_endpoint_pre_delist",
        "eodhd_reviews": eodhd_reviews,
        "yahoo_reviews": yahoo_reviews,
        "passed_horizons": {horizon: review["multiplier"] for horizon, review in passed_eodhd.items()},
        "yahoo_symbol": yahoo_symbol,
        "yahoo_rows": yahoo_rows,
        "confidence": 0.78 if require_yahoo else 0.66,
        "validation_tier": "cross_provider_partial_horizon" if require_yahoo else "eodhd_only_no_yahoo_requested",
    }


def scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


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
        "candidate_horizons",
        "passed_horizons",
        "confidence",
        "validation_tier",
        "yahoo_symbol",
        "yahoo_rows",
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
        "partial_horizon_review_status",
        "partial_horizon_training_action",
        "partial_horizon_reason",
        "partial_horizon_passed_horizons",
        "partial_horizon_yahoo_symbol",
        "partial_horizon_yahoo_rows",
        "partial_horizon_confidence",
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
                        "partial_horizon_review_status": review.get("review_status"),
                        "partial_horizon_training_action": review.get("training_action"),
                        "partial_horizon_reason": review.get("reason"),
                        "partial_horizon_passed_horizons": scalar(review.get("passed_horizons")),
                        "partial_horizon_yahoo_symbol": review.get("yahoo_symbol") or "",
                        "partial_horizon_yahoo_rows": scalar(review.get("yahoo_rows")),
                "partial_horizon_confidence": scalar(review.get("confidence")),
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
                "math_validation_status": "partial_horizon_verified",
                "review_stage": scalar(review.get("validation_tier") or "training_ready_partial_horizon_salvage"),
                "training_readiness": "training_ready",
                "review_status": "pass",
                "reviewed_at": now,
                "agent_b_yahoo_symbol": scalar(review.get("yahoo_symbol")),
                "agent_b_yahoo_rows": scalar(review.get("yahoo_rows")),
                "agent_c_status": "pass",
                "agent_c_reason": scalar(review.get("reason")),
                "agent_c_outcome_type": scalar(review.get("validation_tier") or "partial_horizon_verified"),
                "source_count": "4" if review.get("yahoo_rows") else "5",
                "fundamentals_name": scalar(review.get("company_name")),
                "fundamentals_type": scalar(review.get("fundamentals_type")),
                "original_validation_status": "math_incomplete",
                "original_review_stage": "math_incomplete_partial_horizon_salvage",
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
    parser.add_argument("--no-yahoo", action="store_true", help="Do not require Yahoo cross-provider reproduction.")
    parser.add_argument(
        "--allow-eodhd-internal-when-yahoo-missing",
        action="store_true",
        help="Promote rows with unavailable Yahoo history only if stricter EODHD delisted/fundamental sanity checks pass.",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = load_csv(VALIDATION_IN)
    existing_training = load_csv(TRAINING_READY_IN)
    training_keys = {row_key(row) for row in existing_training}
    candidates = [
        row
        for row in rows
        if row_key(row) not in training_keys
        and row.get("math_validation_status") == "math_incomplete"
        and available_horizons(row)
    ]
    if args.limit:
        candidates = candidates[: args.limit]

    yahoo_cache: dict[tuple[str, str, str], ProviderSeries] = {}
    reviews = [
        review_row(
            row,
            yahoo_cache,
            require_yahoo=not args.no_yahoo,
            allow_eodhd_internal=args.allow_eodhd_internal_when_yahoo_missing,
        )
        for row in candidates
    ]
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
        "require_yahoo": not args.no_yahoo,
        "allow_eodhd_internal_when_yahoo_missing": args.allow_eodhd_internal_when_yahoo_missing,
        "input_rows": len(rows),
        "existing_training_ready_rows": len(existing_training),
        "partial_horizon_candidates": len(candidates),
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
