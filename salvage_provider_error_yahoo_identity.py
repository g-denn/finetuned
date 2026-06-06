#!/usr/bin/env python3
"""Recover provider-error rows using Yahoo prices plus EODHD identity evidence.

This is a lower-priority repair path for rows where EODHD fetched fundamentals
for the security but EODHD price history was empty/missing at the publication
date. It does not use Yahoo blindly:

- EODHD fundamentals must identify the instrument as common stock
- name/symbol markers for warrants, units, notes, preferreds, funds, etc. block
  the row
- only ordinary returns between 0.05x and 10x are accepted
- Yahoo adjusted-price series must pass date-gap and adjustment sanity checks
- delisted endpoints must be before the EODHD delisted date when known

No API keys are read or stored.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from manual_review_validation import PricePoint, ProviderSeries, add_years, first_on_or_after, relative_diff
from salvage_partial_horizon_labels import (
    MAX_ADJUSTMENT_RATIO_DRIFT,
    MAX_ENDPOINT_GAP_DAYS,
    MAX_SINGLE_DAY_ADJUSTED_UP,
    MAX_START_GAP_DAYS,
    MIN_SINGLE_DAY_ADJUSTED_DOWN,
    looks_non_common_instrument,
    parse_bool,
    parse_float,
    scalar,
)
from review_extreme_winners_15x import parse_date


BASE_DIR = Path("eodhd_output/full_run")
VALIDATION_IN = BASE_DIR / "validation_results_with_provider_symbol_repair.csv"
TRAINING_READY_IN = BASE_DIR / "training_ready_after_provider_symbol_repair.csv"
IDEAS_JSON = Path("eodhd_output/all_ideas.json")

ROW_REVIEWS_CSV = BASE_DIR / "provider_error_yahoo_identity_reviews.csv"
ROW_REVIEWS_JSONL = BASE_DIR / "provider_error_yahoo_identity_reviews.jsonl"
VALIDATION_OUT = BASE_DIR / "validation_results_with_provider_error_yahoo_identity.csv"
TRAINING_READY_OUT = BASE_DIR / "training_ready_after_provider_error_yahoo_identity.csv"
SUMMARY_JSON = BASE_DIR / "provider_error_yahoo_identity_summary.json"

HORIZON_YEARS = (1, 3, 5, 10, 20)
MIN_RETURN = 0.05
MAX_RETURN = 10.0
MIN_PRICE_ROWS_PER_YEAR = 80


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("idea_id") or "", row.get("eodhd_symbol") or "", row.get("publication_date") or "")


def load_ideas() -> dict[str, dict[str, Any]]:
    payload = json.loads(IDEAS_JSON.read_text(encoding="utf-8"))
    return {str(row.get("idea_id") or ""): row for row in payload if isinstance(row, dict)}


def yahoo_symbol_for_row(row: dict[str, str], idea: dict[str, Any] | None) -> str:
    yahoo = str((idea or {}).get("yahoo_symbol") or "").strip()
    if yahoo:
        return yahoo
    symbol = row.get("eodhd_symbol") or ""
    if symbol.endswith(".US"):
        return symbol[:-3]
    suffix_map = {
        ".LSE": ".L",
        ".AU": ".AX",
        ".TSE": ".T",
        ".KO": ".KS",
        ".KQ": ".KQ",
        ".XETRA": ".DE",
        ".SHG": ".SS",
        ".SHE": ".SZ",
        ".SG": ".SI",
    }
    for eodhd_suffix, yahoo_suffix in suffix_map.items():
        if symbol.endswith(eodhd_suffix):
            return symbol[: -len(eodhd_suffix)] + yahoo_suffix
    return symbol


def fetch_yahoo_direct(yahoo_symbol: str, start: date, end: date, retries: int = 1) -> ProviderSeries:
    period1 = int(datetime.combine(start - timedelta(days=7), datetime.min.time(), tzinfo=UTC).timestamp())
    period2 = int(datetime.combine(end + timedelta(days=7), datetime.min.time(), tzinfo=UTC).timestamp())
    params = urllib.parse.urlencode(
        {
            "period1": str(period1),
            "period2": str(period2),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(yahoo_symbol)}?{params}"
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = json.loads(response.read())
            result = (raw.get("chart", {}).get("result") or [None])[0]
            if not result:
                return ProviderSeries("yahoo", yahoo_symbol, [], ["empty_yahoo_result"])
            timestamps = result.get("timestamp") or []
            quote = (result.get("indicators", {}).get("quote") or [{}])[0]
            adj = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose") or []
            closes = quote.get("close") or []
            prices: list[PricePoint] = []
            for index, timestamp in enumerate(timestamps):
                close = closes[index] if index < len(closes) else None
                adjusted = adj[index] if index < len(adj) else None
                if close is None or adjusted is None:
                    continue
                prices.append(
                    PricePoint(
                        day=datetime.fromtimestamp(int(timestamp), UTC).date(),
                        close=float(close),
                        adjusted_close=float(adjusted),
                        source="yahoo",
                    )
                )
            return ProviderSeries("yahoo", yahoo_symbol, sorted(prices, key=lambda item: item.day), [])
        except urllib.error.HTTPError as exc:
            return ProviderSeries("yahoo", yahoo_symbol, [], [f"http_{exc.code}"])
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt < retries:
                time.sleep(2 + attempt * 2)
                continue
            return ProviderSeries("yahoo", yahoo_symbol, [], [f"network:{exc!r}"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return ProviderSeries("yahoo", yahoo_symbol, [], [f"validation:{exc!r}"])
    return ProviderSeries("yahoo", yahoo_symbol, [], ["unknown"])


def window_sanity(prices: list[PricePoint], start: PricePoint, endpoint: PricePoint, years: int) -> tuple[bool, str, dict[str, Any]]:
    window = [point for point in prices if start.day <= point.day <= endpoint.day]
    if len(window) < MIN_PRICE_ROWS_PER_YEAR * years:
        return False, "too_few_yahoo_price_rows", {"price_rows": len(window)}
    ratios = [point.adjusted_close / point.close for point in window if point.close > 0 and point.adjusted_close > 0]
    if not ratios:
        return False, "missing_yahoo_adjustment_ratios", {}
    ratio_min = min(ratios)
    ratio_max = max(ratios)
    ratio_drift = ratio_max / ratio_min if ratio_min > 0 else math.inf
    if ratio_drift > MAX_ADJUSTMENT_RATIO_DRIFT:
        return False, "yahoo_adjustment_ratio_drift_too_large", {
            "adjustment_ratio_min": ratio_min,
            "adjustment_ratio_max": ratio_max,
            "adjustment_ratio_drift": ratio_drift,
        }
    moves = [
        window[index].adjusted_close / window[index - 1].adjusted_close
        for index in range(1, len(window))
        if window[index - 1].adjusted_close > 0 and window[index].adjusted_close > 0
    ]
    max_up = max(moves) if moves else None
    min_down = min(moves) if moves else None
    if max_up is not None and max_up > MAX_SINGLE_DAY_ADJUSTED_UP:
        return False, "yahoo_single_day_adjusted_jump_too_large", {"max_single_day_adjusted_move": max_up}
    if min_down is not None and min_down < MIN_SINGLE_DAY_ADJUSTED_DOWN:
        return False, "yahoo_single_day_adjusted_drop_too_large", {"min_single_day_adjusted_move": min_down}
    return True, "yahoo_adjustment_sanity_passed", {
        "price_rows": len(window),
        "adjustment_ratio_min": ratio_min,
        "adjustment_ratio_max": ratio_max,
        "adjustment_ratio_drift": ratio_drift,
        "max_single_day_adjusted_move": max_up,
        "min_single_day_adjusted_move": min_down,
    }


def review_horizons(row: dict[str, str], series: ProviderSeries) -> tuple[dict[str, float], dict[str, Any], list[str]]:
    pub_date = parse_date(row.get("publication_date") or "")
    if pub_date is None:
        return {}, {}, ["missing_publication_date"]
    start = first_on_or_after(series.prices, pub_date)
    if start is None:
        return {}, {}, ["missing_yahoo_start_price"]
    start_gap = (start.day - pub_date).days
    if start_gap < 0 or start_gap > MAX_START_GAP_DAYS:
        return {}, {}, ["yahoo_start_price_too_far_from_publication_date"]
    delisted_date = parse_date(row.get("fundamentals_delisted_date") or "")
    passed: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    failures: list[str] = []
    for years in HORIZON_YEARS:
        target = add_years(pub_date, years)
        if target > date.today():
            continue
        endpoint = first_on_or_after(series.prices, target)
        if endpoint is None:
            failures.append(f"{years}y:missing_yahoo_endpoint")
            continue
        endpoint_gap = (endpoint.day - target).days
        if endpoint_gap < 0 or endpoint_gap > MAX_ENDPOINT_GAP_DAYS:
            failures.append(f"{years}y:yahoo_endpoint_too_far_from_target")
            continue
        if delisted_date and endpoint.day >= delisted_date:
            failures.append(f"{years}y:endpoint_on_or_after_eodhd_delisted_date")
            continue
        multiplier = endpoint.adjusted_close / start.adjusted_close if start.adjusted_close > 0 else None
        if multiplier is None or multiplier <= MIN_RETURN or multiplier >= MAX_RETURN:
            failures.append(f"{years}y:return_outside_ordinary_band")
            continue
        ok, reason, sanity = window_sanity(series.prices, start, endpoint, years)
        if not ok:
            failures.append(f"{years}y:{reason}")
            continue
        key = f"{years}y"
        passed[key] = multiplier
        diagnostics[key] = {
            "target_date": target.isoformat(),
            "start_trade_date": start.day.isoformat(),
            "endpoint_trade_date": endpoint.day.isoformat(),
            "start_adjusted_close": start.adjusted_close,
            "endpoint_adjusted_close": endpoint.adjusted_close,
            "multiplier": multiplier,
            "start_gap_days": start_gap,
            "endpoint_gap_days": endpoint_gap,
            "sanity": sanity,
        }
    return passed, diagnostics, failures


def base_candidate_ok(row: dict[str, str]) -> tuple[bool, str]:
    if row.get("math_validation_status") != "provider_error":
        return False, "not_provider_error"
    if row.get("fundamentals_status") != "fetched":
        return False, "missing_eodhd_fundamentals_identity"
    if row.get("fundamentals_type") != "Common Stock":
        return False, "not_common_stock"
    if looks_non_common_instrument(row):
        return False, "instrument_name_or_symbol_not_common_stock"
    return True, "candidate"


def review_row(row: dict[str, str], idea: dict[str, Any] | None, cache: dict[str, ProviderSeries]) -> dict[str, Any]:
    ok, reason = base_candidate_ok(row)
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
    }
    if not ok:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": reason, "passed_horizons": {}}
    yahoo_symbol = yahoo_symbol_for_row(row, idea)
    if not yahoo_symbol:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": "missing_yahoo_symbol", "passed_horizons": {}}
    pub_date = parse_date(row.get("publication_date") or "")
    if pub_date is None:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": "missing_publication_date", "passed_horizons": {}}
    end = min(add_years(pub_date, 20), date.today())
    cache_key = f"{yahoo_symbol}|{pub_date.isoformat()}|{end.isoformat()}"
    if cache_key not in cache:
        cache[cache_key] = fetch_yahoo_direct(yahoo_symbol, pub_date, end, retries=1)
    series = cache[cache_key]
    if not series.prices:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": "yahoo_price_unavailable",
            "yahoo_symbol": yahoo_symbol,
            "yahoo_warnings": series.warnings,
            "passed_horizons": {},
        }
    passed, diagnostics, failures = review_horizons(row, series)
    if not passed:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": ";".join(failures[:5]) or "no_ordinary_horizon_passed",
            "yahoo_symbol": yahoo_symbol,
            "yahoo_rows": len(series.prices),
            "horizon_failures": failures,
            "passed_horizons": {},
        }
    return {
        **base,
        "review_status": "pass",
        "training_action": "add_to_training_ready",
        "reason": "provider_error_repaired_with_yahoo_price_and_eodhd_fundamentals_identity",
        "yahoo_symbol": yahoo_symbol,
        "yahoo_rows": len(series.prices),
        "passed_horizons": passed,
        "horizon_diagnostics": diagnostics,
        "horizon_failures": failures,
        "confidence": 0.58,
    }


def select_candidates(rows: list[dict[str, str]], training_keys: set[tuple[str, str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row_key(row) not in training_keys and row.get("math_validation_status") == "provider_error"]


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
        "yahoo_symbol",
        "yahoo_rows",
        "fundamentals_type",
        "fundamentals_is_delisted",
        "fundamentals_delisted_date",
        "is_in_delisted_cache",
        "confidence",
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
        "provider_error_yahoo_identity_status",
        "provider_error_yahoo_identity_action",
        "provider_error_yahoo_identity_reason",
        "provider_error_yahoo_identity_passed_horizons",
        "provider_error_yahoo_identity_symbol",
        "provider_error_yahoo_identity_rows",
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
                        "provider_error_yahoo_identity_status": review.get("review_status"),
                        "provider_error_yahoo_identity_action": review.get("training_action"),
                        "provider_error_yahoo_identity_reason": review.get("reason"),
                        "provider_error_yahoo_identity_passed_horizons": scalar(review.get("passed_horizons")),
                        "provider_error_yahoo_identity_symbol": review.get("yahoo_symbol") or "",
                        "provider_error_yahoo_identity_rows": scalar(review.get("yahoo_rows")),
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
                "math_validation_status": "provider_error_repaired_yahoo_price_eodhd_identity",
                "review_stage": "provider_error_yahoo_price_eodhd_identity",
                "training_readiness": "training_ready",
                "review_status": "pass",
                "reviewed_at": now,
                "agent_b_yahoo_symbol": scalar(review.get("yahoo_symbol")),
                "agent_b_yahoo_rows": scalar(review.get("yahoo_rows")),
                "agent_c_status": "pass",
                "agent_c_reason": scalar(review.get("reason")),
                "agent_c_outcome_type": "provider_error_yahoo_price_eodhd_identity",
                "source_count": "3",
                "fundamentals_name": scalar(review.get("company_name")),
                "fundamentals_type": scalar(review.get("fundamentals_type")),
                "original_validation_status": "provider_error",
                "original_review_stage": "provider_error_yahoo_identity_salvage",
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
    ideas = load_ideas()
    candidates = select_candidates(rows, training_keys)
    if args.limit:
        candidates = candidates[: args.limit]
    yahoo_cache: dict[str, ProviderSeries] = {}
    reviews = [review_row(row, ideas.get(row.get("idea_id") or ""), yahoo_cache) for row in candidates]
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
        "provider_error_candidates": len(candidates),
        "existing_training_ready_rows": len(existing_training),
        "new_training_ready_rows": added,
        "combined_training_ready_rows": len(existing_training) + added,
        "review_status_counts": dict(Counter(review.get("review_status") for review in reviews)),
        "training_action_counts": dict(Counter(review.get("training_action") for review in reviews)),
        "top_reasons": dict(Counter(str(review.get("reason")) for review in reviews).most_common(25)),
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
