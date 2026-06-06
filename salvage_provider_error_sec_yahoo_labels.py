#!/usr/bin/env python3
"""Salvage low-risk provider-error rows using SEC identity plus Yahoo prices.

This pass is deliberately narrow. It is only for active US common-equity cases
where EODHD could not fetch data, but:

- the original idea has a plain Yahoo/US ticker
- the SEC current ticker file matches the ticker and company name
- Yahoo chart metadata says the instrument is an equity
- Yahoo adjusted prices pass the same ordinary-return/date/adjustment sanity
  checks used elsewhere

It does not promote delisted outcomes, severe losers, extreme winners, funds,
warrants, units, preferreds, bonds, or weak identity matches.

No API keys are read or stored.
"""

from __future__ import annotations

import csv
import difflib
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from manual_review_validation import PricePoint, ProviderSeries, add_years, first_on_or_after
from salvage_partial_horizon_labels import (
    MAX_ADJUSTMENT_RATIO_DRIFT,
    MAX_ENDPOINT_GAP_DAYS,
    MAX_SINGLE_DAY_ADJUSTED_UP,
    MAX_START_GAP_DAYS,
    MIN_SINGLE_DAY_ADJUSTED_DOWN,
    looks_non_common_instrument,
    parse_float,
    scalar,
)


BASE_DIR = Path("eodhd_output/full_run")
VALIDATION_IN = BASE_DIR / "validation_results_with_active_ordinary_salvage.csv"
TRAINING_READY_IN = BASE_DIR / "training_ready_after_active_ordinary_salvage.csv"
IDEAS_JSON = Path("eodhd_output/all_ideas.json")
SEC_TICKERS_JSON = Path("eodhd_output/sec_cache/company_tickers_exchange.json")
YAHOO_CACHE = BASE_DIR / "yahoo_sec_price_cache"

VALIDATION_OUT = BASE_DIR / "validation_results_with_sec_yahoo_salvage.csv"
TRAINING_READY_OUT = BASE_DIR / "training_ready_after_sec_yahoo_salvage.csv"
ROW_REVIEWS_CSV = BASE_DIR / "sec_yahoo_salvage_reviews.csv"
SUMMARY_JSON = BASE_DIR / "sec_yahoo_salvage_summary.json"

HORIZON_YEARS = (1, 3, 5, 10, 20)
MIN_RETURN = 0.05
MAX_RETURN = 10.0
MIN_PRICE_ROWS_PER_YEAR = 80
MIN_SEC_NAME_SCORE = 0.86

BAD_NAME_MARKERS = (
    " warrant",
    " warrants",
    " right",
    " rights",
    " unit",
    " units",
    " note",
    " notes",
    " bond",
    " debenture",
    " preferred",
    " pfd",
    " preference",
    " fund",
    " etf",
    " trust",
    " lp",
    " l.p.",
)
BAD_SYMBOL_PATTERNS = re.compile(r"(\.|\-|/)?(WS|WT|W|U|UN|RT|PRA|PRB|PRC|PRD|PFD)$", re.I)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("idea_id") or "", row.get("eodhd_symbol") or "", row.get("publication_date") or "")


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def normalize_name(value: str | None) -> str:
    text = (value or "").lower()
    text = text.replace("&", " and ")
    text = re.sub(
        r"\b(inc|corp|corporation|co|company|ltd|limited|plc|sa|ag|nv|se|ab|holdings?|group|the|class|cl|common|stock|com|new)\b",
        " ",
        text,
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def name_score(left: str | None, right: str | None) -> float:
    a = normalize_name(left)
    b = normalize_name(right)
    if not a or not b:
        return 0.0
    sequence = difflib.SequenceMatcher(None, a, b).ratio()
    left_tokens = set(a.split())
    right_tokens = set(b.split())
    jaccard = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    return max(sequence, jaccard)


def load_ideas() -> dict[str, dict[str, Any]]:
    payload = json.loads(IDEAS_JSON.read_text(encoding="utf-8"))
    return {str(row.get("idea_id") or ""): row for row in payload if isinstance(row, dict)}


def load_sec_tickers() -> dict[str, dict[str, Any]]:
    payload = json.loads(SEC_TICKERS_JSON.read_text(encoding="utf-8"))
    fields = payload.get("fields") or []
    data = payload.get("data") or []
    out: dict[str, dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, list) or len(item) != len(fields):
            continue
        record = dict(zip(fields, item, strict=False))
        ticker = str(record.get("ticker") or "").upper().strip()
        if ticker:
            out[ticker] = record
    return out


def safe_cache_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in ".-" else "_" for char in value)[:80]


def yahoo_symbol_from_idea(idea: dict[str, Any]) -> str:
    return str(idea.get("yahoo_symbol") or "").strip().upper()


def raw_code(row: dict[str, str]) -> str:
    raw = (row.get("raw_symbol") or row.get("eodhd_symbol") or "").upper().strip()
    raw = raw.replace(".", " ").replace("-", " ")
    return raw.split()[0] if raw.split() else ""


def plain_us_yahoo_symbol(symbol: str) -> bool:
    if not symbol:
        return False
    if "." in symbol:
        return False
    if len(symbol) > 6:
        return False
    if BAD_SYMBOL_PATTERNS.search(symbol):
        return False
    return bool(re.fullmatch(r"[A-Z][A-Z0-9-]{0,5}", symbol))


def bad_instrument_name(name: str) -> bool:
    lowered = f" {name.lower()} "
    return any(marker in lowered for marker in BAD_NAME_MARKERS)


def fetch_yahoo_chart(yahoo_symbol: str, start: date, end: date, retries: int = 1) -> tuple[ProviderSeries, dict[str, Any]]:
    YAHOO_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = YAHOO_CACHE / f"{safe_cache_token(yahoo_symbol)}_{start.isoformat()}_{end.isoformat()}.json"
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw = None
        if isinstance(raw, dict):
            return parse_yahoo_chart(yahoo_symbol, raw)

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
            cache_path.write_text(json.dumps(raw), encoding="utf-8")
            time.sleep(0.03)
            return parse_yahoo_chart(yahoo_symbol, raw)
        except urllib.error.HTTPError as exc:
            return ProviderSeries("yahoo", yahoo_symbol, [], [f"http_{exc.code}"]), {}
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt < retries:
                time.sleep(2 + attempt * 2)
                continue
            return ProviderSeries("yahoo", yahoo_symbol, [], [f"network:{exc!r}"]), {}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return ProviderSeries("yahoo", yahoo_symbol, [], [f"validation:{exc!r}"]), {}
    return ProviderSeries("yahoo", yahoo_symbol, [], ["unknown"]), {}


def parse_yahoo_chart(yahoo_symbol: str, raw: dict[str, Any]) -> tuple[ProviderSeries, dict[str, Any]]:
    result = (raw.get("chart", {}).get("result") or [None])[0]
    if not result:
        return ProviderSeries("yahoo", yahoo_symbol, [], ["empty_yahoo_result"]), {}
    meta = result.get("meta") or {}
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
    return ProviderSeries("yahoo", yahoo_symbol, sorted(prices, key=lambda item: item.day), []), meta


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


def identity_check(row: dict[str, str], idea: dict[str, Any], sec_by_ticker: dict[str, dict[str, Any]]) -> tuple[bool, str, dict[str, Any]]:
    yahoo_symbol = yahoo_symbol_from_idea(idea)
    company_name = str(idea.get("company_name") or row.get("raw_symbol") or "").strip()
    if not plain_us_yahoo_symbol(yahoo_symbol):
        return False, "not_plain_us_yahoo_symbol", {}
    if raw_code(row) and raw_code(row) != yahoo_symbol:
        return False, "raw_symbol_does_not_match_yahoo_symbol", {}
    if bad_instrument_name(company_name):
        return False, "source_name_non_common_instrument_marker", {}
    sec = sec_by_ticker.get(yahoo_symbol)
    if not sec:
        return False, "missing_sec_current_ticker_identity", {}
    score = name_score(company_name, str(sec.get("name") or ""))
    if score < MIN_SEC_NAME_SCORE:
        return False, "sec_name_match_too_weak", {"sec_name_score": score, "sec_name": sec.get("name")}
    identity_row = {
        **row,
        "raw_symbol": yahoo_symbol,
        "eodhd_symbol": f"{yahoo_symbol}.US",
        "fundamentals_type": "Common Stock",
        "fundamentals_name": str(sec.get("name") or company_name),
        "company_name": company_name,
    }
    if looks_non_common_instrument(identity_row):
        return False, "instrument_name_or_symbol_not_common_stock", {}
    return True, "sec_current_ticker_identity_matched", {
        "yahoo_symbol": yahoo_symbol,
        "company_name": company_name,
        "sec_name": sec.get("name"),
        "sec_cik": sec.get("cik"),
        "sec_exchange": sec.get("exchange"),
        "sec_name_score": score,
        "identity_row": identity_row,
    }


def review_row(
    row: dict[str, str],
    idea: dict[str, Any],
    sec_by_ticker: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base = {
        "idea_id": row.get("idea_id"),
        "raw_symbol": row.get("raw_symbol"),
        "eodhd_symbol": row.get("eodhd_symbol"),
        "publication_date": row.get("publication_date"),
        "company_name": idea.get("company_name") or row.get("fundamentals_name"),
    }
    ok, reason, identity = identity_check(row, idea, sec_by_ticker)
    if not ok:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": reason, **identity, "passed_horizons": {}}
    pub_date = parse_date(row.get("publication_date"))
    if pub_date is None:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": "missing_publication_date", "passed_horizons": {}}
    yahoo_symbol = identity["yahoo_symbol"]
    end = min(add_years(pub_date, 20), date.today())
    series, meta = fetch_yahoo_chart(yahoo_symbol, pub_date, end, retries=1)
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
    if str(meta.get("instrumentType") or "").upper() not in {"EQUITY", ""}:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": "yahoo_instrument_not_equity",
            "yahoo_symbol": yahoo_symbol,
            "yahoo_instrument_type": meta.get("instrumentType"),
            "passed_horizons": {},
        }
    passed, diagnostics, failures = review_horizons(identity["identity_row"], series)
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
        "reason": "sec_identity_yahoo_price_sanity_passed",
        "yahoo_symbol": yahoo_symbol,
        "yahoo_rows": len(series.prices),
        "sec_name": identity.get("sec_name"),
        "sec_cik": identity.get("sec_cik"),
        "sec_exchange": identity.get("sec_exchange"),
        "sec_name_score": identity.get("sec_name_score"),
        "yahoo_exchange": meta.get("exchangeName") or meta.get("fullExchangeName"),
        "passed_horizons": passed,
        "horizon_diagnostics": diagnostics,
        "horizon_failures": failures,
        "confidence": 0.58,
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
        "yahoo_symbol",
        "yahoo_rows",
        "sec_name",
        "sec_cik",
        "sec_exchange",
        "sec_name_score",
        "confidence",
    ]
    with ROW_REVIEWS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for review in reviews:
            writer.writerow({field: scalar(review.get(field)) for field in fieldnames})


def write_validation(rows: list[dict[str, str]], reviews_by_key: dict[tuple[str, str, str], dict[str, Any]]) -> None:
    extra = [
        "sec_yahoo_salvage_status",
        "sec_yahoo_salvage_action",
        "sec_yahoo_salvage_reason",
        "sec_yahoo_salvage_passed_horizons",
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
                        "sec_yahoo_salvage_status": review.get("review_status"),
                        "sec_yahoo_salvage_action": review.get("training_action"),
                        "sec_yahoo_salvage_reason": review.get("reason"),
                        "sec_yahoo_salvage_passed_horizons": scalar(review.get("passed_horizons")),
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
                "math_validation_status": "sec_yahoo_identity_verified",
                "review_stage": "sec_identity_yahoo_price_sanity",
                "training_readiness": "training_ready",
                "review_status": "pass",
                "reviewed_at": now,
                "agent_b_yahoo_symbol": scalar(review.get("yahoo_symbol")),
                "agent_b_yahoo_rows": scalar(review.get("yahoo_rows")),
                "agent_c_status": "pass",
                "agent_c_reason": scalar(review.get("reason")),
                "agent_c_outcome_type": "sec_identity_yahoo_price_sanity",
                "source_count": "3",
                "fundamentals_name": scalar(review.get("sec_name") or review.get("company_name")),
                "fundamentals_type": "Common Stock",
                "original_validation_status": "provider_error_or_math_incomplete",
                "original_review_stage": "sec_identity_yahoo_price_sanity",
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
    ideas = load_ideas()
    sec_by_ticker = load_sec_tickers()
    candidates = [
        row
        for row in rows
        if row.get("math_validation_status") in {"provider_error", "math_incomplete"}
        and row_key(row) not in training_keys
    ]
    reviews = [review_row(row, ideas.get(row.get("idea_id") or "") or {}, sec_by_ticker) for row in candidates]
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
        "provider_error_or_math_incomplete_candidates": len(candidates),
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
