#!/usr/bin/env python3
"""Manual-review validation helpers for EODHD performance labels.

The goal is not to make every row automatic. The goal is to turn manual review
into a deterministic evidence process:

- Agent A: EODHD cached label proposal.
- Agent B: independent provider reproduction, currently Yahoo chart API.
- Agent C: qualitative/security-outcome evidence for high-risk rows.
- Adversarial gate: reject provider-adjustment discontinuities, missing starts,
  missing endpoints, mismatched cross-provider returns, and unsupported
  extreme/security-outcome claims.

No API keys are stored here.
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
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from eodhd_backfill import extract_fundamentals_summary


HORIZONS = (1, 3, 5, 10, 20)
DEFAULT_CACHE_DIR = Path("eodhd_output/full_run/symbol_cache")
DEFAULT_FUNDAMENTALS_CACHE_DIR = Path("eodhd_output/full_run/fundamentals_cache")
DEFAULT_RESULTS_CSV = Path("eodhd_output/full_run/validation_results.csv")
DEFAULT_OUTPUT = Path("eodhd_output/manual_review_results.json")
DEFAULT_QUALITATIVE_EVIDENCE = Path("eodhd_output/manual_review_qualitative_evidence.json")

EXTREME_WINNER_MULTIPLIER = 15.0
SEVERE_LOSER_MULTIPLIER = 0.05
PROVIDER_RETURN_DIFF_LIMIT = 0.15
PROVIDER_ADJUSTMENT_DIFF_LIMIT = 0.25

KNOWN_BAD_PROVIDER_ADJUSTMENTS = {
    ("MCEM", "2014-01-31"): {
        "outcome_type": "bad_provider_data",
        "verdict": "reject",
        "reason": "known_bad_provider_adjustment",
        "business_explanation": "EODHD adjusted close implies an implausible 1723x return; independent Yahoo adjusted data implies about 2.97x.",
        "corporate_action_evidence": "Provider adjustment discontinuity conflicts with independent adjusted prices.",
        "sources": [
            {
                "source_id": "handoff:mcem_bad_adjustment",
                "publisher": "manual_review_handoff",
                "source_type": "analyst_note",
                "supports": "bad_provider_data",
            }
        ],
        "confidence": 0.95,
    }
}


class ReviewErrorCategory:
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    CLIENT = "client"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PricePoint:
    day: date
    close: float
    adjusted_close: float
    source: str


@dataclass(frozen=True)
class ProviderSeries:
    source: str
    symbol: str
    prices: list[PricePoint]
    warnings: list[str]


def parse_semicolon_list(value: Any) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "t", "yes", "y"}


def row_flags(row: dict[str, str]) -> set[str]:
    return set(parse_semicolon_list(row.get("failure_modes"))) | set(parse_semicolon_list(row.get("warning_modes")))


def row_multipliers(row: dict[str, str]) -> dict[str, float | None]:
    return {f"{years}y": parse_float(row.get(f"perf_{years}y")) for years in HORIZONS}


def is_extreme_multiplier(value: float | None) -> bool:
    return value is not None and (value > EXTREME_WINNER_MULTIPLIER or value < SEVERE_LOSER_MULTIPLIER)


def normalized_symbol(value: str | None) -> str:
    return (value or "").strip().upper()


def evidence_keys(row: dict[str, str]) -> list[str]:
    pub_date = row.get("publication_date")
    keys: list[str] = []
    if row.get("idea_id"):
        keys.append(f"idea:{row['idea_id']}")
    for symbol_field in ("raw_symbol", "eodhd_symbol"):
        symbol = normalized_symbol(row.get(symbol_field))
        if symbol and pub_date:
            keys.append(f"{symbol}|{pub_date}")
            if symbol.endswith(".US"):
                keys.append(f"{symbol[:-3]}|{pub_date}")
    return keys


def load_qualitative_evidence(path: Path = DEFAULT_QUALITATIVE_EVIDENCE) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("records")
        if records is None:
            records = list(payload.values())
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"Qualitative evidence must be a JSON object or list: {path}")

    evidence: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        pub_date = record.get("publication_date")
        if record.get("idea_id"):
            evidence[f"idea:{record['idea_id']}"] = record
        for symbol_field in ("raw_symbol", "eodhd_symbol"):
            symbol = normalized_symbol(record.get(symbol_field))
            if symbol and pub_date:
                evidence[f"{symbol}|{pub_date}"] = record
                if symbol.endswith(".US"):
                    evidence[f"{symbol[:-3]}|{pub_date}"] = record
    return evidence


def load_fundamentals_summary(symbol: str, cache_dir: Path = DEFAULT_FUNDAMENTALS_CACHE_DIR) -> dict[str, Any]:
    path = cache_dir / safe_symbol_filename(symbol)
    if not path.exists():
        return {"symbol": symbol, "fundamentals_status": "not_fetched"}
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "symbol": symbol,
            "fundamentals_status": "provider_error",
            "fundamentals_error": f"invalid_cached_json:{exc}",
        }
    return extract_fundamentals_summary(symbol, bundle)


def lookup_qualitative_evidence(row: dict[str, str], evidence: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for key in evidence_keys(row):
        if key in evidence:
            return evidence[key]
    return None


def known_bad_provider_adjustment(row: dict[str, str]) -> dict[str, Any] | None:
    pub_date = row.get("publication_date")
    for symbol in {normalized_symbol(row.get("raw_symbol")), normalized_symbol(row.get("eodhd_symbol")).removesuffix(".US")}:
        if (symbol, pub_date) in KNOWN_BAD_PROVIDER_ADJUSTMENTS:
            return KNOWN_BAD_PROVIDER_ADJUSTMENTS[(symbol, pub_date)]
    return None


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def add_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year + years)


def first_on_or_after(prices: list[PricePoint], target: date) -> PricePoint | None:
    for point in prices:
        if point.day >= target:
            return point
    return None


def nearest_window_values(prices: list[PricePoint], target: date, days: int = 182) -> list[float]:
    start = target - timedelta(days=days)
    end = target + timedelta(days=days)
    return [point.adjusted_close for point in prices if start <= point.day <= end and point.adjusted_close > 0]


def safe_multiplier(start: PricePoint | None, endpoint: PricePoint | None) -> float | None:
    if start is None or endpoint is None:
        return None
    if start.adjusted_close <= 0 or endpoint.adjusted_close < 0:
        return None
    return endpoint.adjusted_close / start.adjusted_close


def relative_diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    denominator = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / denominator


def safe_symbol_filename(symbol: str) -> str:
    return "".join(char if char.isalnum() or char in ".-" else "_" for char in symbol) + ".json"


def load_eodhd_series(symbol: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> ProviderSeries:
    path = cache_dir / safe_symbol_filename(symbol)
    if not path.exists():
        return ProviderSeries("eodhd", symbol, [], [f"missing_eodhd_cache:{path}"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    prices: list[PricePoint] = []
    warnings = list(payload.get("price_warnings") or [])
    for row in payload.get("prices") or []:
        if not isinstance(row, dict):
            warnings.append("ignored_non_object_eodhd_price")
            continue
        try:
            prices.append(
                PricePoint(
                    day=parse_date(str(row["date"])),
                    close=float(row["close"]),
                    adjusted_close=float(row["adjusted_close"]),
                    source="eodhd",
                )
            )
        except (KeyError, TypeError, ValueError):
            warnings.append(f"ignored_malformed_eodhd_price:{row!r}")
    return ProviderSeries("eodhd", symbol, sorted(prices, key=lambda p: p.day), warnings)


def yahoo_symbol_from_eodhd(symbol: str) -> str:
    if symbol.endswith(".US"):
        return symbol[:-3]
    suffix_map = {
        ".KO": ".KS",
        ".KQ": ".KQ",
        ".LSE": ".L",
        ".AU": ".AX",
        ".TO": ".TO",
        ".V": ".V",
        ".HK": ".HK",
        ".TSE": ".T",
        ".T": ".T",
        ".PA": ".PA",
        ".XETRA": ".DE",
        ".F": ".F",
        ".SG": ".SI",
        ".SHG": ".SS",
        ".SHE": ".SZ",
    }
    for eod_suffix, yahoo_suffix in suffix_map.items():
        if symbol.endswith(eod_suffix):
            return symbol[: -len(eod_suffix)] + yahoo_suffix
    return symbol


def fetch_yahoo_series(symbol: str, start: date, end: date, retries: int = 2) -> ProviderSeries:
    yahoo_symbol = yahoo_symbol_from_eodhd(symbol)
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
    warnings: list[str] = []
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
            for idx, ts in enumerate(timestamps):
                close = closes[idx] if idx < len(closes) else None
                adjusted = adj[idx] if idx < len(adj) else None
                if close is None or adjusted is None:
                    continue
                prices.append(
                    PricePoint(
                        day=datetime.fromtimestamp(int(ts), UTC).date(),
                        close=float(close),
                        adjusted_close=float(adjusted),
                        source="yahoo",
                    )
                )
            return ProviderSeries("yahoo", yahoo_symbol, sorted(prices, key=lambda p: p.day), warnings)
        except urllib.error.HTTPError as exc:
            category = categorize_http_error(exc.code)
            if category == ReviewErrorCategory.RATE_LIMIT and attempt < retries:
                time.sleep(2 + attempt * 3)
                continue
            return ProviderSeries("yahoo", yahoo_symbol, [], [f"{category}:http_{exc.code}"])
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt < retries:
                time.sleep(2 + attempt * 3)
                continue
            return ProviderSeries("yahoo", yahoo_symbol, [], [f"{ReviewErrorCategory.NETWORK}:{exc!r}"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return ProviderSeries("yahoo", yahoo_symbol, [], [f"{ReviewErrorCategory.VALIDATION}:{exc!r}"])
    return ProviderSeries("yahoo", yahoo_symbol, [], [ReviewErrorCategory.UNKNOWN])


def categorize_http_error(status: int) -> str:
    if status == 429:
        return ReviewErrorCategory.RATE_LIMIT
    if 400 <= status < 500:
        return ReviewErrorCategory.CLIENT
    if 500 <= status < 600:
        return ReviewErrorCategory.SERVER
    return ReviewErrorCategory.UNKNOWN


def provider_returns(series: ProviderSeries, publication_date: date) -> dict[str, dict[str, Any]]:
    start = first_on_or_after(series.prices, publication_date)
    result: dict[str, dict[str, Any]] = {}
    for years in HORIZONS:
        target = add_years(publication_date, years)
        endpoint = first_on_or_after(series.prices, target)
        median_values = nearest_window_values(series.prices, target)
        result[f"{years}y"] = {
            "target_date": target.isoformat(),
            "start_trade_date": start.day.isoformat() if start else None,
            "endpoint_trade_date": endpoint.day.isoformat() if endpoint else None,
            "start_adjusted_close": start.adjusted_close if start else None,
            "endpoint_adjusted_close": endpoint.adjusted_close if endpoint else None,
            "multiplier": safe_multiplier(start, endpoint),
            "median_52w_multiplier": statistics.median(median_values) / start.adjusted_close
            if start and start.adjusted_close > 0 and median_values
            else None,
        }
    return result


def start_adjustment_ratio(series: ProviderSeries, publication_date: date) -> float | None:
    start = first_on_or_after(series.prices, publication_date)
    if not start or start.close == 0:
        return None
    return start.adjusted_close / start.close


def horizon_has_extreme(horizon_reviews: dict[str, dict[str, Any]]) -> bool:
    for review in horizon_reviews.values():
        multiplier = (review.get("eodhd") or {}).get("multiplier")
        if is_extreme_multiplier(multiplier):
            return True
    return False


def qualitative_review_required(row: dict[str, str], horizon_reviews: dict[str, dict[str, Any]]) -> bool:
    flags = row_flags(row)
    high_risk_flags = {
        "extreme_return_requires_stronger_evidence",
        "price_history_ends_before_long_horizon",
        "symbol_in_delisted_cache",
        "reverse_split_provider_adjusted",
        "lineage_override_requires_agent_review",
        "first_price_far_after_publication",
        "invalid_start_price_for_publication_date",
        "primary_horizon_missing_price",
        "high_risk_warning_needs_review",
    }
    if known_bad_provider_adjustment(row):
        return True
    if horizon_has_extreme(horizon_reviews):
        return True
    if flags & high_risk_flags:
        return True
    if parse_bool(row.get("is_in_delisted_cache")):
        return True
    return (row.get("validation_status") or "") == "needs_manual_review"


def normalize_sources(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    sources: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            sources.append(item)
        elif item:
            sources.append({"source_id": str(item), "source_type": "unstructured"})
    return sources


def agent_c_qualitative_review(
    row: dict[str, str],
    horizon_reviews: dict[str, dict[str, Any]],
    qualitative_evidence: dict[str, dict[str, Any]],
    fundamentals_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    known_failure = known_bad_provider_adjustment(row)
    required = qualitative_review_required(row, horizon_reviews)
    evidence = known_failure or lookup_qualitative_evidence(row, qualitative_evidence)

    if not required and evidence is None:
        return {
            "source": "qualitative_evidence",
            "reviewer_status": "not_required",
            "outcome_type": "ordinary_public_company",
            "reason": "no_high_risk_manual_review_flags",
            "confidence": None,
            "sources": [],
            "fundamentals_summary": compact_fundamentals_summary(fundamentals_summary),
        }

    if evidence is None:
        fundamentals_status = (fundamentals_summary or {}).get("fundamentals_status")
        return {
            "source": "qualitative_evidence",
            "reviewer_status": "manual_review",
            "outcome_type": "unknown",
            "reason": "fundamentals_available_needs_qualitative_synthesis"
            if fundamentals_status == "fetched"
            else "missing_qualitative_evidence",
            "required_checks": [
                "security_identity",
                "corporate_actions",
                "price_sanity",
                "business_reality",
                "training_inclusion",
            ],
            "confidence": None,
            "sources": [],
            "fundamentals_summary": compact_fundamentals_summary(fundamentals_summary),
        }

    verdict = str(evidence.get("verdict") or evidence.get("reviewer_status") or "manual_review")
    outcome_type = str(evidence.get("outcome_type") or "unknown")
    confidence = parse_float(evidence.get("confidence"))
    sources = normalize_sources(evidence.get("sources"))
    if verdict == "reject":
        reviewer_status = "reject"
        reason = str(evidence.get("reason") or "qualitative_rejected")
    elif (
        verdict == "pass"
        and outcome_type != "unknown"
        and confidence is not None
        and confidence >= 0.7
        and sources
    ):
        reviewer_status = "pass"
        reason = str(evidence.get("reason") or "qualitative_evidence_supports_return")
    else:
        reviewer_status = "manual_review"
        reason = "qualitative_evidence_incomplete"

    return {
        "source": "qualitative_evidence",
        "reviewer_status": reviewer_status,
        "outcome_type": outcome_type,
        "reason": reason,
        "business_explanation": evidence.get("business_explanation"),
        "revenue_growth_evidence": evidence.get("revenue_growth_evidence"),
        "profitability_evidence": evidence.get("profitability_evidence"),
        "market_cap_evidence": evidence.get("market_cap_evidence"),
        "liquidity_evidence": evidence.get("liquidity_evidence"),
        "corporate_action_evidence": evidence.get("corporate_action_evidence"),
        "confidence": confidence,
        "sources": sources,
        "reviewed_at": evidence.get("reviewed_at"),
        "reviewer": evidence.get("reviewer"),
        "fundamentals_summary": compact_fundamentals_summary(fundamentals_summary),
    }


def compact_fundamentals_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {"fundamentals_status": "not_fetched"}
    keys = [
        "symbol",
        "fundamentals_status",
        "fundamentals_type",
        "fundamentals_name",
        "fundamentals_exchange",
        "fundamentals_is_delisted",
        "fundamentals_delisted_date",
        "fundamentals_sector",
        "fundamentals_industry",
        "fundamentals_market_cap",
        "fundamentals_revenue_ttm",
        "fundamentals_ebitda",
        "fundamentals_profit_margin",
        "fundamentals_return_on_equity_ttm",
        "fundamentals_latest_yearly_income_date",
        "fundamentals_latest_quarterly_income_date",
        "fundamentals_yearly_revenue_first",
        "fundamentals_yearly_revenue_last",
        "fundamentals_yearly_net_income_first",
        "fundamentals_yearly_net_income_last",
        "fundamentals_has_financials",
        "fundamentals_error",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


def risk_priority(row: dict[str, str]) -> tuple[int, float, str, str]:
    flags = row_flags(row)
    multipliers = [value for value in row_multipliers(row).values() if value is not None]
    raw_symbol = normalized_symbol(row.get("raw_symbol"))
    pub_date = row.get("publication_date") or ""
    staged_priority_raw = row.get("manual_review_priority")
    try:
        staged_priority = int(staged_priority_raw or "")
    except ValueError:
        staged_priority = 0
    if staged_priority_raw not in {None, ""} and staged_priority:
        target_multiplier = parse_float(row.get("review_target_multiplier")) or 0
        sort_multiplier = -target_multiplier if target_multiplier > EXTREME_WINNER_MULTIPLIER else target_multiplier
        return (staged_priority, sort_multiplier, raw_symbol, pub_date)
    if known_bad_provider_adjustment(row):
        return (0, 0, raw_symbol, pub_date)
    winners = [value for value in multipliers if value > EXTREME_WINNER_MULTIPLIER]
    if winners:
        return (1, -max(winners), raw_symbol, pub_date)
    losers = [value for value in multipliers if value < SEVERE_LOSER_MULTIPLIER]
    if losers:
        return (2, min(losers), raw_symbol, pub_date)
    if (
        parse_bool(row.get("is_in_delisted_cache"))
        or "symbol_in_delisted_cache" in flags
        or "price_history_ends_before_long_horizon" in flags
        or "primary_horizon_missing_price" in flags
    ):
        return (3, 0, raw_symbol, pub_date)
    if "reverse_split_provider_adjusted" in flags:
        return (4, 0, raw_symbol, pub_date)
    if "lineage_override_requires_agent_review" in flags or raw_symbol != normalized_symbol(row.get("eodhd_symbol")).removesuffix(".US"):
        return (5, 0, raw_symbol, pub_date)
    if "provider_return_conflict" in flags or "provider_adjustment_factor_conflict" in flags:
        return (6, 0, raw_symbol, pub_date)
    if (row.get("validation_status") or "") == "needs_manual_review":
        return (7, 0, raw_symbol, pub_date)
    return (100, 0, raw_symbol, pub_date)


def review_row(
    row: dict[str, str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
    fetch_yahoo: bool = True,
    qualitative_evidence: dict[str, dict[str, Any]] | None = None,
    fundamentals_cache_dir: Path = DEFAULT_FUNDAMENTALS_CACHE_DIR,
) -> dict[str, Any]:
    pub_date = parse_date(row["publication_date"])
    symbol = row["eodhd_symbol"]
    eodhd = load_eodhd_series(symbol, cache_dir)
    max_end = add_years(pub_date, 20)
    yahoo = fetch_yahoo_series(symbol, pub_date, min(max_end, date.today())) if fetch_yahoo else ProviderSeries("yahoo", yahoo_symbol_from_eodhd(symbol), [], ["skipped"])

    eodhd_returns = provider_returns(eodhd, pub_date)
    yahoo_returns = provider_returns(yahoo, pub_date) if yahoo.prices else {}
    horizon_reviews: dict[str, dict[str, Any]] = {}
    row_failures: list[str] = []
    row_warnings: list[str] = []

    eod_start_ratio = start_adjustment_ratio(eodhd, pub_date)
    yahoo_start_ratio = start_adjustment_ratio(yahoo, pub_date) if yahoo.prices else None
    adjustment_factor_conflict = False
    if eod_start_ratio is not None and yahoo_start_ratio is not None:
        ratio_diff = relative_diff(eod_start_ratio, yahoo_start_ratio)
        if ratio_diff is not None and ratio_diff > PROVIDER_ADJUSTMENT_DIFF_LIMIT:
            adjustment_factor_conflict = True
            row_warnings.append("provider_adjustment_factor_conflict")

    for years in HORIZONS:
        key = f"{years}y"
        e = eodhd_returns.get(key) or {}
        y = yahoo_returns.get(key) or {}
        diff = relative_diff(e.get("multiplier"), y.get("multiplier"))
        target_date = parse_date(str(e["target_date"])) if e.get("target_date") else add_years(pub_date, years)
        matured = target_date <= date.today()
        verdict = "unresolved"
        reasons: list[str] = []
        if not matured:
            verdict = "not_mature_yet"
            reasons.append("horizon_not_mature_yet")
        elif e.get("multiplier") is None:
            verdict = "fail"
            reasons.append("missing_eodhd_multiplier")
        elif fetch_yahoo and not yahoo.prices:
            verdict = "manual_review"
            reasons.append("yahoo_cross_check_unavailable")
        elif yahoo.prices and y.get("multiplier") is None:
            verdict = "manual_review"
            reasons.append("missing_yahoo_multiplier")
        elif diff is not None and diff > PROVIDER_RETURN_DIFF_LIMIT:
            verdict = "fail"
            reasons.append("provider_return_conflict")
        elif e.get("multiplier") is not None:
            verdict = "pass"
            if is_extreme_multiplier(e.get("multiplier")):
                reasons.append("extreme_price_reproduced")
        horizon_reviews[key] = {
            "verdict": verdict,
            "reasons": reasons,
            "eodhd": e,
            "yahoo": y if y else None,
            "relative_diff": diff,
            "matured": matured,
        }

    if adjustment_factor_conflict:
        conflicted_primary_returns = [
            horizon_reviews[f"{years}y"]["verdict"] == "fail" for years in (1, 3, 5)
        ]
        if any(conflicted_primary_returns):
            row_failures.append("provider_adjustment_factor_conflict_with_return_mismatch")

    if eodhd.warnings:
        row_warnings.extend(f"eodhd:{warning}" for warning in eodhd.warnings)
    if yahoo.warnings:
        row_warnings.extend(f"yahoo:{warning}" for warning in yahoo.warnings)

    fundamentals_summary = load_fundamentals_summary(symbol, fundamentals_cache_dir)
    agent_c = agent_c_qualitative_review(row, horizon_reviews, qualitative_evidence or {}, fundamentals_summary)
    if agent_c["reviewer_status"] == "reject":
        row_failures.append(str(agent_c["reason"]))

    for review in horizon_reviews.values():
        if review["verdict"] != "pass":
            continue
        if is_extreme_multiplier((review.get("eodhd") or {}).get("multiplier")):
            if agent_c["reviewer_status"] == "pass":
                review["reasons"].append("qualitative_evidence_supports_extreme_return")
            elif agent_c["reviewer_status"] == "reject":
                review["verdict"] = "fail"
                review["reasons"].append("qualitative_evidence_rejects_return")
            else:
                review["verdict"] = "manual_review"
                review["reasons"].append("qualitative_evidence_required_for_extreme_return")

    primary_verdicts = [
        horizon_reviews[f"{years}y"]["verdict"]
        for years in (1, 3, 5)
        if horizon_reviews[f"{years}y"]["matured"]
    ]
    if any(verdict == "fail" for verdict in primary_verdicts):
        review_status = "reject"
    elif primary_verdicts and all(verdict == "pass" for verdict in primary_verdicts) and agent_c["reviewer_status"] in {"pass", "not_required"}:
        review_status = "pass"
    else:
        review_status = "manual_review"

    if row_failures:
        review_status = "reject"

    return {
        "idea_id": row.get("idea_id"),
        "raw_symbol": row.get("raw_symbol"),
        "eodhd_symbol": symbol,
        "publication_date": row.get("publication_date"),
        "review_status": review_status,
        "row_failures": row_failures,
        "row_warnings": row_warnings,
        "agent_a_eodhd": {
            "source": "eodhd_cache",
            "input_validation_status": row.get("validation_status"),
            "input_math_validation_status": row.get("math_validation_status"),
            "input_review_stage": row.get("review_stage"),
            "input_training_readiness": row.get("training_readiness"),
            "input_manual_review_reason": row.get("manual_review_reason"),
            "input_label_quality": row.get("label_quality"),
            "input_failure_modes": parse_semicolon_list(row.get("failure_modes")),
            "input_warning_modes": parse_semicolon_list(row.get("warning_modes")),
            "price_rows": len(eodhd.prices),
            "first_price_date": eodhd.prices[0].day.isoformat() if eodhd.prices else None,
            "last_price_date": eodhd.prices[-1].day.isoformat() if eodhd.prices else None,
            "start_adjustment_ratio": eod_start_ratio,
            "proposed_returns": row_multipliers(row),
        },
        "agent_b_yahoo": {
            "source": "yahoo_chart",
            "symbol": yahoo.symbol,
            "price_rows": len(yahoo.prices),
            "first_price_date": yahoo.prices[0].day.isoformat() if yahoo.prices else None,
            "last_price_date": yahoo.prices[-1].day.isoformat() if yahoo.prices else None,
            "start_adjustment_ratio": yahoo_start_ratio,
        },
        "agent_c_qualitative": agent_c,
        "agent_c_fundamentals": compact_fundamentals_summary(fundamentals_summary),
        "horizon_reviews": horizon_reviews,
    }


def load_result_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_rows(rows: list[dict[str, str]], symbols: list[str] | None, limit: int | None) -> list[dict[str, str]]:
    if symbols:
        wanted = {symbol.upper() for symbol in symbols}
        rows = [row for row in rows if (row.get("raw_symbol") or "").upper() in wanted]
    rows = [row for row in rows if row.get("eodhd_symbol") and row.get("publication_date")]
    rows = sorted(rows, key=risk_priority)
    return rows[:limit] if limit else rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--qualitative-evidence", default=str(DEFAULT_QUALITATIVE_EVIDENCE))
    parser.add_argument("--fundamentals-cache-dir", default=str(DEFAULT_FUNDAMENTALS_CACHE_DIR))
    parser.add_argument("--no-yahoo", action="store_true")
    args = parser.parse_args()

    rows = select_rows(load_result_rows(Path(args.results_csv)), args.symbols, args.limit)
    qualitative_evidence = load_qualitative_evidence(Path(args.qualitative_evidence))
    reviews = [
        review_row(
            row,
            Path(args.cache_dir),
            fetch_yahoo=not args.no_yahoo,
            qualitative_evidence=qualitative_evidence,
            fundamentals_cache_dir=Path(args.fundamentals_cache_dir),
        )
        for row in rows
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reviews, indent=2, sort_keys=True), encoding="utf-8")
    counts: dict[str, int] = {}
    for review in reviews:
        counts[review["review_status"]] = counts.get(review["review_status"], 0) + 1
    print(json.dumps({"reviewed": len(reviews), "counts": counts, "output": str(out.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
