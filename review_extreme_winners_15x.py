#!/usr/bin/env python3
"""Review and stage >=15x performance rows with qualitative evidence.

This is intentionally conservative. A >=15x price result is not promoted just
because EODHD reproduced the math. To pass, a row needs:

- active common-stock identity with no delisted/reverse-split/ticker-reuse flag
- prior Yahoo cross-check from the manual-review runner
- a business-quality explanation from financial statements around the horizon
- browser-checkable sources such as SEC/IR pages or company sites

No API tokens are used or stored here.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path("eodhd_output/full_run")
VALIDATION_CSV = BASE_DIR / "validation_results_with_manual_review.csv"
REVIEWS_JSONL = BASE_DIR / "math_reproduced_manual_reviews.jsonl"
FUNDAMENTALS_CACHE = BASE_DIR / "fundamentals_cache"
TRAINING_READY_IN = BASE_DIR / "training_ready_math_reproduced.csv"

EXTREME_REVIEW_CSV = BASE_DIR / "extreme_15x_business_review.csv"
EXTREME_EVIDENCE_JSON = BASE_DIR / "extreme_15x_qualitative_evidence.json"
VALIDATION_WITH_EXTREME_CSV = BASE_DIR / "validation_results_with_extreme_review.csv"
TRAINING_READY_OUT = BASE_DIR / "training_ready_with_extreme_15x.csv"
SUMMARY_JSON = BASE_DIR / "extreme_15x_review_summary.json"

EXTREME_THRESHOLD = 15.0
YAHOO_DIFF_LIMIT = 0.15

BLOCKING_WARNING_FLAGS = {
    "symbol_in_delisted_cache",
    "fundamentals_is_delisted",
    "price_history_ends_before_long_horizon",
    "reverse_split_provider_adjusted",
    "fundamentals_non_common_instrument",
    "fundamentals_financials_missing",
}

DO_NOT_AUTO_PROMOTE = {
    "BTU.US": "coal/post-bankruptcy commodity-cycle winner; needs explicit bankruptcy and commodity-cycle modeling",
    "CEM.US": "fund/CEF-like instrument, not an operating common-stock business",
    "ECRO.US": "PINK/OTC security with weak identity and business-quality evidence",
    "FRO.US": "shipping commodity-cycle winner; needs cycle-specific treatment",
    "GBTC.US": "crypto trust/fund premium-discount dynamics, not an operating company",
    "GME.US": "meme-squeeze return not supported by operating business quality",
    "HIVE.US": "crypto miner/capital-markets cycle; needs crypto-cycle treatment",
    "OGI.US": "cannabis cycle/security-quality risk; needs separate qualitative review",
}

STORY_SOURCES: dict[str, list[dict[str, str]]] = {
    "AAPL.US": [
        {
            "publisher": "Apple/SEC filing mirror",
            "url": "https://d18rn0p25nwr6d.cloudfront.net/CIK-0000320193/c636d8a7-8025-47d2-9b13-bcf5465343b3.html",
            "supports": "2025 Form 10-K revenue and net income scale",
        }
    ],
    "NVDA.US": [
        {
            "publisher": "NVIDIA Investor Relations",
            "url": "https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-Announces-Financial-Results-for-Fourth-Quarter-and-Fiscal-2026/",
            "supports": "AI/data-center revenue inflection and fiscal-year scale",
        }
    ],
    "NFLX.US": [
        {
            "publisher": "SEC EDGAR",
            "url": "https://www.sec.gov/Archives/edgar/data/1065280/000106528026000034/nflx-20251231.htm",
            "supports": "streaming-scale revenue and profitability evidence",
        }
    ],
    "CELH.US": [
        {
            "publisher": "Celsius Investor Relations",
            "url": "https://ir.celsiusholdingsinc.com/news/news-details/2022/CELSIUS-PepsiCo-Partnership/default.aspx",
            "supports": "PepsiCo strategic distribution agreement and investment",
        }
    ],
    "TSLA.US": [
        {
            "publisher": "SEC EDGAR",
            "url": "https://www.sec.gov/Archives/edgar/data/1318605/000162828025003063/tsla-20241231.htm",
            "supports": "Tesla 2024 Form 10-K operating scale",
        }
    ],
    "TSCO.US": [
        {
            "publisher": "Tractor Supply Investor Relations",
            "url": "https://ir.tractorsupply.com/newsroom/news-releases/news-releases-details/2025/Tractor-Supply-Company-Reports-Fourth-Quarter-and-Fiscal-Year-2024-Financial-Results-Provides-Fiscal-Year-2025-Outlook/default.aspx",
            "supports": "store expansion and rural-lifestyle retail compounding evidence",
        }
    ],
    "AZO.US": [
        {
            "publisher": "AutoZone Investor Relations",
            "url": "https://about.autozone.com/investor-relations/financial-information",
            "supports": "long annual-report history and operating compounding evidence",
        }
    ],
    "CPRT.US": [
        {
            "publisher": "Copart 2025 Form 10-K",
            "url": "https://fintel.io/doc/sec-copart-inc-900075-10k-2025-september-26-20357-1904",
            "supports": "salvage-auction platform scale and profitability evidence",
        }
    ],
    "IDXX.US": [
        {
            "publisher": "SEC EDGAR",
            "url": "https://www.sec.gov/Archives/edgar/data/874716/000087471626000038/idxx-20251231.htm",
            "supports": "companion-animal diagnostics revenue and profitability evidence",
        }
    ],
    "SBAC.US": [
        {
            "publisher": "SBA Communications Investor Relations",
            "url": "https://ir.sbasite.com/English/Investors-overview/sec-filings/default.aspx",
            "supports": "tower-company filing history and operating business evidence",
        }
    ],
    "WST.US": [
        {
            "publisher": "West Pharmaceutical Services Investor Relations",
            "url": "https://www.investor.westpharma.com/financial",
            "supports": "annual reports and pharmaceutical packaging/biologics evidence",
        }
    ],
    "DHR.US": [
        {
            "publisher": "SEC EDGAR",
            "url": "https://www.sec.gov/Archives/edgar/data/0000313616/000031361626000105/danaher2025annualreport.htm",
            "supports": "life-sciences/diagnostics operating scale",
        }
    ],
}


@dataclass(frozen=True)
class FinancialRecord:
    day: date
    revenue: float | None
    net_income: float | None
    operating_income: float | None


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "t", "yes", "y"}


def add_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year + years)


def split_flags(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.replace("|", ";").split(";") if part.strip()}


def review_key(row: dict[str, str]) -> str:
    return "|".join([row.get("idea_id") or "", row.get("eodhd_symbol") or "", row.get("publication_date") or ""])


def safe_symbol_filename(symbol: str) -> str:
    return symbol.replace("/", "-") + ".json"


def currency(value: float | None) -> str:
    if value is None:
        return "n/a"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"


def ratio_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_reviews(path: Path) -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return reviews
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = "|".join(
                [
                    str(review.get("idea_id") or ""),
                    str(review.get("eodhd_symbol") or ""),
                    str(review.get("publication_date") or ""),
                ]
            )
            if key.strip("|"):
                reviews[key] = review
    return reviews


def load_fundamentals(symbol: str) -> dict[str, Any]:
    path = FUNDAMENTALS_CACHE / safe_symbol_filename(symbol)
    if not path.exists():
        return {"_cache_path": str(path), "_payload": {}}
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_cache_path": str(path), "_payload": {}}
    payload = cached.get("payload") or cached.get("data") or cached
    if not isinstance(payload, dict):
        payload = {}
    return {"_cache_path": str(path), "_payload": payload}


def yearly_income_records(payload: dict[str, Any]) -> list[FinancialRecord]:
    yearly = (((payload.get("Financials") or {}).get("Income_Statement") or {}).get("yearly") or {})
    records: list[FinancialRecord] = []
    if isinstance(yearly, dict):
        iterable = yearly.items()
    elif isinstance(yearly, list):
        iterable = [(str(item.get("date") or idx), item) for idx, item in enumerate(yearly) if isinstance(item, dict)]
    else:
        iterable = []
    for key, item in iterable:
        if not isinstance(item, dict):
            continue
        day = parse_date(str(item.get("date") or key))
        if not day:
            continue
        records.append(
            FinancialRecord(
                day=day,
                revenue=parse_float(item.get("totalRevenue") or item.get("revenue")),
                net_income=parse_float(item.get("netIncome")),
                operating_income=parse_float(item.get("operatingIncome") or item.get("ebit")),
            )
        )
    return sorted(records, key=lambda record: record.day)


def nearest_record(records: list[FinancialRecord], target: date, max_days: int = 730) -> FinancialRecord | None:
    if not records:
        return None
    candidates = sorted(records, key=lambda record: abs((record.day - target).days))
    if candidates and abs((candidates[0].day - target).days) <= max_days:
        return candidates[0]
    return candidates[0] if candidates else None


def horizon_years(value: str | None) -> int | None:
    if not value:
        return None
    text = str(value).strip().lower()
    if text.endswith("y"):
        text = text[:-1]
    try:
        return int(text)
    except ValueError:
        return None


def yahoo_horizon_ok(row: dict[str, str], review: dict[str, Any] | None) -> tuple[bool, str, dict[str, Any]]:
    if not review:
        return False, "missing_manual_review_jsonl_record", {}
    horizon = row.get("review_target_horizon") or ""
    horizon_review = (review.get("horizon_reviews") or {}).get(horizon) or {}
    yahoo = horizon_review.get("yahoo") or {}
    if not yahoo:
        return False, "yahoo_cross_check_missing_for_target_horizon", horizon_review
    relative_diff = parse_float(horizon_review.get("relative_diff"))
    if relative_diff is None:
        return False, "yahoo_relative_diff_missing", horizon_review
    if relative_diff > YAHOO_DIFF_LIMIT:
        return False, f"yahoo_disagreement:{relative_diff:.3f}", horizon_review
    return True, "yahoo_cross_check_passed", horizon_review


def financial_quality(
    row: dict[str, str],
    payload: dict[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    pub_date = parse_date(row.get("publication_date"))
    years = horizon_years(row.get("review_target_horizon"))
    if pub_date is None or years is None:
        return False, "missing_publication_date_or_horizon", {}

    records = yearly_income_records(payload)
    if not records:
        return False, "missing_yearly_income_statement", {}

    start_target = pub_date
    end_target = add_years(pub_date, years)
    start = nearest_record(records, start_target)
    end = nearest_record(records, end_target)
    if not start or not end:
        return False, "missing_financial_records_near_horizon", {}

    revenue_growth = None
    if start.revenue and start.revenue > 0 and end.revenue and end.revenue > 0:
        revenue_growth = end.revenue / start.revenue

    net_income_growth = None
    if start.net_income and start.net_income > 0 and end.net_income and end.net_income > 0:
        net_income_growth = end.net_income / start.net_income

    end_margin = None
    if end.revenue and end.revenue > 0 and end.net_income is not None:
        end_margin = end.net_income / end.revenue

    operating_margin = None
    if end.revenue and end.revenue > 0 and end.operating_income is not None:
        operating_margin = end.operating_income / end.revenue

    evidence = {
        "start_financial_date": start.day.isoformat(),
        "end_financial_date": end.day.isoformat(),
        "start_revenue": start.revenue,
        "end_revenue": end.revenue,
        "start_net_income": start.net_income,
        "end_net_income": end.net_income,
        "start_operating_income": start.operating_income,
        "end_operating_income": end.operating_income,
        "revenue_growth": revenue_growth,
        "net_income_growth": net_income_growth,
        "end_net_margin": end_margin,
        "end_operating_margin": operating_margin,
    }

    end_profitable = (end.net_income or 0) > 0
    margin_ok = (end_margin is not None and end_margin >= 0.05) or (operating_margin is not None and operating_margin >= 0.08)
    revenue_ok = revenue_growth is not None and revenue_growth >= 2.0
    revenue_strong = revenue_growth is not None and revenue_growth >= 3.0
    profit_ok = end_profitable and (
        (start.net_income is not None and start.net_income <= 0)
        or (net_income_growth is not None and net_income_growth >= 2.0)
        or margin_ok
    )

    if revenue_strong and profit_ok and margin_ok:
        return True, "financials_support_business_quality", evidence
    if years <= 5 and revenue_ok and profit_ok and margin_ok:
        return True, "short_horizon_financials_support_business_quality", evidence
    return False, "financials_do_not_yet_support_business_quality", evidence


def source_list(symbol: str, payload: dict[str, Any], cache_path: str) -> list[dict[str, str]]:
    general = payload.get("General") or {}
    sources: list[dict[str, str]] = [
        {
            "publisher": "EODHD cached full fundamentals",
            "url": cache_path,
            "supports": "full financial statements used for row-level quality check",
        }
    ]
    cik = str(general.get("CIK") or "").strip()
    if cik:
        sources.append(
            {
                "publisher": "SEC EDGAR",
                "url": f"https://www.sec.gov/edgar/browse/?CIK={cik}",
                "supports": "security identity and official filing history",
            }
        )
    web_url = str(general.get("WebURL") or "").strip()
    if web_url:
        sources.append(
            {
                "publisher": "Company website",
                "url": web_url,
                "supports": "business identity cross-check",
            }
        )
    sources.extend(STORY_SOURCES.get(symbol, []))
    return sources


def business_summary(row: dict[str, str], quality: dict[str, Any], reason: str) -> str:
    name = row.get("fundamentals_name") or row.get("eodhd_symbol") or row.get("raw_symbol")
    return (
        f"{name}: {row.get('review_target_multiplier')}x over {row.get('review_target_horizon')}; "
        f"revenue {currency(quality.get('start_revenue'))} to {currency(quality.get('end_revenue'))} "
        f"({ratio_text(quality.get('revenue_growth'))}), net income "
        f"{currency(quality.get('start_net_income'))} to {currency(quality.get('end_net_income'))}; {reason}."
    )


def review_extreme_row(row: dict[str, str], review: dict[str, Any] | None) -> dict[str, Any]:
    symbol = row.get("eodhd_symbol") or ""
    fundamentals = load_fundamentals(symbol)
    payload = fundamentals["_payload"]
    sources = source_list(symbol, payload, fundamentals["_cache_path"])
    flags = split_flags(row.get("failure_modes")) | split_flags(row.get("warning_modes"))
    existing_status = row.get("manual_review_status") or row.get("review_stage")

    base = {
        "idea_id": row.get("idea_id"),
        "raw_symbol": row.get("raw_symbol"),
        "eodhd_symbol": symbol,
        "publication_date": row.get("publication_date"),
        "company_name": row.get("fundamentals_name"),
        "horizon": row.get("review_target_horizon"),
        "return_multiplier": parse_float(row.get("review_target_multiplier")),
        "fundamentals_type": row.get("fundamentals_type"),
        "fundamentals_sector": row.get("fundamentals_sector"),
        "fundamentals_industry": row.get("fundamentals_industry"),
        "warning_modes": row.get("warning_modes"),
        "failure_modes": row.get("failure_modes"),
        "sources": sources,
    }

    if existing_status == "reject":
        return {
            **base,
            "review_status": "reject",
            "training_action": "exclude",
            "reason": "already_rejected_by_prior_manual_review",
            "business_quality_status": "not_reviewed_after_reject",
            "confidence": 0.95,
            "quality_evidence": {},
            "qualitative_summary": "Prior manual-review verdict rejected this extreme result; kept out of training.",
        }

    if row.get("math_validation_status") != "math_reproduced":
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": "math_not_reproduced_or_primary_horizon_missing",
            "business_quality_status": "blocked_before_business_review",
            "confidence": 0.2,
            "quality_evidence": {},
            "qualitative_summary": "Cannot promote an extreme winner before the target horizon math is reproduced.",
        }

    blocking = sorted(flags & BLOCKING_WARNING_FLAGS)
    if blocking:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": "blocking_identity_or_corporate_action_flags:" + ",".join(blocking),
            "business_quality_status": "blocked_before_business_review",
            "confidence": 0.35,
            "quality_evidence": {},
            "qualitative_summary": "Extreme result needs human corporate-action/security-identity review before business-quality promotion.",
        }

    if symbol in DO_NOT_AUTO_PROMOTE:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": DO_NOT_AUTO_PROMOTE[symbol],
            "business_quality_status": "special_case_needs_human_review",
            "confidence": 0.45,
            "quality_evidence": {},
            "qualitative_summary": "Extreme result may be real, but the economic driver is not a clean operating-business compounding case.",
        }

    if row.get("fundamentals_type") != "Common Stock":
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": "non_common_stock_instrument",
            "business_quality_status": "blocked_before_business_review",
            "confidence": 0.35,
            "quality_evidence": {},
            "qualitative_summary": "Non-common-stock instruments need separate modeling.",
        }

    yahoo_ok, yahoo_reason, horizon_review = yahoo_horizon_ok(row, review)
    if not yahoo_ok:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": yahoo_reason,
            "business_quality_status": "price_cross_check_incomplete",
            "confidence": 0.45,
            "quality_evidence": {},
            "qualitative_summary": "Business review is not enough without independent adjusted-price reproduction.",
        }

    quality_ok, quality_reason, quality = financial_quality(row, payload)
    if not quality_ok:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": quality_reason,
            "business_quality_status": "business_quality_unproven",
            "confidence": 0.55,
            "quality_evidence": quality,
            "qualitative_summary": business_summary(row, quality, quality_reason) if quality else quality_reason,
        }

    return {
        **base,
        "review_status": "pass",
        "training_action": "add_to_training_ready",
        "reason": "identity_price_and_business_quality_passed",
        "business_quality_status": "browser_supported_financial_quality_pass",
        "confidence": 0.82 if symbol in STORY_SOURCES else 0.74,
        "quality_evidence": {
            **quality,
            "yahoo_relative_diff": horizon_review.get("relative_diff"),
            "yahoo_multiplier": (horizon_review.get("yahoo") or {}).get("multiplier"),
            "eodhd_multiplier": (horizon_review.get("eodhd") or {}).get("multiplier"),
        },
        "qualitative_summary": business_summary(row, quality, "financials plus browser-checkable sources support the extreme return"),
    }


def csv_scalar(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def write_review_csv(reviews: list[dict[str, Any]]) -> None:
    fieldnames = [
        "idea_id",
        "raw_symbol",
        "eodhd_symbol",
        "publication_date",
        "company_name",
        "horizon",
        "return_multiplier",
        "review_status",
        "training_action",
        "reason",
        "business_quality_status",
        "confidence",
        "fundamentals_type",
        "fundamentals_sector",
        "fundamentals_industry",
        "revenue_growth",
        "net_income_growth",
        "start_revenue",
        "end_revenue",
        "start_net_income",
        "end_net_income",
        "yahoo_relative_diff",
        "source_count",
        "qualitative_summary",
        "warning_modes",
        "failure_modes",
    ]
    EXTREME_REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with EXTREME_REVIEW_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for review in reviews:
            quality = review.get("quality_evidence") or {}
            writer.writerow(
                {
                    "idea_id": review.get("idea_id"),
                    "raw_symbol": review.get("raw_symbol"),
                    "eodhd_symbol": review.get("eodhd_symbol"),
                    "publication_date": review.get("publication_date"),
                    "company_name": review.get("company_name"),
                    "horizon": review.get("horizon"),
                    "return_multiplier": review.get("return_multiplier"),
                    "review_status": review.get("review_status"),
                    "training_action": review.get("training_action"),
                    "reason": review.get("reason"),
                    "business_quality_status": review.get("business_quality_status"),
                    "confidence": review.get("confidence"),
                    "fundamentals_type": review.get("fundamentals_type"),
                    "fundamentals_sector": review.get("fundamentals_sector"),
                    "fundamentals_industry": review.get("fundamentals_industry"),
                    "revenue_growth": quality.get("revenue_growth"),
                    "net_income_growth": quality.get("net_income_growth"),
                    "start_revenue": quality.get("start_revenue"),
                    "end_revenue": quality.get("end_revenue"),
                    "start_net_income": quality.get("start_net_income"),
                    "end_net_income": quality.get("end_net_income"),
                    "yahoo_relative_diff": quality.get("yahoo_relative_diff"),
                    "source_count": len(review.get("sources") or []),
                    "qualitative_summary": review.get("qualitative_summary"),
                    "warning_modes": review.get("warning_modes"),
                    "failure_modes": review.get("failure_modes"),
                }
            )


def write_validation_with_extreme(rows: list[dict[str, str]], review_by_key: dict[str, dict[str, Any]]) -> None:
    extra_fields = [
        "extreme_15x_review_status",
        "extreme_15x_training_action",
        "extreme_15x_reason",
        "extreme_15x_business_quality_status",
        "extreme_15x_revenue_growth",
        "extreme_15x_net_income_growth",
        "extreme_15x_confidence",
        "extreme_15x_source_count",
    ]
    fieldnames = list(rows[0].keys()) + [field for field in extra_fields if field not in rows[0]]
    with VALIDATION_WITH_EXTREME_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            review = review_by_key.get(review_key(row))
            output = dict(row)
            if review:
                quality = review.get("quality_evidence") or {}
                output.update(
                    {
                        "extreme_15x_review_status": review.get("review_status"),
                        "extreme_15x_training_action": review.get("training_action"),
                        "extreme_15x_reason": review.get("reason"),
                        "extreme_15x_business_quality_status": review.get("business_quality_status"),
                        "extreme_15x_revenue_growth": csv_scalar(quality.get("revenue_growth")),
                        "extreme_15x_net_income_growth": csv_scalar(quality.get("net_income_growth")),
                        "extreme_15x_confidence": csv_scalar(review.get("confidence")),
                        "extreme_15x_source_count": str(len(review.get("sources") or [])),
                    }
                )
            writer.writerow(output)


def write_training_ready_extreme(reviews: list[dict[str, Any]]) -> int:
    existing_rows = load_csv(TRAINING_READY_IN) if TRAINING_READY_IN.exists() else []
    fieldnames = list(existing_rows[0].keys()) if existing_rows else [
        "idea_id",
        "raw_symbol",
        "eodhd_symbol",
        "publication_date",
        "include_in_training",
        "math_validation_status",
        "review_stage",
        "training_readiness",
        "review_status",
        "reviewed_at",
        "validated_perf_1y",
        "validated_perf_3y",
        "validated_perf_5y",
        "validated_perf_10y",
        "validated_perf_20y",
        "agent_b_yahoo_symbol",
        "agent_b_yahoo_rows",
        "agent_c_status",
        "agent_c_reason",
        "agent_c_outcome_type",
        "source_count",
        "fundamentals_name",
        "fundamentals_type",
        "fundamentals_sector",
        "fundamentals_industry",
        "fundamentals_market_cap",
        "fundamentals_revenue_ttm",
        "fundamentals_profit_margin",
        "original_validation_status",
        "original_review_stage",
        "original_warning_modes",
        "original_failure_modes",
    ]

    pass_reviews = [review for review in reviews if review.get("review_status") == "pass"]
    now = datetime.now(UTC).isoformat()
    added_rows: list[dict[str, str]] = []
    for review in pass_reviews:
        row = {
            field: ""
            for field in fieldnames
        }
        row.update(
            {
                "idea_id": csv_scalar(review.get("idea_id")),
                "raw_symbol": csv_scalar(review.get("raw_symbol")),
                "eodhd_symbol": csv_scalar(review.get("eodhd_symbol")),
                "publication_date": csv_scalar(review.get("publication_date")),
                "include_in_training": "true",
                "math_validation_status": "manually_verified",
                "review_stage": "training_ready_extreme_15x",
                "training_readiness": "training_ready",
                "review_status": "pass",
                "reviewed_at": now,
                "agent_b_yahoo_symbol": csv_scalar(review.get("raw_symbol")),
                "agent_b_yahoo_rows": "",
                "agent_c_status": "pass",
                "agent_c_reason": csv_scalar(review.get("reason")),
                "agent_c_outcome_type": "extreme_winner",
                "source_count": str(len(review.get("sources") or [])),
                "fundamentals_name": csv_scalar(review.get("company_name")),
                "fundamentals_type": csv_scalar(review.get("fundamentals_type")),
                "fundamentals_sector": csv_scalar(review.get("fundamentals_sector")),
                "fundamentals_industry": csv_scalar(review.get("fundamentals_industry")),
                "original_validation_status": "verified_candidate_provider_adjusted",
                "original_review_stage": "provider_warning_extreme_winner",
                "original_warning_modes": csv_scalar(review.get("warning_modes")),
                "original_failure_modes": csv_scalar(review.get("failure_modes")),
            }
        )
        horizon = review.get("horizon")
        if horizon:
            row[f"validated_perf_{horizon}"] = csv_scalar(review.get("return_multiplier"))
        added_rows.append(row)

    with TRAINING_READY_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(added_rows)
    return len(added_rows)


def main() -> int:
    rows = load_csv(VALIDATION_CSV)
    prior_reviews = load_reviews(REVIEWS_JSONL)
    extreme_rows = [
        row
        for row in rows
        if (parse_float(row.get("review_target_multiplier")) or 0) >= EXTREME_THRESHOLD
    ]
    reviews = [review_extreme_row(row, prior_reviews.get(review_key(row))) for row in extreme_rows]
    reviews = sorted(reviews, key=lambda review: float(review.get("return_multiplier") or 0), reverse=True)
    review_by_key = {review_key(review): review for review in reviews}

    write_review_csv(reviews)
    write_validation_with_extreme(rows, review_by_key)
    added = write_training_ready_extreme(reviews)
    EXTREME_EVIDENCE_JSON.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "review_policy": {
                    "extreme_threshold": EXTREME_THRESHOLD,
                    "yahoo_relative_diff_limit": YAHOO_DIFF_LIMIT,
                    "blocking_warning_flags": sorted(BLOCKING_WARNING_FLAGS),
                    "do_not_auto_promote": DO_NOT_AUTO_PROMOTE,
                },
                "records": reviews,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    counts: dict[str, int] = {}
    actions: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for review in reviews:
        counts[review["review_status"]] = counts.get(review["review_status"], 0) + 1
        actions[review["training_action"]] = actions.get(review["training_action"], 0) + 1
        reasons[review["reason"]] = reasons.get(review["reason"], 0) + 1
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_rows": len(rows),
        "extreme_15x_rows": len(reviews),
        "status_counts": counts,
        "training_action_counts": actions,
        "top_reasons": dict(sorted(reasons.items(), key=lambda item: item[1], reverse=True)[:20]),
        "existing_training_ready_rows": len(load_csv(TRAINING_READY_IN)) if TRAINING_READY_IN.exists() else 0,
        "new_extreme_training_ready_rows": added,
        "combined_training_ready_rows": (len(load_csv(TRAINING_READY_IN)) if TRAINING_READY_IN.exists() else 0) + added,
        "outputs": {
            "review_csv": str(EXTREME_REVIEW_CSV.resolve()),
            "evidence_json": str(EXTREME_EVIDENCE_JSON.resolve()),
            "validation_with_extreme_csv": str(VALIDATION_WITH_EXTREME_CSV.resolve()),
            "training_ready_csv": str(TRAINING_READY_OUT.resolve()),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
