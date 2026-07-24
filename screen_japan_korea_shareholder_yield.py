#!/usr/bin/env python3
"""Screen Japan/Korea EODHD fundamentals for shareholder yield and quality.

The screener reads raw EODHD Fundamentals API payloads produced by
eodhd_japan_korea_fundamentals.py and ranks companies by shareholder yield,
5-year average ROE, revenue CAGR, net income CAGR, and free cash flow CAGR.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATASET_DIR = Path(os.environ.get("EODHD_JK_OUT_DIR", str(ROOT / "eodhd_output" / "japan_korea_fundamentals")))
RAW_DIR = DATASET_DIR / "raw"
MANIFEST_PATH = DATASET_DIR / "stock_pull_manifest.json"
SCREEN_DIR = DATASET_DIR / "screening"


CASH_FLOW_BUYBACK_FIELDS = [
    "salePurchaseOfStock",
    "salePurchaseOfStockNet",
    "repurchaseOfCapitalStock",
    "repurchasesOfCapitalStock",
    "purchaseOfStock",
    "commonStockRepurchased",
]

CASH_FLOW_DIVIDEND_FIELDS = [
    "dividendsPaid",
    "cashDividendsPaid",
    "commonDividendsPaid",
    "paymentOfDividends",
]

CASH_FLOW_CFO_FIELDS = [
    "totalCashFromOperatingActivities",
    "netCashProvidedByOperatingActivities",
    "operatingCashFlow",
]

CASH_FLOW_CAPEX_FIELDS = [
    "capitalExpenditures",
    "capitalExpenditure",
    "capitalExpendituresReported",
]

INCOME_REVENUE_FIELDS = ["totalRevenue", "revenue", "netSales"]
INCOME_NET_INCOME_FIELDS = ["netIncome", "netIncomeApplicableToCommonShares", "netIncomeFromContinuingOps"]
BALANCE_EQUITY_FIELDS = ["totalStockholderEquity", "totalStockholdersEquity", "totalEquity", "commonStockEquity"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-market-cap", type=float, default=0.0, help="Minimum market cap to include.")
    parser.add_argument("--min-years", type=int, default=4, help="Minimum annual observations for CAGR/average metrics.")
    parser.add_argument("--top", type=int, default=100, help="Number of ranked rows to write to top CSV.")
    parser.add_argument("--include-negative-yield", action="store_true", help="Keep rows with negative shareholder yield.")
    parser.add_argument("--output-prefix", default="shareholder_yield_screen", help="Output filename prefix.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace("\\", "_").replace(":", "_")


def to_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def first_number(record: dict[str, Any], field_names: list[str]) -> tuple[float | None, str | None]:
    for field in field_names:
        if field in record:
            number = to_float(record.get(field))
            if number is not None:
                return number, field
    lower_map = {str(key).lower(): key for key in record}
    for field in field_names:
        key = lower_map.get(field.lower())
        if key is not None:
            number = to_float(record.get(key))
            if number is not None:
                return number, str(key)
    return None, None


def yearly_records(fundamentals: dict[str, Any], statement_key: str) -> list[tuple[str, dict[str, Any]]]:
    records = fundamentals.get("Financials", {}).get(statement_key, {}).get("yearly", {})
    if not isinstance(records, dict):
        return []
    rows = [(str(date_key), record) for date_key, record in records.items() if isinstance(record, dict)]
    rows.sort(key=lambda item: item[0], reverse=True)
    return rows


def latest_market_cap(fundamentals: dict[str, Any]) -> tuple[float | None, str | None]:
    highlights = fundamentals.get("Highlights", {})
    if isinstance(highlights, dict):
        value, field = first_number(highlights, ["MarketCapitalization", "MarketCap"])
        if value is not None:
            return value, f"Highlights.{field}"
    general = fundamentals.get("General", {})
    if isinstance(general, dict):
        value, field = first_number(general, ["MarketCapitalization", "MarketCap"])
        if value is not None:
            return value, f"General.{field}"
    return None, None


def latest_annual_cash_flow_record(fundamentals: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    records = yearly_records(fundamentals, "Cash_Flow")
    if not records:
        return None, None
    return records[0]


def latest_dividend_metric(fundamentals: dict[str, Any]) -> tuple[float | None, str | None, str | None]:
    date_key, record = latest_annual_cash_flow_record(fundamentals)
    if date_key is None or record is None:
        return None, None, None
    value, field = first_number(record, CASH_FLOW_DIVIDEND_FIELDS)
    if value is None:
        return None, None, date_key
    return abs(value), f"Financials.Cash_Flow.yearly.{date_key}.{field}", date_key


def latest_buyback_metric(fundamentals: dict[str, Any]) -> tuple[float | None, str | None, str | None, str | None]:
    date_key, record = latest_annual_cash_flow_record(fundamentals)
    if date_key is None or record is None:
        return None, None, None, "missing latest annual cash-flow record"
    value, field = first_number(record, CASH_FLOW_BUYBACK_FIELDS)
    if value is None:
        return 0.0, None, date_key, "missing latest-year buyback/share-repurchase cash-flow field"
    if value < 0:
        return abs(value), f"Financials.Cash_Flow.yearly.{date_key}.{field}", date_key, None
    if value > 0:
        return 0.0, f"Financials.Cash_Flow.yearly.{date_key}.{field}", date_key, (
            "latest-year stock cash-flow field was positive; treated as issuance/inflow, not buyback"
        )
    return 0.0, f"Financials.Cash_Flow.yearly.{date_key}.{field}", date_key, None


def latest_cash_flow_metric(fundamentals: dict[str, Any], fields: list[str]) -> tuple[float | None, str | None, str | None]:
    date_key, record = latest_annual_cash_flow_record(fundamentals)
    if date_key is None or record is None:
        return None, None, None
    value, field = first_number(record, fields)
    if value is not None:
        return abs(value), f"Financials.Cash_Flow.yearly.{date_key}.{field}", date_key
    return None, None, None


def value_series(fundamentals: dict[str, Any], statement_key: str, fields: list[str], limit: int) -> tuple[list[tuple[str, float]], list[str]]:
    out: list[tuple[str, float]] = []
    source_fields: list[str] = []
    for date_key, record in yearly_records(fundamentals, statement_key):
        value, field = first_number(record, fields)
        if value is None:
            continue
        out.append((date_key, value))
        source_fields.append(f"Financials.{statement_key}.yearly.{date_key}.{field}")
        if len(out) >= limit:
            break
    return out, source_fields


def free_cash_flow_series(fundamentals: dict[str, Any], limit: int) -> tuple[list[tuple[str, float]], list[str], list[str]]:
    rows: list[tuple[str, float]] = []
    sources: list[str] = []
    warnings: list[str] = []
    for date_key, record in yearly_records(fundamentals, "Cash_Flow"):
        cfo, cfo_field = first_number(record, CASH_FLOW_CFO_FIELDS)
        capex, capex_field = first_number(record, CASH_FLOW_CAPEX_FIELDS)
        if cfo is None or capex is None:
            continue
        fcf = cfo + capex if capex < 0 else cfo - capex
        rows.append((date_key, fcf))
        sources.append(f"Financials.Cash_Flow.yearly.{date_key}.{cfo_field}+{capex_field}")
        if capex > 0:
            warnings.append(f"{date_key}: capex was positive; treated FCF as CFO - capex")
        if len(rows) >= limit:
            break
    return rows, sources, warnings


def cagr_from_latest_series(series: list[tuple[str, float]], min_years: int) -> tuple[float | None, str | None]:
    if len(series) < min_years:
        return None, f"only {len(series)} usable annual observations"
    latest_date, latest = series[0]
    oldest_date, oldest = series[min(len(series), 5) - 1]
    periods = min(len(series), 5) - 1
    if periods <= 0:
        return None, "not enough periods"
    if oldest <= 0 or latest <= 0:
        return None, f"non-positive CAGR endpoint: {oldest_date}={oldest}, {latest_date}={latest}"
    return (latest / oldest) ** (1.0 / periods) - 1.0, None


def average_roe(fundamentals: dict[str, Any], min_years: int) -> tuple[float | None, list[str], list[str]]:
    income = dict(yearly_records(fundamentals, "Income_Statement"))
    balance = dict(yearly_records(fundamentals, "Balance_Sheet"))
    rows: list[float] = []
    sources: list[str] = []
    warnings: list[str] = []
    for date_key in sorted(set(income) & set(balance), reverse=True):
        net_income, ni_field = first_number(income[date_key], INCOME_NET_INCOME_FIELDS)
        equity, equity_field = first_number(balance[date_key], BALANCE_EQUITY_FIELDS)
        if net_income is None or equity in (None, 0):
            continue
        rows.append(net_income / equity)
        sources.append(
            f"Financials.Income_Statement.yearly.{date_key}.{ni_field}/"
            f"Financials.Balance_Sheet.yearly.{date_key}.{equity_field}"
        )
        if equity is not None and equity < 0:
            warnings.append(f"{date_key}: negative equity makes ROE hard to interpret")
        if len(rows) >= 5:
            break
    if len(rows) < min_years:
        warnings.append(f"only {len(rows)} usable ROE observations")
        return None, sources, warnings
    return sum(rows) / len(rows), sources, warnings


def pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def build_row(manifest_row: dict[str, Any], min_years: int) -> dict[str, Any]:
    symbol = manifest_row["symbol"]
    payload = read_json(RAW_DIR / safe_name(symbol) / "fundamentals.json")
    if not isinstance(payload, dict):
        raise ValueError(f"Fundamentals payload for {symbol} is not a JSON object")

    general = payload.get("General", {}) if isinstance(payload.get("General"), dict) else {}
    market_cap, market_cap_source = latest_market_cap(payload)
    buyback, buyback_source, buyback_date, buyback_warning = latest_buyback_metric(payload)
    dividends, dividend_source, dividend_date = latest_dividend_metric(payload)
    buyback = buyback or 0.0
    dividends = dividends or 0.0

    warnings: list[str] = []
    if market_cap is None or market_cap <= 0:
        warnings.append("missing or non-positive market cap")
        shareholder_yield = None
        buyback_yield = None
        dividend_yield = None
    else:
        buyback_yield = buyback / market_cap
        dividend_yield = dividends / market_cap
        shareholder_yield = buyback_yield + dividend_yield
    if buyback_warning:
        warnings.append(buyback_warning)
    if dividend_source is None:
        warnings.append("missing latest-year dividend-paid cash-flow field")

    revenue_series, revenue_sources = value_series(payload, "Income_Statement", INCOME_REVENUE_FIELDS, 5)
    net_income_series, net_income_sources = value_series(payload, "Income_Statement", INCOME_NET_INCOME_FIELDS, 5)
    fcf_series, fcf_sources, fcf_warnings = free_cash_flow_series(payload, 5)
    warnings.extend(fcf_warnings)
    revenue_cagr, revenue_warning = cagr_from_latest_series(revenue_series, min_years)
    net_income_cagr, net_income_warning = cagr_from_latest_series(net_income_series, min_years)
    fcf_cagr, fcf_warning = cagr_from_latest_series(fcf_series, min_years)
    for warning in [revenue_warning, net_income_warning, fcf_warning]:
        if warning:
            warnings.append(warning)

    roe_avg, roe_sources, roe_warnings = average_roe(payload, min_years)
    warnings.extend(roe_warnings)

    score_parts = [
        shareholder_yield or 0.0,
        (roe_avg or 0.0) * 0.35,
        (revenue_cagr or 0.0) * 0.20,
        (net_income_cagr or 0.0) * 0.20,
        (fcf_cagr or 0.0) * 0.25,
    ]
    composite_score = sum(score_parts)

    return {
        "symbol": symbol,
        "name": general.get("Name") or manifest_row.get("name"),
        "country_key": manifest_row.get("country_key"),
        "exchange": manifest_row.get("exchange"),
        "sector": general.get("Sector"),
        "industry": general.get("Industry"),
        "currency": general.get("CurrencyCode") or manifest_row.get("currency"),
        "market_cap": market_cap,
        "latest_buyback_cash": buyback,
        "latest_dividends_paid_cash": dividends,
        "shareholder_yield": shareholder_yield,
        "buyback_yield": buyback_yield,
        "dividend_yield": dividend_yield,
        "average_roe_5y": roe_avg,
        "revenue_cagr_5y": revenue_cagr,
        "net_income_cagr_5y": net_income_cagr,
        "free_cash_flow_cagr_5y": fcf_cagr,
        "composite_score": composite_score,
        "buyback_date": buyback_date,
        "dividend_date": dividend_date,
        "revenue_observations": len(revenue_series),
        "net_income_observations": len(net_income_series),
        "free_cash_flow_observations": len(fcf_series),
        "roe_observations": len(roe_sources),
        "source_market_cap": market_cap_source,
        "source_buyback": buyback_source,
        "source_dividends": dividend_source,
        "source_revenue": ";".join(revenue_sources),
        "source_net_income": ";".join(net_income_sources),
        "source_free_cash_flow": ";".join(fcf_sources),
        "source_roe": ";".join(roe_sources),
        "data_quality_warnings": "; ".join(dict.fromkeys(warnings)),
        "raw_fundamentals_path": f"raw/{safe_name(symbol)}/fundamentals.json",
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def main() -> int:
    args = parse_args()
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST_PATH}. Run eodhd_japan_korea_fundamentals.py first.")
    manifest = read_json(MANIFEST_PATH)
    if not isinstance(manifest, list):
        raise SystemExit("Manifest is not a JSON array.")

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in manifest:
        if not isinstance(item, dict) or not item.get("symbol"):
            continue
        raw_path = RAW_DIR / safe_name(item["symbol"]) / "fundamentals.json"
        if not raw_path.exists():
            skipped.append({"symbol": item.get("symbol"), "reason": "missing raw fundamentals"})
            continue
        try:
            row = build_row(item, args.min_years)
        except Exception as exc:  # noqa: BLE001 - preserve row-level data-quality failure.
            skipped.append({"symbol": item.get("symbol"), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if args.min_market_cap and (row["market_cap"] is None or row["market_cap"] < args.min_market_cap):
            skipped.append({"symbol": item.get("symbol"), "reason": "below min market cap"})
            continue
        if not args.include_negative_yield and row["shareholder_yield"] is not None and row["shareholder_yield"] < 0:
            skipped.append({"symbol": item.get("symbol"), "reason": "negative shareholder yield"})
            continue
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["shareholder_yield"] is not None,
            row["shareholder_yield"] or -999.0,
            row["average_roe_5y"] or -999.0,
            row["free_cash_flow_cagr_5y"] or -999.0,
        ),
        reverse=True,
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index

    all_path = SCREEN_DIR / f"{args.output_prefix}_all.csv"
    top_path = SCREEN_DIR / f"{args.output_prefix}_top.csv"
    write_rows(all_path, rows)
    write_rows(top_path, rows[: args.top])
    write_rows(SCREEN_DIR / "screening_skipped.csv", skipped)
    summary = {
        "updated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "input_manifest_symbols": len(manifest),
        "screened_rows": len(rows),
        "skipped_rows": len(skipped),
        "top_rows": min(args.top, len(rows)),
        "ranking": [
            "Primary sort: shareholder_yield descending.",
            "Tie-breakers: average_roe_5y and free_cash_flow_cagr_5y descending.",
            "Composite score is provided for reference, not used as the primary rank.",
        ],
        "metric_definitions": {
            "shareholder_yield": "(latest-year negative buyback/share-repurchase cash outflow + latest-year dividends paid cash outflow) / market cap",
            "buyback_yield": "latest-year stock cash-flow outflow / market cap; positive stock cash-flow values are treated as issuance/inflow, not buyback",
            "dividend_yield": "latest dividends paid cash outflow / market cap",
            "average_roe_5y": "mean of annual net income / shareholder equity for the latest usable five fiscal years",
            "revenue_cagr_5y": "CAGR from oldest to latest usable annual revenue among the latest five records",
            "net_income_cagr_5y": "CAGR from oldest to latest usable annual net income among the latest five records",
            "free_cash_flow_cagr_5y": "CAGR of CFO plus negative capex, or CFO minus positive capex, among latest five annual records",
        },
        "outputs": {
            "all": str(all_path),
            "top": str(top_path),
            "skipped": str(SCREEN_DIR / "screening_skipped.csv"),
        },
    }
    (SCREEN_DIR / "screening_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
