#!/usr/bin/env python3
"""Cross-market shareholder-yield screen for Korea EODHD and Japan Yahoo data.

Shareholder yield here is:

    (average annual buyback cash + average annual dividends paid) / current market cap

over the latest available five fiscal years. Buybacks count only when cash-flow
fields indicate a net cash outflow to repurchase stock. Positive issuance/inflow
does not count as a buyback.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
EODHD_DIR = ROOT / "eodhd_output" / "japan_korea_fundamentals"
JPX_XLS = ROOT / "eodhd_output" / "jpx_listed_companies_data_j.xls"
OUT_DIR = EODHD_DIR / "screening"
YF_CACHE_DIR = EODHD_DIR / "japan_yahoo_cache"

EODHD_BUYBACK_FIELDS = [
    "salePurchaseOfStock",
    "salePurchaseOfStockNet",
    "repurchaseOfCapitalStock",
    "repurchasesOfCapitalStock",
    "purchaseOfStock",
    "commonStockRepurchased",
]
EODHD_DIVIDEND_FIELDS = ["dividendsPaid", "cashDividendsPaid", "commonDividendsPaid", "paymentOfDividends"]

YF_BUYBACK_ROWS = [
    "Repurchase Of Capital Stock",
    "Common Stock Payments",
    "Net Common Stock Issuance",
]
YF_DIVIDEND_ROWS = ["Cash Dividends Paid", "Common Stock Dividend Paid"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", default="KR,JP", help="Comma-separated markets to include: KR,JP.")
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--japan-limit", type=int, default=None, help="Optional cap for Japan tickers this run.")
    parser.add_argument("--japan-offset", type=int, default=0)
    parser.add_argument("--japan-time-budget-sec", type=float, default=0.0, help="0 means no time budget.")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--min-market-cap", type=float, default=0.0)
    parser.add_argument("--include-share-classes", action="store_true", help="Include preferred/duplicate share-class rows.")
    parser.add_argument("--output-prefix", default="cross_market_shareholder_yield_5y")
    return parser.parse_args()


def to_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def first_number(record: dict[str, Any], fields: list[str]) -> tuple[float | None, str | None]:
    lower = {str(k).lower(): k for k in record}
    for field in fields:
        key = field if field in record else lower.get(field.lower())
        if key is None:
            continue
        value = to_float(record.get(key))
        if value is not None:
            return value, str(key)
    return None, None


def safe_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace("\\", "_").replace(":", "_")


def is_share_class(symbol: str, name: str, known_symbols: set[str] | None = None) -> bool:
    text = f"{symbol} {name}".lower()
    if re.search(r"\b(pref|preference|preferred|우선주|1우|2우|3우)\b", text):
        return True
    if known_symbols:
        match = re.match(r"^(\d{5})([579])\.(KO|KQ)$", symbol)
        if match and f"{match.group(1)}0.{match.group(3)}" in known_symbols:
            return True
    return False


def eodhd_yearly_cash_flows(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    records = payload.get("Financials", {}).get("Cash_Flow", {}).get("yearly", {})
    if not isinstance(records, dict):
        return []
    rows = [(str(date), row) for date, row in records.items() if isinstance(row, dict)]
    rows.sort(key=lambda item: item[0], reverse=True)
    return rows[:5]


def eodhd_market_cap(payload: dict[str, Any]) -> tuple[float | None, str | None]:
    highlights = payload.get("Highlights", {})
    if isinstance(highlights, dict):
        value, field = first_number(highlights, ["MarketCapitalization", "MarketCap"])
        if value is not None:
            return value, f"Highlights.{field}"
    return None, None


def eodhd_company_row(
    manifest_row: dict[str, Any],
    include_share_classes: bool,
    known_symbols: set[str] | None = None,
) -> dict[str, Any] | None:
    symbol = manifest_row["symbol"]
    raw_path = EODHD_DIR / "raw" / safe_name(symbol) / "fundamentals.json"
    if not raw_path.exists():
        return None
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    general = payload.get("General", {}) if isinstance(payload.get("General"), dict) else {}
    name = general.get("Name") or manifest_row.get("name") or symbol
    warnings: list[str] = []
    if is_share_class(symbol, name, known_symbols):
        warnings.append("possible preferred/duplicate share class; company-level cash flows may not match class-level market cap")
        if not include_share_classes:
            return None

    market_cap, market_cap_source = eodhd_market_cap(payload)
    cash_rows = eodhd_yearly_cash_flows(payload)
    dividends: list[float] = []
    buybacks: list[float] = []
    dividend_sources: list[str] = []
    buyback_sources: list[str] = []
    years: list[str] = []
    for date, row in cash_rows:
        years.append(date)
        div_value, div_field = first_number(row, EODHD_DIVIDEND_FIELDS)
        if div_value is not None:
            dividends.append(abs(div_value))
            dividend_sources.append(f"Financials.Cash_Flow.yearly.{date}.{div_field}")
        buy_value, buy_field = first_number(row, EODHD_BUYBACK_FIELDS)
        if buy_value is None:
            buybacks.append(0.0)
        elif buy_value < 0:
            buybacks.append(abs(buy_value))
            buyback_sources.append(f"Financials.Cash_Flow.yearly.{date}.{buy_field}")
        else:
            buybacks.append(0.0)
    if len(cash_rows) < 5:
        warnings.append(f"only {len(cash_rows)} annual cash-flow rows available")
    if len(dividends) < 3:
        warnings.append(f"only {len(dividends)} dividend observations")
    if not buyback_sources:
        warnings.append("no negative buyback/share-repurchase cash outflows found in latest five years")

    avg_div = sum(dividends) / 5.0 if cash_rows else None
    avg_buy = sum(buybacks) / 5.0 if cash_rows else None
    total_avg = (avg_div or 0.0) + (avg_buy or 0.0)
    shareholder_yield = total_avg / market_cap if market_cap and market_cap > 0 else None
    if shareholder_yield is None:
        warnings.append("missing or non-positive market cap")

    return {
        "symbol": symbol,
        "market": "KR",
        "name": name,
        "company_description": general.get("Description") or "",
        "sector": general.get("Sector") or "",
        "industry": general.get("Industry") or "",
        "currency": general.get("CurrencyCode") or manifest_row.get("currency") or "",
        "market_cap": market_cap,
        "avg_annual_buyback_5y": avg_buy,
        "avg_annual_dividends_5y": avg_div,
        "avg_annual_cash_returned_5y": total_avg,
        "shareholder_yield_5y_avg": shareholder_yield,
        "buyback_yield_5y_avg": avg_buy / market_cap if market_cap and avg_buy is not None else None,
        "dividend_yield_5y_avg": avg_div / market_cap if market_cap and avg_div is not None else None,
        "cash_flow_years_used": ";".join(years),
        "cash_flow_observations": len(cash_rows),
        "source_market_cap": market_cap_source,
        "source_buybacks": ";".join(buyback_sources),
        "source_dividends": ";".join(dividend_sources),
        "data_source": "EODHD Fundamentals API local raw payload",
        "data_quality_warnings": "; ".join(dict.fromkeys(warnings)),
    }


def load_jpx_domestic_stocks() -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_excel(JPX_XLS)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        market = str(row.get("市場・商品区分") or "")
        raw_code = row.get("コード")
        code = str(raw_code or "").strip()
        if isinstance(raw_code, (int, float)) and math.isfinite(raw_code):
            code = str(int(raw_code)).zfill(4)
        if "内国株式" not in market:
            continue
        if not re.fullmatch(r"\d{4}", code):
            continue
        rows.append(
            {
                "symbol": f"{code}.T",
                "code": code,
                "name": str(row.get("銘柄名") or ""),
                "jpx_market_segment": market,
                "jpx_industry_33": str(row.get("33業種区分") or ""),
                "jpx_industry_17": str(row.get("17業種区分") or ""),
                "jpx_size": str(row.get("規模区分") or ""),
            }
        )
    rows.sort(key=lambda item: item["symbol"])
    return rows


def cache_path(symbol: str) -> Path:
    return YF_CACHE_DIR / f"{safe_name(symbol)}.json"


def yahoo_fetch(symbol: str) -> dict[str, Any]:
    root_text = str(ROOT)
    sys.path[:] = [path for path in sys.path if Path(path or os.getcwd()).resolve() != ROOT]
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    info = ticker.get_info()
    cashflow = ticker.cashflow
    cf_payload: dict[str, dict[str, float | None]] = {}
    if cashflow is not None and not cashflow.empty:
        for row_name in cashflow.index:
            values: dict[str, float | None] = {}
            for col, value in cashflow.loc[row_name].items():
                number = to_float(value)
                values[str(getattr(col, "date", lambda: col)()) if hasattr(col, "date") else str(col)] = number
            cf_payload[str(row_name)] = values
    return {"info": info, "cashflow": cf_payload, "fetched_at": time.time()}


def yahoo_cached(symbol: str) -> tuple[dict[str, Any] | None, str | None]:
    path = cache_path(symbol)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8")), "cache"
        except json.JSONDecodeError:
            pass
    try:
        payload = yahoo_fetch(symbol)
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload, "fetched"


def row_values(payload: dict[str, Any], labels: list[str]) -> tuple[dict[str, float], str | None]:
    cashflow = payload.get("cashflow") if isinstance(payload, dict) else None
    if not isinstance(cashflow, dict):
        return {}, None
    for label in labels:
        values = cashflow.get(label)
        if isinstance(values, dict):
            out = {date: value for date, value in values.items() if isinstance(value, (int, float)) and math.isfinite(value)}
            if out:
                return out, label
    return {}, None


def japan_company_row(jpx_row: dict[str, Any], include_share_classes: bool) -> dict[str, Any] | None:
    symbol = jpx_row["symbol"]
    name = jpx_row["name"]
    warnings: list[str] = []
    if is_share_class(symbol, name):
        warnings.append("possible preferred/duplicate share class; excluded unless --include-share-classes")
        if not include_share_classes:
            return None
    payload, status = yahoo_cached(symbol)
    if payload is None:
        return {
            "symbol": symbol,
            "market": "JP",
            "name": name,
            "data_quality_warnings": f"Yahoo fetch failed: {status}",
            "data_source": "Yahoo Finance via yfinance",
        }
    info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
    market_cap = to_float(info.get("marketCap"))
    div_values, div_label = row_values(payload, YF_DIVIDEND_ROWS)
    buy_values, buy_label = row_values(payload, YF_BUYBACK_ROWS)
    dates = sorted(set(div_values) | set(buy_values), reverse=True)[:5]
    if len(dates) < 5:
        warnings.append(f"only {len(dates)} annual cash-flow columns available")
    dividends = [abs(div_values[d]) for d in dates if d in div_values]
    buybacks: list[float] = []
    for date in dates:
        value = buy_values.get(date)
        buybacks.append(abs(value) if value is not None and value < 0 else 0.0)
    if len(dividends) < 3:
        warnings.append(f"only {len(dividends)} dividend observations")
    if not any(buybacks):
        warnings.append("no negative buyback/share-repurchase cash outflows found in latest five years")
    avg_div = sum(dividends) / 5.0 if dates else None
    avg_buy = sum(buybacks) / 5.0 if dates else None
    total_avg = (avg_div or 0.0) + (avg_buy or 0.0)
    shareholder_yield = total_avg / market_cap if market_cap and market_cap > 0 else None
    if shareholder_yield is None:
        warnings.append("missing or non-positive market cap")
    return {
        "symbol": symbol,
        "market": "JP",
        "name": info.get("longName") or name,
        "company_description": info.get("longBusinessSummary") or "",
        "sector": info.get("sector") or "",
        "industry": info.get("industry") or jpx_row.get("jpx_industry_33") or "",
        "currency": info.get("currency") or "JPY",
        "market_cap": market_cap,
        "avg_annual_buyback_5y": avg_buy,
        "avg_annual_dividends_5y": avg_div,
        "avg_annual_cash_returned_5y": total_avg,
        "shareholder_yield_5y_avg": shareholder_yield,
        "buyback_yield_5y_avg": avg_buy / market_cap if market_cap and avg_buy is not None else None,
        "dividend_yield_5y_avg": avg_div / market_cap if market_cap and avg_div is not None else None,
        "cash_flow_years_used": ";".join(dates),
        "cash_flow_observations": len(dates),
        "source_market_cap": "Yahoo.info.marketCap",
        "source_buybacks": f"Yahoo.cashflow.{buy_label}" if buy_label else "",
        "source_dividends": f"Yahoo.cashflow.{div_label}" if div_label else "",
        "data_source": f"Yahoo Finance via yfinance ({status}) + JPX listed-company spreadsheet",
        "jpx_market_segment": jpx_row.get("jpx_market_segment", ""),
        "jpx_industry_33": jpx_row.get("jpx_industry_33", ""),
        "jpx_industry_17": jpx_row.get("jpx_industry_17", ""),
        "jpx_size": jpx_row.get("jpx_size", ""),
        "data_quality_warnings": "; ".join(dict.fromkeys(warnings)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    markets = {market.strip().upper() for market in args.markets.split(",") if market.strip()}
    rows: list[dict[str, Any]] = []

    if "KR" in markets:
        manifest = json.loads((EODHD_DIR / "stock_pull_manifest.json").read_text(encoding="utf-8"))
        known_symbols = {item.get("symbol") for item in manifest if isinstance(item, dict) and item.get("symbol")}
        for item in manifest:
            if not isinstance(item, dict):
                continue
            row = eodhd_company_row(item, args.include_share_classes, known_symbols)
            if row is not None:
                rows.append(row)

    japan_attempted = 0
    if "JP" in markets:
        jpx_rows = load_jpx_domestic_stocks()
        selected = jpx_rows[args.japan_offset :]
        if args.japan_limit is not None:
            selected = selected[: args.japan_limit]
        deadline = time.monotonic() + args.japan_time_budget_sec if args.japan_time_budget_sec > 0 else None
        for item in selected:
            if deadline is not None and time.monotonic() >= deadline:
                break
            japan_attempted += 1
            row = japan_company_row(item, args.include_share_classes)
            if row is not None:
                rows.append(row)
            time.sleep(args.sleep)

    for row in rows:
        market_cap = row.get("market_cap")
        if args.min_market_cap and (not market_cap or market_cap < args.min_market_cap):
            row["excluded_by_min_market_cap"] = True
        else:
            row["excluded_by_min_market_cap"] = False
    ranked = [
        row
        for row in rows
        if not row.get("excluded_by_min_market_cap")
        and isinstance(row.get("shareholder_yield_5y_avg"), (int, float))
        and row.get("market_cap")
    ]
    ranked.sort(
        key=lambda row: (
            row.get("cash_flow_observations") == 5,
            row.get("shareholder_yield_5y_avg") or -999,
            row.get("buyback_yield_5y_avg") or -999,
            row.get("dividend_yield_5y_avg") or -999,
        ),
        reverse=True,
    )
    for index, row in enumerate(ranked, 1):
        row["rank"] = index

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_path = OUT_DIR / f"{args.output_prefix}_all.csv"
    top_path = OUT_DIR / f"{args.output_prefix}_top.csv"
    write_csv(all_path, ranked)
    write_csv(top_path, ranked[: args.top])
    summary = {
        "markets": sorted(markets),
        "rows_ranked": len(ranked),
        "japan_attempted_this_run": japan_attempted,
        "japan_cache_files": len(list(YF_CACHE_DIR.glob("*.json"))) if YF_CACHE_DIR.exists() else 0,
        "metric": "(average annual buyback cash + average annual dividends paid over latest five fiscal years) / current market cap",
        "outputs": {"all": str(all_path), "top": str(top_path)},
    }
    (OUT_DIR / f"{args.output_prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
