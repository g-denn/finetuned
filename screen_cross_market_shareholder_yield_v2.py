#!/usr/bin/env python3
"""Corrected Japan/Korea shareholder-yield screen.

This version avoids the bad comparison that inflated earlier Korea results:
raw statement-level "Cash Dividends Paid" can be company/consolidated/entity
level, while the market cap can be listing/share-class level.

Corrected shareholder yield:

    avg dividend per share over the latest five calendar years / current price
    + avg annual net share-count reduction over the latest five years

The second term is a buyback-yield proxy: shrinking share count is counted as a
positive buyback yield; issuance/dilution reduces the average.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
EODHD_DIR = ROOT / "eodhd_output" / "japan_korea_fundamentals"
JPX_XLS = ROOT / "eodhd_output" / "jpx_listed_companies_data_j.xls"
OUT_DIR = EODHD_DIR / "screening"
YF_CACHE_DIR = EODHD_DIR / "shareholder_yield_v2_yahoo_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", default="KR,JP")
    parser.add_argument("--top", type=int, default=300)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--symbols", default="", help="Optional comma-separated canonical tickers to process.")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--time-budget-sec", type=float, default=0.0)
    parser.add_argument("--sleep", type=float, default=0.03)
    parser.add_argument("--min-market-cap", type=float, default=0.0)
    parser.add_argument("--include-share-classes", action="store_true")
    parser.add_argument("--output-prefix", default="cross_market_shareholder_yield_v2_corrected")
    return parser.parse_args()


def to_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace("\\", "_").replace(":", "_")


def yf_symbol(market: str, symbol: str) -> str:
    if market == "JP":
        return symbol
    if symbol.endswith(".KO"):
        return symbol[:-3] + ".KS"
    return symbol


def is_share_class(symbol: str, name: str, known_symbols: set[str] | None = None) -> bool:
    text = f"{symbol} {name}".lower()
    if re.search(r"\b(pref|preference|preferred|우선주|1우|2우|3우)\b", text):
        return True
    if known_symbols:
        match = re.match(r"^(\d{5})([579])\.(KO|KQ)$", symbol)
        if match and f"{match.group(1)}0.{match.group(3)}" in known_symbols:
            return True
    return False


def remove_workspace_yfinance_shadow() -> None:
    sys.path[:] = [path for path in sys.path if Path(path or os.getcwd()).resolve() != ROOT]


def cache_path(symbol: str) -> Path:
    return YF_CACHE_DIR / f"{safe_name(symbol)}.json"


def fetch_yahoo(symbol: str, refresh: bool = False) -> tuple[dict[str, Any] | None, str]:
    path = cache_path(symbol)
    if path.exists() and not refresh:
        try:
            return json.loads(path.read_text(encoding="utf-8")), "cache"
        except json.JSONDecodeError:
            pass

    remove_workspace_yfinance_shadow()
    import yfinance as yf

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.get_info()
        dividends = []
        div_series = ticker.dividends
        if div_series is not None and not div_series.empty:
            for index, value in div_series.items():
                dividends.append({"date": str(index.date()), "value": to_float(value)})

        shares = []
        share_series = ticker.get_shares_full(start="2020-01-01")
        if share_series is not None and not share_series.empty:
            for index, value in share_series.items():
                shares.append({"date": str(index.date()), "value": to_float(value)})

        payload = {"info": info, "dividends": dividends, "shares": shares, "fetched_at_utc": dt.datetime.now(dt.UTC).isoformat()}
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload, "fetched"


def annual_dividends(dividends: list[dict[str, Any]], years: int = 5) -> tuple[list[tuple[int, float]], list[str]]:
    by_year: dict[int, float] = defaultdict(float)
    for item in dividends:
        value = to_float(item.get("value"))
        if value is None or value <= 0:
            continue
        try:
            year = int(str(item.get("date"))[:4])
        except ValueError:
            continue
        by_year[year] += value
    if not by_year:
        return [], ["no historical dividend-per-share observations"]
    latest_year = max(by_year)
    selected = [(year, by_year.get(year, 0.0)) for year in range(latest_year, latest_year - years, -1)]
    warnings: list[str] = []
    nonzero = sum(1 for _, value in selected if value > 0)
    if nonzero < 3:
        warnings.append(f"only {nonzero} non-zero annual dividend-per-share observations in latest {years} years")
    return selected, warnings


def annual_share_counts(shares: list[dict[str, Any]]) -> list[tuple[int, float]]:
    latest_by_year: dict[int, tuple[str, float]] = {}
    for item in shares:
        value = to_float(item.get("value"))
        if value is None or value <= 0:
            continue
        date = str(item.get("date") or "")
        try:
            year = int(date[:4])
        except ValueError:
            continue
        if year not in latest_by_year or date > latest_by_year[year][0]:
            latest_by_year[year] = (date, value)
    return sorted([(year, value) for year, (_, value) in latest_by_year.items()], reverse=True)


def buyback_yield_from_shares(shares: list[dict[str, Any]], years: int = 5) -> tuple[float | None, list[str], str]:
    annual = annual_share_counts(shares)
    if len(annual) < 2:
        return None, ["not enough share-count history for buyback yield"], ""
    latest_year = annual[0][0]
    annual_map = dict(annual)
    selected = [(year, annual_map.get(year)) for year in range(latest_year, latest_year - years - 1, -1)]
    usable = [(year, value) for year, value in selected if value is not None]
    warnings: list[str] = []
    if len(usable) < 3:
        warnings.append(f"only {len(usable)} annual share-count observations")
    annual_yields: list[float] = []
    for (new_year, new_shares), (old_year, old_shares) in zip(usable, usable[1:]):
        if old_shares and new_shares:
            annual_yields.append((old_shares - new_shares) / old_shares)
    if not annual_yields:
        return None, warnings + ["could not compute annual share-count change"], ";".join(f"{y}:{v:.0f}" for y, v in usable)
    return sum(annual_yields) / len(annual_yields), warnings, ";".join(f"{y}:{v:.0f}" for y, v in usable)


def eodhd_cash_buyback_yield(symbol: str, market_cap: float | None) -> tuple[float | None, str, list[str]]:
    if not market_cap or market_cap <= 0:
        return None, "", ["missing market cap for cash buyback yield"]
    raw_path = EODHD_DIR / "raw" / safe_name(symbol) / "fundamentals.json"
    if not raw_path.exists():
        return None, "", []
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    records = payload.get("Financials", {}).get("Cash_Flow", {}).get("yearly", {})
    if not isinstance(records, dict):
        return None, "", []
    buyback_fields = [
        "repurchaseOfCapitalStock",
        "repurchasesOfCapitalStock",
        "commonStockRepurchased",
        "purchaseOfStock",
        "salePurchaseOfStock",
        "salePurchaseOfStockNet",
    ]
    rows = [(str(date), row) for date, row in records.items() if isinstance(row, dict)]
    rows.sort(key=lambda item: item[0], reverse=True)
    amounts: list[float] = []
    sources: list[str] = []
    for date, row in rows[:5]:
        value = None
        field_used = None
        lower = {str(k).lower(): k for k in row}
        for field in buyback_fields:
            key = field if field in row else lower.get(field.lower())
            if key is None:
                continue
            number = to_float(row.get(key))
            if number is not None:
                value = number
                field_used = str(key)
                break
        if value is not None and value < 0:
            amounts.append(abs(value))
            sources.append(f"{date}.{field_used}")
        else:
            amounts.append(0.0)
    if not amounts:
        return 0.0, "", ["no cash buyback fields found"]
    return (sum(amounts) / 5.0) / market_cap, ";".join(sources), []


def conservative_buyback_yield(
    market: str,
    symbol: str,
    market_cap: float | None,
    share_count_yield: float | None,
) -> tuple[float, str, list[str]]:
    warnings: list[str] = []
    if market == "KR":
        cash_yield, cash_sources, cash_warnings = eodhd_cash_buyback_yield(symbol, market_cap)
        warnings.extend(cash_warnings)
        if cash_yield is None:
            return max(share_count_yield or 0.0, 0.0), "share_count_only", warnings
        if share_count_yield is None:
            warnings.append("using cash buyback yield without share-count confirmation")
            return max(cash_yield, 0.0), f"EODHD_cash:{cash_sources}", warnings
        if share_count_yield <= 0:
            if cash_yield > 0:
                warnings.append("cash buybacks found but share count did not shrink; conservative buyback yield set to 0")
            return 0.0, f"EODHD_cash:{cash_sources}; share_count_confirmation_negative", warnings
        return min(max(cash_yield, 0.0), share_count_yield), f"EODHD_cash:{cash_sources}; capped_by_share_count", warnings
    return max(share_count_yield or 0.0, 0.0), "share_count_proxy", warnings


def build_row(
    market: str,
    symbol: str,
    name: str,
    description: str,
    sector: str,
    industry: str,
    currency: str,
    known_symbols: set[str] | None,
    include_share_classes: bool,
    refresh: bool,
) -> dict[str, Any] | None:
    if is_share_class(symbol, name, known_symbols):
        if not include_share_classes:
            return None

    ysym = yf_symbol(market, symbol)
    payload, status = fetch_yahoo(ysym, refresh)
    if payload is None:
        return {
            "market": market,
            "symbol": symbol,
            "yahoo_symbol": ysym,
            "name": name,
            "data_quality_warnings": f"Yahoo fetch failed: {status}",
        }

    info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
    market_cap = to_float(info.get("marketCap"))
    current_price = to_float(info.get("currentPrice")) or to_float(info.get("regularMarketPrice")) or to_float(info.get("previousClose"))
    shares_out = to_float(info.get("sharesOutstanding"))
    if current_price is None and market_cap and shares_out:
        current_price = market_cap / shares_out

    warnings: list[str] = []
    div_rows, div_warnings = annual_dividends(payload.get("dividends") or [])
    warnings.extend(div_warnings)
    avg_dps = sum(value for _, value in div_rows) / 5.0 if div_rows else None
    dividend_yield = avg_dps / current_price if avg_dps is not None and current_price and current_price > 0 else None
    if dividend_yield is None:
        warnings.append("missing current price or dividend history for dividend yield")
    elif dividend_yield > 0.15:
        warnings.append("extreme dividend-per-share yield over 15%; likely special dividend, split adjustment, or data issue; verify manually")

    share_count_buyback_yield, buyback_warnings, share_years = buyback_yield_from_shares(payload.get("shares") or [])
    warnings.extend(buyback_warnings)
    buyback_yield, buyback_source, conservative_warnings = conservative_buyback_yield(
        market,
        symbol,
        market_cap,
        share_count_buyback_yield,
    )
    warnings.extend(conservative_warnings)
    shareholder_yield = (dividend_yield or 0.0) + buyback_yield

    if market_cap is None or market_cap <= 0:
        warnings.append("missing or non-positive current market cap")

    return {
        "market": market,
        "symbol": symbol,
        "yahoo_symbol": ysym,
        "name": info.get("longName") or name,
        "company_description": info.get("longBusinessSummary") or description,
        "sector": info.get("sector") or sector,
        "industry": info.get("industry") or industry,
        "currency": info.get("currency") or currency,
        "market_cap": market_cap,
        "current_price": current_price,
        "shares_outstanding": shares_out,
        "avg_annual_dividend_per_share_5y": avg_dps,
        "dividend_yield_5y_avg": dividend_yield,
        "buyback_yield_5y_avg_conservative": buyback_yield,
        "buyback_yield_5y_avg_share_count_raw": share_count_buyback_yield,
        "shareholder_yield_5y_avg_corrected": shareholder_yield,
        "dividend_years_used": ";".join(f"{year}:{value:g}" for year, value in div_rows),
        "share_count_years_used": share_years,
        "buyback_source": buyback_source,
        "data_source": f"Yahoo Finance ({status}); universe from {'JPX' if market == 'JP' else 'EODHD Korea manifest'}",
        "data_quality_warnings": "; ".join(dict.fromkeys(warnings)),
    }


def load_korea_universe() -> tuple[list[dict[str, str]], set[str]]:
    manifest = json.loads((EODHD_DIR / "stock_pull_manifest.json").read_text(encoding="utf-8"))
    known = {item.get("symbol") for item in manifest if isinstance(item, dict) and item.get("symbol")}
    rows: list[dict[str, str]] = []
    for item in manifest:
        if not isinstance(item, dict) or not item.get("symbol"):
            continue
        raw_path = EODHD_DIR / "raw" / safe_name(item["symbol"]) / "fundamentals.json"
        general: dict[str, Any] = {}
        if raw_path.exists():
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            if isinstance(payload.get("General"), dict):
                general = payload["General"]
        rows.append(
            {
                "market": "KR",
                "symbol": item["symbol"],
                "name": general.get("Name") or item.get("name") or item["symbol"],
                "description": general.get("Description") or "",
                "sector": general.get("Sector") or "",
                "industry": general.get("Industry") or "",
                "currency": general.get("CurrencyCode") or item.get("currency") or "KRW",
            }
        )
    return rows, known


def load_japan_universe() -> list[dict[str, str]]:
    import pandas as pd

    df = pd.read_excel(JPX_XLS)
    rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        market_segment = str(row.get("市場・商品区分") or "")
        raw_code = row.get("コード")
        code = str(raw_code or "").strip()
        if isinstance(raw_code, (int, float)) and math.isfinite(raw_code):
            code = str(int(raw_code)).zfill(4)
        if "内国株式" not in market_segment or not re.fullmatch(r"\d{4}", code):
            continue
        rows.append(
            {
                "market": "JP",
                "symbol": f"{code}.T",
                "name": str(row.get("銘柄名") or code),
                "description": "",
                "sector": "",
                "industry": str(row.get("33業種区分") or ""),
                "currency": "JPY",
            }
        )
    rows.sort(key=lambda row: row["symbol"])
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    markets = {value.strip().upper() for value in args.markets.split(",") if value.strip()}
    universe: list[dict[str, str]] = []
    known_symbols: set[str] | None = None
    if "KR" in markets:
        kr_rows, known_symbols = load_korea_universe()
        universe.extend(kr_rows)
    if "JP" in markets:
        universe.extend(load_japan_universe())
    wanted_symbols = {value.strip().upper() for value in args.symbols.split(",") if value.strip()}
    if wanted_symbols:
        universe = [row for row in universe if row["symbol"].upper() in wanted_symbols]
    universe = universe[args.offset :]
    if args.limit is not None:
        universe = universe[: args.limit]

    deadline = time.monotonic() + args.time_budget_sec if args.time_budget_sec > 0 else None
    rows: list[dict[str, Any]] = []
    for item in universe:
        if deadline is not None and time.monotonic() >= deadline:
            break
        row = build_row(
            item["market"],
            item["symbol"],
            item["name"],
            item.get("description", ""),
            item.get("sector", ""),
            item.get("industry", ""),
            item.get("currency", ""),
            known_symbols if item["market"] == "KR" else None,
            args.include_share_classes,
            args.refresh_cache,
        )
        if row is not None:
            rows.append(row)
        time.sleep(args.sleep)

    ranked = [
        row
        for row in rows
        if isinstance(row.get("shareholder_yield_5y_avg_corrected"), (int, float))
        and (not args.min_market_cap or (row.get("market_cap") and row["market_cap"] >= args.min_market_cap))
    ]
    ranked.sort(
        key=lambda row: (
            "missing" not in (row.get("data_quality_warnings") or "").lower(),
            row.get("shareholder_yield_5y_avg_corrected") or -999.0,
            row.get("dividend_yield_5y_avg") or -999.0,
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
        "universe_rows": len(universe),
        "processed_rows": len(rows),
        "ranked_rows": len(ranked),
        "formula": "avg annual dividend per share / current price + avg annual net share-count reduction",
        "outputs": {"all": str(all_path), "top": str(top_path)},
    }
    (OUT_DIR / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
