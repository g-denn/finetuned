#!/usr/bin/env python3
"""Small EODHD validation probe for split-adjusted return labels.

This script intentionally does not store an API key. Set EODHD_API_TOKEN in the
environment before running it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


API_ROOT = "https://eodhd.com/api"


@dataclass(frozen=True)
class PricePoint:
    date: date
    close: float
    adjusted_close: float
    warning: str | None = None


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def request_json(path: str, params: dict[str, str]) -> Any:
    token = os.environ.get("EODHD_API_TOKEN")
    if not token:
        raise SystemExit("Set EODHD_API_TOKEN before running this script.")

    safe_params = {"fmt": "json", "api_token": token}
    safe_params.update(params)
    url = f"{API_ROOT}/{path}?{urllib.parse.urlencode(safe_params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "finetuned-eodhd-probe/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"EODHD HTTP {exc.code} for {path}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"EODHD network error for {path}: {exc}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        preview = body[:500].decode("utf-8", errors="replace")
        raise RuntimeError(f"EODHD returned non-JSON for {path}: {preview}") from exc


def load_prices(symbol: str, start: date, end: date) -> tuple[list[PricePoint], list[str]]:
    raw = request_json(
        f"eod/{symbol}",
        {"from": start.isoformat(), "to": end.isoformat(), "period": "d"},
    )
    warnings: list[str] = []
    if isinstance(raw, dict):
        warning = raw.get("warning") or raw.get("message")
        return [], [str(warning or raw)]
    if not isinstance(raw, list):
        return [], [f"unexpected price payload: {type(raw).__name__}"]

    prices: list[PricePoint] = []
    for item in raw:
        if not isinstance(item, dict):
            warnings.append(f"ignored non-object price row: {item!r}")
            continue
        warning = item.get("warning")
        if warning:
            warnings.append(str(warning))
            if "date" not in item:
                continue
        try:
            prices.append(
                PricePoint(
                    date=parse_date(str(item["date"])),
                    close=float(item["close"]),
                    adjusted_close=float(item["adjusted_close"]),
                    warning=str(warning) if warning else None,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"ignored malformed price row: {exc}: {item!r}")
    return sorted(prices, key=lambda p: p.date), warnings


def load_actions(endpoint: str, symbol: str, start: date, end: date) -> list[dict[str, Any]]:
    raw = request_json(
        f"{endpoint}/{symbol}",
        {"from": start.isoformat(), "to": end.isoformat()},
    )
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def first_on_or_after(prices: list[PricePoint], target: date) -> PricePoint | None:
    for point in prices:
        if point.date >= target:
            return point
    return None


def last_on_or_before(prices: list[PricePoint], target: date) -> PricePoint | None:
    for point in reversed(prices):
        if point.date <= target:
            return point
    return None


def print_probe(symbol: str, start_s: str, end_s: str) -> int:
    start = parse_date(start_s)
    end = parse_date(end_s)
    prices, warnings = load_prices(symbol, start, end)
    splits = load_actions("splits", symbol, start, end)
    dividends = load_actions("div", symbol, start, end)

    print(f"symbol: {symbol}")
    print(f"requested_window: {start} -> {end}")
    print(f"price_rows: {len(prices)}")
    for warning in sorted(set(warnings)):
        print(f"warning: {warning}")

    if not prices:
        print("status: no_usable_price_history")
        print("return_adjusted_close: unavailable")
        return 2

    start_point = first_on_or_after(prices, start)
    end_point = last_on_or_before(prices, end)
    if not start_point or not end_point or end_point.date < start_point.date:
        print("status: insufficient_price_window")
        print("return_adjusted_close: unavailable")
        return 2

    adjusted_return = end_point.adjusted_close / start_point.adjusted_close
    raw_return = end_point.close / start_point.close
    adjustment_ratio_start = start_point.adjusted_close / start_point.close
    adjustment_ratio_end = end_point.adjusted_close / end_point.close

    print(f"actual_price_window: {start_point.date} -> {end_point.date}")
    print(f"start_close: {start_point.close:.6f}")
    print(f"start_adjusted_close: {start_point.adjusted_close:.6f}")
    print(f"end_close: {end_point.close:.6f}")
    print(f"end_adjusted_close: {end_point.adjusted_close:.6f}")
    print(f"return_raw_close: {raw_return:.6f}x")
    print(f"return_adjusted_close: {adjusted_return:.6f}x")
    print(f"adjustment_ratio_start: {adjustment_ratio_start:.8f}")
    print(f"adjustment_ratio_end: {adjustment_ratio_end:.8f}")
    print(f"splits_detected: {len(splits)}")
    for split in splits:
        print(f"  split: {split.get('date')} {split.get('split')}")
    print(f"dividends_detected: {len(dividends)}")
    if dividends:
        print(f"  first_dividend: {dividends[0]}")
        print(f"  last_dividend: {dividends[-1]}")

    terminal_before_requested_end = end_point.date < end
    start_after_requested_start = start_point.date > start
    if start_after_requested_start:
        print("flag: first available price is after requested start date")
    if terminal_before_requested_end:
        print("flag: price history ends before requested endpoint; model delisting/acquisition/bankruptcy instead of using fake future price")

    if terminal_before_requested_end or start_after_requested_start:
        print("status: usable_price_history_requires_corporate_action_review")
        return 1
    if warnings:
        print("status: usable_but_provider_warning")
        return 1
    print("status: usable")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True, help="EODHD symbol, e.g. AAPL.US")
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    args = parser.parse_args()
    return print_probe(args.symbol, args.from_date, args.to_date)


if __name__ == "__main__":
    sys.exit(main())
