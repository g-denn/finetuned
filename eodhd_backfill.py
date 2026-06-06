#!/usr/bin/env python3
"""EODHD historical cache + return-label probe.

The script is intentionally provider-cache-first:

1. Fetch each unique SYMBOL.EXCHANGE once for the whole needed range.
2. Fetch splits/dividends once for the same range.
3. Optionally fetch EODHD delisted archives once per exchange and attach the
   provider's delisted symbol record to matching rows.
4. Optionally fetch full EODHD Fundamentals v1.1 once per unique symbol and
   cache the raw JSON for business-reality review.
5. Calculate 1y/3y/5y/10y/20y multipliers locally from adjusted_close.
6. Flag rows that need corporate-action review instead of pretending prices exist.

Set EODHD_API_TOKEN in the environment. Do not commit API keys.

EODHD's own skill docs note two important identity limitations:
- Search API returns active tickers only.
- Symbol Change History starts from 2022-07-22.

Old VIC-style rows therefore still need a separate lineage evidence file for
pre-2022 ticker changes, acquisitions, bankruptcies, and ticker reuse.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import hashlib
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


API_ROOT = "https://eodhd.com/api"
HORIZONS = (1, 3, 5, 10, 20)
PRIMARY_HORIZONS = (1, 3, 5)

API_TOKEN_RE = re.compile(r"api_token=[^&\s'\"\\)]+")
COMMON_STOCK_TYPES = {"common stock", "common share", "ordinary share", "ordinary shares"}


SAMPLE_10 = [
    {
        "idea_id": "sample-aapl-2020",
        "raw_symbol": "AAPL",
        "eodhd_symbol": "AAPL.US",
        "company_name": "Apple Inc.",
        "publication_date": "2020-08-28",
    },
    {
        "idea_id": "sample-etsy-2016",
        "raw_symbol": "ETSY",
        "eodhd_symbol": "ETSY.US",
        "company_name": "Etsy Inc.",
        "publication_date": "2016-02-16",
    },
    {
        "idea_id": "sample-gigm-2002",
        "raw_symbol": "GIGM",
        "eodhd_symbol": "GIGM.US",
        "company_name": "GigaMedia Limited",
        "publication_date": "2002-02-19",
    },
    {
        "idea_id": "sample-ksw-2008",
        "raw_symbol": "KSW",
        "eodhd_symbol": "KSW.US",
        "company_name": "KSW Inc.",
        "publication_date": "2008-12-19",
    },
    {
        "idea_id": "sample-mhh-2008",
        "raw_symbol": "MHH",
        "eodhd_symbol": "MHH.US",
        "company_name": "Mastech Digital Inc.",
        "publication_date": "2008-12-15",
    },
    {
        "idea_id": "sample-sfi-2009",
        "raw_symbol": "SFI",
        "eodhd_symbol": "SFI.US",
        "company_name": "iStar Financial Inc.",
        "publication_date": "2009-02-26",
    },
    {
        "idea_id": "sample-star-2015",
        "raw_symbol": "STAR",
        "eodhd_symbol": "STAR.US",
        "company_name": "iStar Inc.",
        "publication_date": "2015-04-06",
    },
    {
        "idea_id": "sample-avxs-2018",
        "raw_symbol": "AVXS",
        "eodhd_symbol": "AVXS.US",
        "company_name": "AveXis Inc.",
        "publication_date": "2018-01-02",
    },
    {
        "idea_id": "sample-pq-2006",
        "raw_symbol": "PQ",
        "eodhd_symbol": "PQ.US",
        "company_name": "PetroQuest Energy Inc.",
        "publication_date": "2006-02-11",
    },
    {
        "idea_id": "sample-twtr-2017",
        "raw_symbol": "TWTR",
        "eodhd_symbol": "TWTR.US",
        "company_name": "Twitter Inc.",
        "publication_date": "2017-01-03",
    },
]


@dataclass(frozen=True)
class PricePoint:
    date: date
    close: float
    adjusted_close: float
    raw: dict[str, Any]


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def add_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year + years)


def require_token() -> str:
    token = os.environ.get("EODHD_API_TOKEN")
    if not token:
        raise SystemExit("Set EODHD_API_TOKEN before running this script.")
    return token


def redact_api_token(message: str) -> str:
    return API_TOKEN_RE.sub("api_token=<redacted>", message)


def encode_api_path(path: str) -> str:
    return "/".join(urllib.parse.quote(part, safe=".-_") for part in path.split("/"))


def eodhd_exchange_from_symbol(symbol: str | None) -> str | None:
    if not symbol or "." not in symbol:
        return "US" if symbol else None
    return symbol.rsplit(".", 1)[1].upper()


def eodhd_code_from_symbol(symbol: str | None) -> str | None:
    if not symbol:
        return None
    code = symbol.rsplit(".", 1)[0] if "." in symbol else symbol
    code = code.strip().upper()
    return code or None


def safe_cache_token(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value)


def row_value(row: dict[str, Any], *names: str) -> str | None:
    lowered = {key.lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def delisted_cache_path(out_dir: Path, exchange: str) -> Path:
    if exchange.upper() == "US":
        return out_dir / "delisted_symbols_us.json"
    return out_dir / f"delisted_symbols_{safe_cache_token(exchange)}.json"


def delisted_row_code(row: dict[str, Any]) -> str | None:
    value = row_value(row, "Code", "code", "Symbol", "symbol")
    return value.upper() if value else None


def delisted_row_exchange(row: dict[str, Any], fallback_exchange: str) -> str:
    value = row_value(row, "Exchange", "ExchangeCode", "exchange_code", "exchange")
    return (value or fallback_exchange).upper()


def qualified_delisted_symbol(row: dict[str, Any], fallback_exchange: str) -> str | None:
    code = delisted_row_code(row)
    if not code:
        return None
    return f"{code}.{delisted_row_exchange(row, fallback_exchange)}"


def build_delisted_index(rows_by_exchange: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for exchange, rows in rows_by_exchange.items():
        exchange = exchange.upper()
        for row in rows:
            if not isinstance(row, dict):
                continue
            qualified = qualified_delisted_symbol(row, exchange)
            code = delisted_row_code(row)
            if qualified:
                index[qualified.upper()] = row
            if code and exchange == "US":
                index.setdefault(code.upper(), row)
    return index


def delisted_lookup_keys(raw_symbol: str | None, eodhd_symbol: str | None) -> list[str]:
    keys: list[str] = []
    if eodhd_symbol:
        keys.append(eodhd_symbol.upper())
    code = eodhd_code_from_symbol(eodhd_symbol)
    exchange = eodhd_exchange_from_symbol(eodhd_symbol)
    if code and exchange:
        keys.append(f"{code}.{exchange}".upper())
        if exchange == "US":
            keys.append(code.upper())
    raw = (raw_symbol or "").strip().upper()
    if raw:
        keys.append(raw)
        if exchange and "." not in raw and " " not in raw:
            keys.append(f"{raw}.{exchange}".upper())
    return list(dict.fromkeys(keys))


def find_delisted_record(
    raw_symbol: str | None,
    eodhd_symbol: str | None,
    delisted_index: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for key in delisted_lookup_keys(raw_symbol, eodhd_symbol):
        record = delisted_index.get(key)
        if record:
            return record
    return None


def delisted_metadata(
    raw_symbol: str | None,
    eodhd_symbol: str | None,
    delisted_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    record = find_delisted_record(raw_symbol, eodhd_symbol, delisted_index)
    return {
        "is_in_delisted_cache": record is not None,
        "delisted_provider_code": delisted_row_code(record) if record else None,
        "delisted_provider_exchange": delisted_row_exchange(record, eodhd_exchange_from_symbol(eodhd_symbol) or "US")
        if record
        else None,
        "delisted_provider_name": row_value(record, "Name", "name") if record else None,
        "delisted_provider_type": row_value(record, "Type", "type") if record else None,
        "delisted_provider_isin": row_value(record, "Isin", "ISIN", "isin") if record else None,
        "delisted_provider_record": record,
    }


def load_delisted_symbol_lists(
    out_dir: Path,
    exchanges: list[str],
    refresh: bool,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_exchange: dict[str, list[dict[str, Any]]] = {}
    for exchange in exchanges:
        exchange = exchange.upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9]{0,9}", exchange):
            print(f"delisted {exchange} skipped: unsupported_exchange_code", file=sys.stderr)
            continue
        cache_path = delisted_cache_path(out_dir, exchange)
        if cache_path.exists() and not refresh:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            rows_by_exchange[exchange] = [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []
            continue

        try:
            payload, warnings = request_json(f"exchange-symbol-list/{exchange}", {"delisted": "1"})
            if warnings:
                print(f"delisted {exchange} warnings: {warnings}", file=sys.stderr)
            rows = [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []
            rows_by_exchange[exchange] = rows
            cache_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - archive availability is exchange-specific.
            error_message = redact_api_token(str(exc))
            print(f"delisted {exchange} failed: {error_message}", file=sys.stderr)
            (out_dir / f"delisted_symbols_{safe_cache_token(exchange)}.error.json").write_text(
                json.dumps(
                    {"exchange": exchange, "error": error_message, "failed_at": datetime.utcnow().isoformat() + "Z"},
                    indent=2,
                ),
                encoding="utf-8",
            )
    return rows_by_exchange


def delisted_exchanges_for_ideas(ideas: list[dict[str, str]]) -> list[str]:
    exchanges = {
        exchange
        for exchange in (eodhd_exchange_from_symbol(idea.get("eodhd_symbol")) for idea in ideas)
        if exchange and re.fullmatch(r"[A-Z][A-Z0-9]{0,9}", exchange)
    }
    return sorted(exchanges)


def request_json(path: str, params: dict[str, str], retries: int = 2) -> tuple[Any, list[str]]:
    token = require_token()
    all_params = {"fmt": "json", "api_token": token}
    all_params.update(params)
    url = f"{API_ROOT}/{encode_api_path(path)}?{urllib.parse.urlencode(all_params)}"
    warnings: list[str] = []

    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": "finetuned-eodhd-backfill/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            payload = json.loads(raw)
            return payload, warnings
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < retries:
                sleep_seconds = 2 + attempt * 3
                warnings.append(f"rate_limited_retry_{sleep_seconds}s")
                time.sleep(sleep_seconds)
                continue
            raise RuntimeError(f"EODHD HTTP {exc.code} for {path}: {redact_api_token(body[:600])}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < retries:
                sleep_seconds = 2 + attempt * 3
                warnings.append(f"network_retry_{sleep_seconds}s")
                time.sleep(sleep_seconds)
                continue
            raise RuntimeError(f"EODHD network error for {path}: {redact_api_token(str(exc))}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"EODHD returned non-JSON for {path}") from exc

    raise AssertionError("unreachable")


def load_prices(symbol: str, start: date, end: date) -> tuple[list[PricePoint], list[str]]:
    payload, warnings = request_json(
        f"eod/{symbol}",
        {"from": start.isoformat(), "to": end.isoformat(), "period": "d"},
    )
    if isinstance(payload, dict):
        return [], warnings + [str(payload.get("warning") or payload.get("message") or payload)]
    if not isinstance(payload, list):
        return [], warnings + [f"unexpected_price_payload:{type(payload).__name__}"]

    prices: list[PricePoint] = []
    for row in payload:
        if not isinstance(row, dict):
            warnings.append("ignored_non_object_price_row")
            continue
        if row.get("warning"):
            warnings.append(str(row["warning"]))
            if "date" not in row:
                continue
        try:
            close = float(row["close"])
            adjusted_close = float(row["adjusted_close"])
            prices.append(
                PricePoint(
                    date=parse_date(str(row["date"])) or date.min,
                    close=close,
                    adjusted_close=adjusted_close,
                    raw=row,
                )
            )
        except (KeyError, TypeError, ValueError):
            warnings.append(f"ignored_malformed_price_row:{row!r}")
    return sorted(prices, key=lambda point: point.date), warnings


def load_actions(endpoint: str, symbol: str, start: date, end: date) -> tuple[list[dict[str, Any]], list[str]]:
    payload, warnings = request_json(
        f"{endpoint}/{symbol}",
        {"from": start.isoformat(), "to": end.isoformat()},
    )
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)], warnings
    if isinstance(payload, dict) and payload.get("warning"):
        warnings.append(str(payload["warning"]))
    return [], warnings


def cache_path(cache_dir: Path, symbol: str) -> Path:
    safe_symbol = "".join(char if char.isalnum() or char in ".-" else "_" for char in symbol)
    return cache_dir / f"{safe_symbol}.json"


def error_cache_path(cache_dir: Path, symbol: str) -> Path:
    safe_symbol = "".join(char if char.isalnum() or char in ".-" else "_" for char in symbol)
    return cache_dir / f"{safe_symbol}.error.json"


def fundamentals_cache_path(cache_dir: Path, symbol: str) -> Path:
    return cache_path(cache_dir, symbol)


def bulk_fundamentals_cache_path(cache_dir: Path, exchange: str, symbols: list[str]) -> Path:
    digest = hashlib.sha1(",".join(sorted(symbols)).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{safe_cache_token(exchange)}_{digest}.json"


def load_fundamentals_bundle(symbol: str, cache_dir: Path, refresh: bool, retry_errors: bool = False) -> tuple[str, dict[str, Any]]:
    path = fundamentals_cache_path(cache_dir, symbol)
    if path.exists() and not refresh:
        return symbol, json.loads(path.read_text(encoding="utf-8"))
    err_path = error_cache_path(cache_dir, symbol)
    if err_path.exists() and not refresh:
        cached_error = json.loads(err_path.read_text(encoding="utf-8"))
        error_message = str(cached_error.get("error", "cached_provider_error"))
        if not retry_errors or not retryable_fundamentals_error(error_message):
            return symbol, {"symbol": symbol, "cached_error": error_message}

    payload, warnings = request_json(f"v1.1/fundamentals/{symbol}", {})
    bundle = {
        "symbol": symbol,
        "endpoint": "v1.1/fundamentals",
        "warnings": warnings,
        "payload": payload,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    }
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    if err_path.exists():
        err_path.unlink()
    return symbol, bundle


def retryable_fundamentals_error(error_message: str) -> bool:
    lowered = error_message.lower()
    return "bulk-fundamentals" in lowered or "forbidden" in lowered or "payment required" in lowered


def response_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [row for _, row in sorted(payload.items(), key=lambda item: str(item[0])) if isinstance(row, dict)]
    return []


def symbol_from_fundamentals_record(record: dict[str, Any], requested_symbols: list[str]) -> str | None:
    general = record.get("General") if isinstance(record.get("General"), dict) else {}
    primary = row_value(general, "PrimaryTicker", "primary_ticker")
    if primary:
        return primary.upper()
    code = row_value(general, "Code", "code")
    if code:
        code = code.upper()
        matches = [symbol for symbol in requested_symbols if eodhd_code_from_symbol(symbol) == code]
        if len(matches) == 1:
            return matches[0]
    return None


def load_bulk_fundamentals_chunk(
    exchange: str,
    symbols: list[str],
    bulk_cache_dir: Path,
    symbol_cache_dir: Path,
    refresh: bool,
) -> dict[str, dict[str, Any]]:
    bulk_path = bulk_fundamentals_cache_path(bulk_cache_dir, exchange, symbols)
    if bulk_path.exists() and not refresh:
        bulk_bundle = json.loads(bulk_path.read_text(encoding="utf-8"))
    else:
        payload, warnings = request_json(
            f"v1.1/bulk-fundamentals/{exchange}",
            {
                "symbols": ",".join(symbols),
                "limit": str(min(len(symbols), 500)),
                "offset": "0",
                "version": "1.2",
            },
        )
        bulk_bundle = {
            "exchange": exchange,
            "symbols": symbols,
            "endpoint": "v1.1/bulk-fundamentals",
            "warnings": warnings,
            "payload": payload,
            "fetched_at": datetime.utcnow().isoformat() + "Z",
        }
        bulk_path.write_text(json.dumps(bulk_bundle, indent=2, sort_keys=True), encoding="utf-8")

    records_by_symbol: dict[str, dict[str, Any]] = {}
    records = response_records(bulk_bundle.get("payload"))
    for idx, record in enumerate(records):
        symbol = symbol_from_fundamentals_record(record, symbols)
        if symbol is None and idx < len(symbols):
            symbol = symbols[idx]
        if symbol is None:
            continue
        bundle = {
            "symbol": symbol,
            "endpoint": "v1.1/bulk-fundamentals",
            "bulk_exchange": exchange,
            "warnings": bulk_bundle.get("warnings") or [],
            "payload": record,
            "fetched_at": bulk_bundle.get("fetched_at"),
        }
        fundamentals_cache_path(symbol_cache_dir, symbol).write_text(
            json.dumps(bundle, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        records_by_symbol[symbol] = bundle

    missing = sorted(set(symbols) - set(records_by_symbol))
    for symbol in missing:
        error_message = "bulk_fundamentals_missing_symbol"
        bundle = {"symbol": symbol, "cached_error": error_message}
        error_cache_path(symbol_cache_dir, symbol).write_text(
            json.dumps(
                {"symbol": symbol, "error": error_message, "failed_at": datetime.utcnow().isoformat() + "Z"},
                indent=2,
            ),
            encoding="utf-8",
        )
        records_by_symbol[symbol] = bundle
    return records_by_symbol


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def load_fundamentals_bundles_bulk(
    symbols: list[str],
    symbol_cache_dir: Path,
    bulk_cache_dir: Path,
    refresh: bool,
    workers: int,
) -> dict[str, dict[str, Any]]:
    symbol_cache_dir.mkdir(parents=True, exist_ok=True)
    bulk_cache_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        symbol
        for symbol in sorted(set(symbols))
        if refresh or not fundamentals_cache_path(symbol_cache_dir, symbol).exists()
    ]
    bundles: dict[str, dict[str, Any]] = {}
    for symbol in sorted(set(symbols) - set(pending)):
        bundles[symbol] = json.loads(fundamentals_cache_path(symbol_cache_dir, symbol).read_text(encoding="utf-8"))

    grouped: dict[str, list[str]] = {}
    for symbol in pending:
        exchange = eodhd_exchange_from_symbol(symbol) or "US"
        if not re.fullmatch(r"[A-Z][A-Z0-9]{0,9}", exchange):
            bundles[symbol] = {"symbol": symbol, "cached_error": "unsupported_exchange_code_for_bulk_fundamentals"}
            continue
        grouped.setdefault(exchange, []).append(symbol)

    tasks: list[tuple[str, list[str]]] = []
    for exchange, exchange_symbols in grouped.items():
        for chunk in chunked(exchange_symbols, 500):
            tasks.append((exchange, chunk))

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                load_bulk_fundamentals_chunk,
                exchange,
                chunk,
                bulk_cache_dir,
                symbol_cache_dir,
                refresh,
            ): (exchange, chunk)
            for exchange, chunk in tasks
        }
        for future in as_completed(futures):
            exchange, chunk = futures[future]
            completed += 1
            try:
                bundles.update(future.result())
            except Exception as exc:  # noqa: BLE001 - preserve provider/network failures per chunk.
                error_message = redact_api_token(str(exc))
                print(f"bulk_fundamentals {exchange} failed for {len(chunk)} symbols: {error_message}", file=sys.stderr)
                for symbol in chunk:
                    bundles[symbol] = {"symbol": symbol, "cached_error": error_message}
                    error_cache_path(symbol_cache_dir, symbol).write_text(
                        json.dumps(
                            {"symbol": symbol, "error": error_message, "failed_at": datetime.utcnow().isoformat() + "Z"},
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
            if completed % 10 == 0 or completed == len(futures):
                print(f"fetched_bulk_fundamentals_chunks={completed}/{len(futures)} symbols={len(bundles)}/{len(set(symbols))}", file=sys.stderr)
    return bundles


def load_fundamentals_bundles_single(
    symbols: list[str],
    cache_dir: Path,
    refresh: bool,
    retry_errors: bool,
    workers: int,
) -> dict[str, dict[str, Any]]:
    bundles: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(load_fundamentals_bundle, symbol, cache_dir, refresh, retry_errors): symbol
            for symbol in sorted(set(symbols))
        }
        completed = 0
        for future in as_completed(futures):
            symbol = futures[future]
            completed += 1
            try:
                _, bundle = future.result()
                bundles[symbol] = bundle
            except Exception as exc:  # noqa: BLE001 - preserve provider/network failures per symbol.
                error_message = redact_api_token(str(exc))
                bundles[symbol] = {"symbol": symbol, "cached_error": error_message}
                error_cache_path(cache_dir, symbol).write_text(
                    json.dumps(
                        {"symbol": symbol, "error": error_message, "failed_at": datetime.utcnow().isoformat() + "Z"},
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if completed % 100 == 0 or completed == len(futures):
                print(f"fetched_fundamentals={completed}/{len(futures)}", file=sys.stderr)
    return bundles


def mapping_values(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, dict)]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def dated_mapping_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return sorted(str(key) for key in value if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(key)))
    return []


def financial_section_summary(financials: dict[str, Any], section: str) -> dict[str, Any]:
    payload = financials.get(section) if isinstance(financials, dict) else None
    if not isinstance(payload, dict):
        return {
            "yearly_count": 0,
            "quarterly_count": 0,
            "yearly_first_date": None,
            "yearly_last_date": None,
            "quarterly_first_date": None,
            "quarterly_last_date": None,
        }
    yearly_keys = dated_mapping_keys(payload.get("yearly"))
    quarterly_keys = dated_mapping_keys(payload.get("quarterly"))
    return {
        "yearly_count": len(yearly_keys),
        "quarterly_count": len(quarterly_keys),
        "yearly_first_date": yearly_keys[0] if yearly_keys else None,
        "yearly_last_date": yearly_keys[-1] if yearly_keys else None,
        "quarterly_first_date": quarterly_keys[0] if quarterly_keys else None,
        "quarterly_last_date": quarterly_keys[-1] if quarterly_keys else None,
    }


def latest_report_rows(financials: dict[str, Any], section: str, period: str) -> tuple[str | None, dict[str, Any] | None]:
    payload = financials.get(section) if isinstance(financials, dict) else None
    rows = payload.get(period) if isinstance(payload, dict) else None
    keys = dated_mapping_keys(rows)
    if not keys or not isinstance(rows, dict):
        return None, None
    key = keys[-1]
    row = rows.get(key)
    return key, row if isinstance(row, dict) else None


def first_last_numeric(rows: Any, *field_names: str) -> tuple[float | None, float | None]:
    if not isinstance(rows, dict):
        return None, None
    keys = dated_mapping_keys(rows)
    values: list[float] = []
    for key in keys:
        row = rows.get(key)
        if not isinstance(row, dict):
            continue
        lowered = {str(k).lower(): v for k, v in row.items()}
        parsed: float | None = None
        for field in field_names:
            parsed = parse_number(lowered.get(field.lower()))
            if parsed is not None:
                break
        if parsed is not None:
            values.append(parsed)
    if not values:
        return None, None
    return values[0], values[-1]


def parse_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in (float("inf"), float("-inf")) else None


def extract_fundamentals_summary(symbol: str, bundle: dict[str, Any] | None) -> dict[str, Any]:
    if not bundle:
        return {"symbol": symbol, "fundamentals_status": "not_fetched"}
    if bundle.get("cached_error"):
        return {
            "symbol": symbol,
            "fundamentals_status": "provider_error",
            "fundamentals_error": redact_api_token(str(bundle.get("cached_error"))),
        }

    payload = bundle.get("payload")
    if not isinstance(payload, dict):
        return {
            "symbol": symbol,
            "fundamentals_status": "provider_error",
            "fundamentals_error": f"unexpected_payload:{type(payload).__name__}",
        }
    general = payload.get("General") if isinstance(payload.get("General"), dict) else {}
    highlights = payload.get("Highlights") if isinstance(payload.get("Highlights"), dict) else {}
    valuation = payload.get("Valuation") if isinstance(payload.get("Valuation"), dict) else {}
    splits_dividends = payload.get("SplitsDividends") if isinstance(payload.get("SplitsDividends"), dict) else {}
    financials = payload.get("Financials") if isinstance(payload.get("Financials"), dict) else {}

    income_summary = financial_section_summary(financials, "Income_Statement")
    balance_summary = financial_section_summary(financials, "Balance_Sheet")
    cash_flow_summary = financial_section_summary(financials, "Cash_Flow")
    income = financials.get("Income_Statement") if isinstance(financials, dict) else {}
    yearly_income = income.get("yearly") if isinstance(income, dict) else {}
    quarterly_income = income.get("quarterly") if isinstance(income, dict) else {}
    latest_yearly_income_date, latest_yearly_income = latest_report_rows(financials, "Income_Statement", "yearly")
    latest_quarterly_income_date, latest_quarterly_income = latest_report_rows(financials, "Income_Statement", "quarterly")
    yearly_revenue_first, yearly_revenue_last = first_last_numeric(
        yearly_income,
        "totalRevenue",
        "total_revenue",
        "revenue",
    )
    yearly_net_income_first, yearly_net_income_last = first_last_numeric(
        yearly_income,
        "netIncome",
        "net_income",
    )
    quarterly_revenue_first, quarterly_revenue_last = first_last_numeric(
        quarterly_income,
        "totalRevenue",
        "total_revenue",
        "revenue",
    )
    quarterly_net_income_first, quarterly_net_income_last = first_last_numeric(
        quarterly_income,
        "netIncome",
        "net_income",
    )

    summary = {
        "symbol": symbol,
        "fundamentals_status": "fetched",
        "fundamentals_warnings": bundle.get("warnings") or [],
        "fundamentals_code": general.get("Code"),
        "fundamentals_type": general.get("Type"),
        "fundamentals_name": general.get("Name"),
        "fundamentals_exchange": general.get("Exchange"),
        "fundamentals_currency": general.get("CurrencyCode"),
        "fundamentals_country": general.get("CountryName"),
        "fundamentals_country_iso": general.get("CountryISO"),
        "fundamentals_isin": general.get("ISIN"),
        "fundamentals_primary_ticker": general.get("PrimaryTicker"),
        "fundamentals_cik": general.get("CIK"),
        "fundamentals_ipo_date": general.get("IPODate"),
        "fundamentals_sector": general.get("Sector"),
        "fundamentals_industry": general.get("Industry"),
        "fundamentals_home_category": general.get("HomeCategory"),
        "fundamentals_is_delisted": general.get("IsDelisted"),
        "fundamentals_delisted_date": general.get("DelistedDate"),
        "fundamentals_market_cap": parse_number(highlights.get("MarketCapitalization") or general.get("MarketCap")),
        "fundamentals_revenue_ttm": parse_number(highlights.get("RevenueTTM")),
        "fundamentals_ebitda": parse_number(highlights.get("EBITDA")),
        "fundamentals_gross_profit_ttm": parse_number(highlights.get("GrossProfitTTM") or highlights.get("GrossProfit")),
        "fundamentals_profit_margin": parse_number(highlights.get("ProfitMargin")),
        "fundamentals_operating_margin_ttm": parse_number(highlights.get("OperatingMarginTTM")),
        "fundamentals_return_on_equity_ttm": parse_number(highlights.get("ReturnOnEquityTTM")),
        "fundamentals_pe_ratio": parse_number(highlights.get("PERatio") or valuation.get("TrailingPE")),
        "fundamentals_price_sales_ttm": parse_number(valuation.get("PriceSalesTTM")),
        "fundamentals_last_split_factor": splits_dividends.get("LastSplitFactor"),
        "fundamentals_last_split_date": splits_dividends.get("LastSplitDate"),
        "fundamentals_latest_yearly_income_date": latest_yearly_income_date,
        "fundamentals_latest_quarterly_income_date": latest_quarterly_income_date,
        "fundamentals_yearly_revenue_first": yearly_revenue_first,
        "fundamentals_yearly_revenue_last": yearly_revenue_last,
        "fundamentals_yearly_net_income_first": yearly_net_income_first,
        "fundamentals_yearly_net_income_last": yearly_net_income_last,
        "fundamentals_quarterly_revenue_first": quarterly_revenue_first,
        "fundamentals_quarterly_revenue_last": quarterly_revenue_last,
        "fundamentals_quarterly_net_income_first": quarterly_net_income_first,
        "fundamentals_quarterly_net_income_last": quarterly_net_income_last,
        "fundamentals_financials": {
            "income_statement": income_summary,
            "balance_sheet": balance_summary,
            "cash_flow": cash_flow_summary,
        },
        "fundamentals_has_financials": any(
            section["yearly_count"] or section["quarterly_count"]
            for section in (income_summary, balance_summary, cash_flow_summary)
        ),
        "fundamentals_latest_yearly_income": latest_yearly_income,
        "fundamentals_latest_quarterly_income": latest_quarterly_income,
    }
    return summary


def output_fundamentals_summary_csv(summaries: dict[str, dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "symbol",
        "fundamentals_status",
        "fundamentals_code",
        "fundamentals_type",
        "fundamentals_name",
        "fundamentals_exchange",
        "fundamentals_currency",
        "fundamentals_country_iso",
        "fundamentals_isin",
        "fundamentals_primary_ticker",
        "fundamentals_cik",
        "fundamentals_ipo_date",
        "fundamentals_sector",
        "fundamentals_industry",
        "fundamentals_home_category",
        "fundamentals_is_delisted",
        "fundamentals_delisted_date",
        "fundamentals_market_cap",
        "fundamentals_revenue_ttm",
        "fundamentals_ebitda",
        "fundamentals_profit_margin",
        "fundamentals_return_on_equity_ttm",
        "fundamentals_last_split_factor",
        "fundamentals_last_split_date",
        "fundamentals_latest_yearly_income_date",
        "fundamentals_latest_quarterly_income_date",
        "fundamentals_yearly_revenue_first",
        "fundamentals_yearly_revenue_last",
        "fundamentals_yearly_net_income_first",
        "fundamentals_yearly_net_income_last",
        "fundamentals_has_financials",
        "fundamentals_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in sorted(summaries.values(), key=lambda item: str(item.get("symbol") or "")):
            writer.writerow({key: summary.get(key) for key in fieldnames})


def load_symbol_bundle(symbol: str, symbol_ideas: list[dict[str, str]], cache_dir: Path, refresh: bool) -> tuple[str, dict[str, Any]]:
    path = cache_path(cache_dir, symbol)
    if path.exists() and not refresh:
        return symbol, json.loads(path.read_text(encoding="utf-8"))
    err_path = error_cache_path(cache_dir, symbol)
    if err_path.exists() and not refresh:
        cached_error = json.loads(err_path.read_text(encoding="utf-8"))
        return symbol, {"cached_error": cached_error.get("error", "cached_provider_error")}

    min_pub = min(parse_date(idea["publication_date"]) for idea in symbol_ideas)
    if min_pub is None:
        bundle = {"error": "missing_min_publication_date"}
        path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
        return symbol, bundle
    max_end = min(
        date.today(),
        max(add_years(parse_date(idea["publication_date"]) or date.today(), 20) for idea in symbol_ideas),
    )
    start = min_pub - timedelta(days=7)
    end = max_end + timedelta(days=7)
    prices, price_warnings = load_prices(symbol, start, end)
    splits, split_warnings = load_actions("splits", symbol, start, end)
    dividends, dividend_warnings = load_actions("div", symbol, start, end)
    bundle = {
        "symbol": symbol,
        "from": start.isoformat(),
        "to": end.isoformat(),
        "price_warnings": price_warnings,
        "split_warnings": split_warnings,
        "dividend_warnings": dividend_warnings,
        "prices": [point.raw for point in prices],
        "splits": splits,
        "dividends": dividends,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    }
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    return symbol, bundle


def bundle_to_prices(bundle: dict[str, Any]) -> tuple[list[PricePoint], list[str]]:
    warnings = list(bundle.get("price_warnings") or [])
    prices: list[PricePoint] = []
    for row in bundle.get("prices") or []:
        if not isinstance(row, dict):
            warnings.append("ignored_non_object_cached_price_row")
            continue
        try:
            prices.append(
                PricePoint(
                    date=parse_date(str(row["date"])) or date.min,
                    close=float(row["close"]),
                    adjusted_close=float(row["adjusted_close"]),
                    raw=row,
                )
            )
        except (KeyError, TypeError, ValueError):
            warnings.append(f"ignored_malformed_cached_price_row:{row!r}")
    return sorted(prices, key=lambda point: point.date), warnings


def first_on_or_after(prices: list[PricePoint], target: date) -> PricePoint | None:
    for point in prices:
        if point.date >= target:
            return point
    return None


def adjusted_values_in_window(prices: list[PricePoint], start: date, end: date) -> list[float]:
    return [point.adjusted_close for point in prices if start <= point.date <= end]


def calculate_row(
    idea: dict[str, str],
    prices: list[PricePoint],
    splits: list[dict[str, Any]],
    dividends: list[dict[str, Any]],
    delisted_index: dict[str, dict[str, Any]],
    fundamentals_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pub_date = parse_date(idea["publication_date"])
    if pub_date is None:
        raise ValueError("publication_date is required")
    symbol = idea["eodhd_symbol"]
    raw_symbol = idea["raw_symbol"].upper()
    start = first_on_or_after(prices, pub_date)
    delisted = delisted_metadata(raw_symbol, symbol, delisted_index)
    result: dict[str, Any] = {
        "idea_id": idea["idea_id"],
        "raw_symbol": raw_symbol,
        "eodhd_symbol": symbol,
        "company_name": idea.get("company_name"),
        "publication_date": pub_date.isoformat(),
        "start_trade_date": start.date.isoformat() if start else None,
        "start_adjusted_close": start.adjusted_close if start else None,
        "price_rows": len(prices),
        "first_price_date": prices[0].date.isoformat() if prices else None,
        "last_price_date": prices[-1].date.isoformat() if prices else None,
        "split_count": len(splits),
        "dividend_count": len(dividends),
        **delisted,
        **fundamentals_row_fields(fundamentals_summary),
        "horizons": {},
        "failure_modes": [],
        "warning_modes": [],
        "validation_status": "unreviewed",
        "label_quality": "unusable",
    }

    if not prices or not start:
        result["failure_modes"].append("missing_start_price")
        result["validation_status"] = "provider_error"
        apply_review_stage(result)
        return result
    if start.adjusted_close <= 0:
        result["failure_modes"].append("invalid_zero_or_negative_start_adjusted_close")
        result["validation_status"] = "provider_error"
        apply_review_stage(result)
        return result
    start_too_late = prices[0].date > pub_date + timedelta(days=10)
    if prices[0].date > pub_date + timedelta(days=10):
        result["failure_modes"].append("first_price_far_after_publication")

    max_mature_horizon = pub_date
    for years in HORIZONS:
        target = add_years(pub_date, years)
        if target <= date.today():
            max_mature_horizon = target
        endpoint = first_on_or_after(prices, target)
        window_start = target - timedelta(days=182)
        window_end = target + timedelta(days=182)
        median_values = adjusted_values_in_window(prices, window_start, window_end)
        horizon: dict[str, Any] = {
            "target_date": target.isoformat(),
            "matured": target <= date.today(),
            "trade_date": endpoint.date.isoformat() if endpoint and not start_too_late else None,
            "adjusted_close": endpoint.adjusted_close if endpoint and not start_too_late else None,
            "multiplier": endpoint.adjusted_close / start.adjusted_close if endpoint and not start_too_late else None,
            "median_52w_multiplier": statistics.median(median_values) / start.adjusted_close
            if median_values and not start_too_late
            else None,
            "median_52w_observations": len(median_values),
        }
        if start_too_late:
            horizon["status"] = "invalid_start_price"
        elif target <= date.today() and endpoint is None:
            horizon["status"] = "missing_endpoint_price"
        elif target > date.today():
            horizon["status"] = "not_mature_yet"
        else:
            horizon["status"] = "calculated"
        result["horizons"][f"{years}y"] = horizon

    if prices[-1].date < max_mature_horizon - timedelta(days=30):
        result["warning_modes"].append("price_history_ends_before_long_horizon")
    if result["is_in_delisted_cache"]:
        result["warning_modes"].append("symbol_in_delisted_cache")
    apply_fundamentals_warnings(result, fundamentals_summary)
    if has_reverse_split(splits):
        result["warning_modes"].append("reverse_split_provider_adjusted")
    if any_extreme(result):
        result["warning_modes"].append("extreme_return_requires_stronger_evidence")

    if not primary_horizons_covered(result):
        result["failure_modes"].append("primary_horizon_missing_price")
    if start_too_late:
        result["failure_modes"].append("invalid_start_price_for_publication_date")
    if has_unreconciled_high_risk_warning(result):
        result["failure_modes"].append("high_risk_warning_needs_review")

    if result["failure_modes"]:
        result["validation_status"] = "needs_manual_review"
        result["label_quality"] = "low"
    else:
        result["validation_status"] = "verified_candidate_provider_adjusted"
        result["label_quality"] = "medium"
    apply_review_stage(result)
    return result


def fundamentals_row_fields(summary: dict[str, Any] | None) -> dict[str, Any]:
    keys = [
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
        "fundamentals_latest_yearly_income_date",
        "fundamentals_latest_quarterly_income_date",
        "fundamentals_yearly_revenue_first",
        "fundamentals_yearly_revenue_last",
        "fundamentals_yearly_net_income_first",
        "fundamentals_yearly_net_income_last",
        "fundamentals_has_financials",
    ]
    if not summary:
        return {"fundamentals_status": "not_fetched"}
    return {key: summary.get(key) for key in keys}


def truthy_provider_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "t", "yes", "y"}


def apply_fundamentals_warnings(result: dict[str, Any], summary: dict[str, Any] | None) -> None:
    if not summary or summary.get("fundamentals_status") == "not_fetched":
        return
    if summary.get("fundamentals_status") != "fetched":
        result["warning_modes"].append("fundamentals_provider_error")
        return
    if truthy_provider_value(summary.get("fundamentals_is_delisted")):
        result["warning_modes"].append("fundamentals_is_delisted")
    instrument_type = str(summary.get("fundamentals_type") or "").strip().lower()
    if instrument_type and instrument_type not in COMMON_STOCK_TYPES:
        result["warning_modes"].append("fundamentals_non_common_instrument")
    if instrument_type in COMMON_STOCK_TYPES and not truthy_provider_value(summary.get("fundamentals_has_financials")):
        result["warning_modes"].append("fundamentals_financials_missing")


def primary_horizons_covered(result: dict[str, Any]) -> bool:
    horizons = result.get("horizons") or {}
    for years in PRIMARY_HORIZONS:
        horizon = horizons.get(f"{years}y") or {}
        if horizon.get("matured") is True and horizon.get("multiplier") is None:
            return False
    return True


def has_unreconciled_high_risk_warning(result: dict[str, Any]) -> bool:
    warnings = set(result.get("warning_modes") or [])
    if "price_history_ends_before_long_horizon" in warnings:
        return True
    if "symbol_in_delisted_cache" in warnings:
        return True
    if "fundamentals_is_delisted" in warnings:
        return True
    if "fundamentals_non_common_instrument" in warnings:
        return True
    if "fundamentals_financials_missing" in warnings:
        return True
    if "reverse_split_provider_adjusted" in warnings:
        return True
    if "extreme_return_requires_stronger_evidence" in warnings:
        # Extreme is not automatically bad, but it is also not training-ready
        # from provider math alone. Agent C must document business reality and
        # corporate-action evidence before promotion.
        return True
    return False


def any_extreme(result: dict[str, Any]) -> bool:
    for horizon in result["horizons"].values():
        multiplier = horizon.get("multiplier")
        if multiplier is not None and (multiplier > 15 or multiplier < 0.05):
            return True
    return False


def horizon_multipliers(result: dict[str, Any]) -> dict[str, float]:
    multipliers: dict[str, float] = {}
    for horizon_name, horizon in (result.get("horizons") or {}).items():
        multiplier = horizon.get("multiplier")
        if isinstance(multiplier, (int, float)):
            multipliers[horizon_name] = float(multiplier)
    return multipliers


def highest_priority_horizon(result: dict[str, Any]) -> tuple[str | None, float | None]:
    multipliers = horizon_multipliers(result)
    if not multipliers:
        return None, None
    extreme_winners = {name: value for name, value in multipliers.items() if value > 15}
    if extreme_winners:
        return max(extreme_winners.items(), key=lambda item: item[1])
    severe_losers = {name: value for name, value in multipliers.items() if value < 0.05}
    if severe_losers:
        return min(severe_losers.items(), key=lambda item: item[1])
    return max(multipliers.items(), key=lambda item: abs(item[1] - 1))


def review_priority(result: dict[str, Any]) -> tuple[int, str, list[str]]:
    warnings = set(result.get("warning_modes") or [])
    failures = set(result.get("failure_modes") or [])
    multipliers = horizon_multipliers(result)
    tags: list[str] = []

    if any(value > 15 for value in multipliers.values()):
        tags.append("extreme_winner")
        return 10, "extreme_winner_gt_15x", tags
    if any(value < 0.05 for value in multipliers.values()):
        tags.append("severe_loser")
        return 20, "severe_loser_lt_0_05x", tags
    if (
        "fundamentals_is_delisted" in warnings
        or "symbol_in_delisted_cache" in warnings
        or "price_history_ends_before_long_horizon" in warnings
        or "primary_horizon_missing_price" in failures
    ):
        tags.append("delisted_or_early_ended_history")
        return 30, "delisted_or_early_ended_history", tags
    if "reverse_split_provider_adjusted" in warnings:
        tags.append("reverse_split")
        return 40, "reverse_split_provider_adjusted", tags
    if "lineage_override_requires_agent_review" in failures:
        tags.append("ticker_lineage")
        return 50, "ticker_lineage_change_or_override", tags
    if "fundamentals_non_common_instrument" in warnings:
        tags.append("non_common_instrument")
        return 60, "non_common_instrument", tags
    if "fundamentals_financials_missing" in warnings:
        tags.append("fundamentals_gap")
        return 70, "common_stock_financials_missing", tags
    if result.get("validation_status") == "provider_error" or "provider_fetch_failed" in failures:
        tags.append("provider_error")
        return 80, "provider_error_retry_or_resolve", tags
    if failures:
        tags.append("math_incomplete")
        return 90, "math_incomplete_or_invalid_start", tags
    return 999, "low_risk_math_reproduced", tags


def apply_review_stage(result: dict[str, Any]) -> None:
    failures = set(result.get("failure_modes") or [])
    warnings = set(result.get("warning_modes") or [])
    priority, reason, tags = review_priority(result)
    target_horizon, target_multiplier = highest_priority_horizon(result)

    if result.get("validation_status") == "provider_error" or "provider_fetch_failed" in failures:
        math_status = "provider_error"
        stage = "provider_error"
        readiness = "not_training_ready"
    elif not primary_horizons_covered(result) or "invalid_start_price_for_publication_date" in failures:
        math_status = "math_incomplete"
        stage = "math_incomplete"
        readiness = "not_training_ready"
    else:
        math_status = "math_reproduced"
        if "high_risk_warning_needs_review" in failures or any(
            warning in warnings
            for warning in {
                "extreme_return_requires_stronger_evidence",
                "fundamentals_is_delisted",
                "symbol_in_delisted_cache",
                "fundamentals_non_common_instrument",
                "fundamentals_financials_missing",
                "reverse_split_provider_adjusted",
                "price_history_ends_before_long_horizon",
            }
        ):
            stage = "provider_warning"
            readiness = "manual_review_required"
        else:
            stage = "math_reproduced_low_risk"
            readiness = "candidate_low_risk"

    result["math_validation_status"] = math_status
    result["review_stage"] = stage
    result["training_readiness"] = readiness
    result["manual_review_priority"] = priority
    result["manual_review_reason"] = reason
    result["manual_review_tags"] = tags
    result["review_target_horizon"] = target_horizon
    result["review_target_multiplier"] = target_multiplier


def has_reverse_split(splits: list[dict[str, Any]]) -> bool:
    for split in splits:
        ratio = parse_split_ratio(str(split.get("split") or ""))
        if ratio is not None and ratio < 0.67:
            return True
    return False


def parse_split_ratio(value: str) -> float | None:
    if "/" not in value:
        return None
    left, right = value.split("/", 1)
    try:
        denominator = float(right)
        if denominator == 0:
            return None
        return float(left) / denominator
    except ValueError:
        return None


def output_csv(results: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "idea_id",
        "raw_symbol",
        "eodhd_symbol",
        "publication_date",
        "validation_status",
        "math_validation_status",
        "review_stage",
        "training_readiness",
        "manual_review_priority",
        "manual_review_reason",
        "manual_review_tags",
        "review_target_horizon",
        "review_target_multiplier",
        "label_quality",
        "start_trade_date",
        "start_adjusted_close",
        "first_price_date",
        "last_price_date",
        "price_rows",
        "split_count",
        "dividend_count",
        "is_in_delisted_cache",
        "delisted_provider_code",
        "delisted_provider_exchange",
        "delisted_provider_name",
        "delisted_provider_type",
        "delisted_provider_isin",
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
        "fundamentals_latest_yearly_income_date",
        "fundamentals_latest_quarterly_income_date",
        "fundamentals_yearly_revenue_first",
        "fundamentals_yearly_revenue_last",
        "fundamentals_yearly_net_income_first",
        "fundamentals_yearly_net_income_last",
        "fundamentals_has_financials",
        "perf_1y",
        "perf_3y",
        "perf_5y",
        "perf_10y",
        "perf_20y",
        "failure_modes",
        "warning_modes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            horizons = row.get("horizons") or {}
            writer.writerow(
                {
                    **{key: row.get(key) for key in fieldnames},
                    "perf_1y": (horizons.get("1y") or {}).get("multiplier"),
                    "perf_3y": (horizons.get("3y") or {}).get("multiplier"),
                    "perf_5y": (horizons.get("5y") or {}).get("multiplier"),
                    "perf_10y": (horizons.get("10y") or {}).get("multiplier"),
                    "perf_20y": (horizons.get("20y") or {}).get("multiplier"),
                    "failure_modes": ";".join(row.get("failure_modes", [])),
                    "warning_modes": ";".join(row.get("warning_modes", [])),
                    "manual_review_tags": ";".join(row.get("manual_review_tags", [])),
                }
            )


def should_queue_for_manual_review(row: dict[str, Any]) -> bool:
    return row.get("training_readiness") in {"manual_review_required", "not_training_ready"} and row.get(
        "review_stage"
    ) != "math_reproduced_low_risk"


def output_manual_review_queue(results: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "manual_review_priority",
        "manual_review_reason",
        "manual_review_tags",
        "review_stage",
        "math_validation_status",
        "training_readiness",
        "idea_id",
        "raw_symbol",
        "eodhd_symbol",
        "publication_date",
        "review_target_horizon",
        "review_target_multiplier",
        "perf_1y",
        "perf_3y",
        "perf_5y",
        "perf_10y",
        "perf_20y",
        "validation_status",
        "failure_modes",
        "warning_modes",
        "start_trade_date",
        "start_adjusted_close",
        "first_price_date",
        "last_price_date",
        "price_rows",
        "split_count",
        "dividend_count",
        "is_in_delisted_cache",
        "delisted_provider_code",
        "delisted_provider_exchange",
        "delisted_provider_name",
        "delisted_provider_type",
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
        "fundamentals_latest_yearly_income_date",
        "fundamentals_latest_quarterly_income_date",
        "fundamentals_yearly_revenue_first",
        "fundamentals_yearly_revenue_last",
        "fundamentals_yearly_net_income_first",
        "fundamentals_yearly_net_income_last",
        "fundamentals_has_financials",
    ]
    queue = sorted(
        [row for row in results if should_queue_for_manual_review(row)],
        key=lambda row: (
            row.get("manual_review_priority") or 999,
            row.get("publication_date") or "",
            row.get("eodhd_symbol") or "",
            row.get("idea_id") or "",
        ),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in queue:
            horizons = row.get("horizons") or {}
            writer.writerow(
                {
                    **{key: row.get(key) for key in fieldnames},
                    "perf_1y": (horizons.get("1y") or {}).get("multiplier"),
                    "perf_3y": (horizons.get("3y") or {}).get("multiplier"),
                    "perf_5y": (horizons.get("5y") or {}).get("multiplier"),
                    "perf_10y": (horizons.get("10y") or {}).get("multiplier"),
                    "perf_20y": (horizons.get("20y") or {}).get("multiplier"),
                    "failure_modes": ";".join(row.get("failure_modes", [])),
                    "warning_modes": ";".join(row.get("warning_modes", [])),
                    "manual_review_tags": ";".join(row.get("manual_review_tags", [])),
                }
            )


def load_ideas(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.sample_10:
        return SAMPLE_10
    if not args.ideas_file:
        raise SystemExit("Pass --sample-10 or --ideas-file.")
    with Path(args.ideas_file).open("r", encoding="utf-8") as handle:
        ideas = json.load(handle)
    if not isinstance(ideas, list):
        raise SystemExit("--ideas-file must contain a JSON array.")
    return ideas


def load_lineage(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("--lineage-file must contain a JSON object keyed by raw ticker.")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-10", action="store_true")
    parser.add_argument("--ideas-file")
    parser.add_argument("--lineage-file")
    parser.add_argument("--output-dir", default="eodhd_output")
    parser.add_argument("--include-master-delisted", action="store_true")
    parser.add_argument(
        "--delisted-exchanges",
        nargs="*",
        help="Exchange codes for EODHD delisted archives. Defaults to all exchanges present in --ideas-file.",
    )
    parser.add_argument("--include-fundamentals", action="store_true")
    parser.add_argument(
        "--fundamentals-mode",
        choices=("bulk", "single"),
        default="bulk",
        help="Use EODHD bulk-fundamentals by exchange, or single-symbol v1.1 fundamentals.",
    )
    parser.add_argument(
        "--fundamentals-cache-dir",
        help="Directory for full EODHD Fundamentals v1.1 JSON cache. Defaults to OUTPUT_DIR/fundamentals_cache.",
    )
    parser.add_argument(
        "--bulk-fundamentals-cache-dir",
        help="Directory for raw EODHD bulk-fundamentals JSON chunks. Defaults to OUTPUT_DIR/fundamentals_bulk_cache.",
    )
    parser.add_argument(
        "--refresh-fundamentals-cache",
        action="store_true",
        help="Refresh fundamentals only. Does not refresh EOD/splits/dividends cache.",
    )
    parser.add_argument(
        "--retry-fundamentals-errors",
        action="store_true",
        help="Retry cached fundamentals .error.json files instead of treating them as final.",
    )
    parser.add_argument(
        "--fundamentals-only",
        action="store_true",
        help="Only fetch/write fundamentals caches and summaries; skip EOD price validation output.",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.fundamentals_only and not args.include_fundamentals:
        raise SystemExit("--fundamentals-only requires --include-fundamentals.")

    ideas = load_ideas(args)
    lineage = load_lineage(args.lineage_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "symbol_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fundamentals_cache_dir = Path(args.fundamentals_cache_dir) if args.fundamentals_cache_dir else out_dir / "fundamentals_cache"
    bulk_fundamentals_cache_dir = (
        Path(args.bulk_fundamentals_cache_dir) if args.bulk_fundamentals_cache_dir else out_dir / "fundamentals_bulk_cache"
    )
    if args.include_fundamentals:
        fundamentals_cache_dir.mkdir(parents=True, exist_ok=True)
        if args.fundamentals_mode == "bulk":
            bulk_fundamentals_cache_dir.mkdir(parents=True, exist_ok=True)

    delisted_index: dict[str, dict[str, Any]] = {}
    if args.include_master_delisted:
        exchanges = sorted({exchange.upper() for exchange in args.delisted_exchanges}) if args.delisted_exchanges else delisted_exchanges_for_ideas(ideas)
        rows_by_exchange = load_delisted_symbol_lists(out_dir, exchanges, args.refresh_cache)
        delisted_index = build_delisted_index(rows_by_exchange)
        print(
            "delisted_archives="
            + ",".join(f"{exchange}:{len(rows)}" for exchange, rows in sorted(rows_by_exchange.items()))
            + f" indexed_keys={len(delisted_index)}",
            file=sys.stderr,
        )

    results: list[dict[str, Any]] = []
    raw_cache: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, str]]] = {}
    for idea in ideas:
        grouped.setdefault(idea["eodhd_symbol"], []).append(idea)

    fundamentals_bundles: dict[str, dict[str, Any]] = {}
    fundamentals_summaries: dict[str, dict[str, Any]] = {}
    if args.include_fundamentals:
        fundamentals_refresh = args.refresh_cache or args.refresh_fundamentals_cache
        if args.fundamentals_mode == "bulk":
            fundamentals_bundles = load_fundamentals_bundles_bulk(
                list(grouped),
                fundamentals_cache_dir,
                bulk_fundamentals_cache_dir,
                fundamentals_refresh,
                args.concurrency,
            )
        else:
            fundamentals_bundles = load_fundamentals_bundles_single(
                list(grouped),
                fundamentals_cache_dir,
                fundamentals_refresh,
                args.retry_fundamentals_errors,
                args.concurrency,
            )
        fundamentals_summaries = {
            symbol: extract_fundamentals_summary(symbol, fundamentals_bundles.get(symbol))
            for symbol in grouped
        }
        output_fundamentals_summary_csv(fundamentals_summaries, out_dir / "fundamentals_summary.csv")
        (out_dir / "fundamentals_summary.json").write_text(
            json.dumps(fundamentals_summaries, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if args.fundamentals_only:
            print(f"fundamentals_symbols={len(fundamentals_summaries)}")
            print(f"fundamentals_output={out_dir.resolve()}")
            return 0

    bundles: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {
            executor.submit(load_symbol_bundle, symbol, symbol_ideas, cache_dir, args.refresh_cache): symbol
            for symbol, symbol_ideas in grouped.items()
        }
        completed = 0
        for future in as_completed(futures):
            symbol = futures[future]
            completed += 1
            try:
                _, bundle = future.result()
                bundles[symbol] = bundle
            except Exception as exc:  # noqa: BLE001 - preserve provider/network failures per symbol.
                error_message = redact_api_token(str(exc))
                failures[symbol] = error_message
                error_cache_path(cache_dir, symbol).write_text(
                    json.dumps(
                        {"symbol": symbol, "error": error_message, "failed_at": datetime.utcnow().isoformat() + "Z"},
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if completed % 100 == 0 or completed == len(futures):
                print(f"fetched_symbols={completed}/{len(futures)} failures={len(failures)}", file=sys.stderr)

    for symbol, symbol_ideas in grouped.items():
        bundle = bundles.get(symbol)
        if not bundle:
            provider_warning = failures.get(symbol, "unknown_provider_error")
        elif bundle.get("cached_error"):
            provider_warning = str(bundle["cached_error"])
        else:
            provider_warning = ""
        if not bundle or bundle.get("cached_error"):
            for idea in symbol_ideas:
                delisted = delisted_metadata(idea.get("raw_symbol"), symbol, delisted_index)
                fundamentals_summary = fundamentals_summaries.get(symbol)
                row = {
                    "idea_id": idea["idea_id"],
                    "raw_symbol": idea["raw_symbol"],
                    "eodhd_symbol": symbol,
                    "company_name": idea.get("company_name"),
                    "publication_date": idea["publication_date"],
                    "start_trade_date": None,
                    "start_adjusted_close": None,
                    "price_rows": 0,
                    "first_price_date": None,
                    "last_price_date": None,
                    "split_count": 0,
                    "dividend_count": 0,
                    **delisted,
                    **fundamentals_row_fields(fundamentals_summary),
                    "horizons": {f"{years}y": {"multiplier": None} for years in HORIZONS},
                    "failure_modes": ["provider_fetch_failed"],
                    "warning_modes": [],
                    "validation_status": "provider_error",
                    "label_quality": "unusable",
                    "provider_warnings": [provider_warning],
                }
                apply_review_stage(row)
                results.append(row)
            continue
        prices, price_warnings = bundle_to_prices(bundle)
        splits = [row for row in bundle.get("splits") or [] if isinstance(row, dict)]
        dividends = [row for row in bundle.get("dividends") or [] if isinstance(row, dict)]
        split_warnings = list(bundle.get("split_warnings") or [])
        dividend_warnings = list(bundle.get("dividend_warnings") or [])
        raw_cache[symbol] = {
            "delisted_provider_record": find_delisted_record(None, symbol, delisted_index),
            "fundamentals_summary": fundamentals_summaries.get(symbol),
            "price_warnings": price_warnings,
            "split_warnings": split_warnings,
            "dividend_warnings": dividend_warnings,
            "price_row_count": len(prices),
            "first_price_date": prices[0].date.isoformat() if prices else None,
            "last_price_date": prices[-1].date.isoformat() if prices else None,
            "splits": splits,
            "dividends": dividends,
        }
        for idea in symbol_ideas:
            row = calculate_row(idea, prices, splits, dividends, delisted_index, fundamentals_summaries.get(symbol))
            lineage_note = lineage.get(idea["raw_symbol"].upper())
            if lineage_note:
                row["lineage_override"] = lineage_note
                row["failure_modes"].append("lineage_override_requires_agent_review")
                row["validation_status"] = "needs_manual_review"
                row["label_quality"] = "low"
                apply_review_stage(row)
            row["provider_warnings"] = price_warnings + split_warnings + dividend_warnings
            results.append(row)

    (out_dir / "raw_cache_sample.json").write_text(json.dumps(raw_cache, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "validation_results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    output_csv(results, out_dir / "validation_results.csv")
    output_manual_review_queue(results, out_dir / "manual_review_queue.csv")

    print(f"ideas={len(results)}")
    print(f"symbols={len(grouped)}")
    print(f"output_dir={out_dir.resolve()}")
    if not args.quiet:
        for row in results:
            horizons = row["horizons"]
            print(
                f"{row['raw_symbol']:>6} {row['publication_date']} "
                f"1y={fmt((horizons.get('1y') or {}).get('multiplier'))} "
                f"3y={fmt((horizons.get('3y') or {}).get('multiplier'))} "
                f"5y={fmt((horizons.get('5y') or {}).get('multiplier'))} "
                f"10y={fmt((horizons.get('10y') or {}).get('multiplier'))} "
                f"20y={fmt((horizons.get('20y') or {}).get('multiplier'))} "
                f"status={row['validation_status']} "
                f"flags={','.join(row.get('failure_modes') or []) or 'none'} "
                f"warnings={','.join(row.get('warning_modes') or []) or 'none'}"
            )
    return 0


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    return f"{float(value):.4f}x"


if __name__ == "__main__":
    sys.exit(main())
