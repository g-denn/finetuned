#!/usr/bin/env python3
"""Collect EODHD fundamentals for all common stocks in Japan and Korea.

The collector discovers exchanges from EODHD, fetches common-stock symbols for
Japan and South Korea, pulls Fundamentals API v1.1 payloads, and writes both raw
JSON and normalized CSV/JSONL files for later screening.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get("EODHD_JK_OUT_DIR", str(ROOT / "eodhd_output" / "japan_korea_fundamentals")))
RAW_DIR = OUT_DIR / "raw"
NORMALIZED_DIR = OUT_DIR / "normalized"
BASE_URL = "https://eodhd.com/api"
FUNDAMENTALS_BASE_URL = "https://eodhd.com/api/v1.1"

TARGET_COUNTRIES = {
    "JP": {"iso2": "JP", "iso3": "JPN", "names": {"japan", "jpn"}},
    "KR": {"iso2": "KR", "iso3": "KOR", "names": {"south korea", "korea", "republic of korea", "kor"}},
}

TRANSCRIPT_DOC_STATUS = {
    "documented_in_eodhd_fundamentals_or_calendar_docs": False,
    "checked_at_utc": None,
    "note": (
        "EODHD documents earnings history, earnings trends, annual/quarterly earnings, "
        "income statement, balance sheet, and cash flow statement fields. It does not "
        "document earnings transcript text in the Fundamentals or Calendar API docs."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on symbols fetched this run.")
    parser.add_argument("--offset", type=int, default=0, help="Start offset into the manifest.")
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay between API calls.")
    parser.add_argument(
        "--max-calls-per-minute",
        type=float,
        default=0.0,
        help="Optional global API throttle. 0 disables the calls-per-minute limiter.",
    )
    parser.add_argument("--retries", type=int, default=2, help="Retries per API request.")
    parser.add_argument("--progress-every", type=int, default=50, help="Refresh progress files after this many symbols.")
    parser.add_argument("--quiet", action="store_true", help="Do not print per-symbol fetch results.")
    parser.add_argument("--force", action="store_true", help="Refetch fundamentals even when raw JSON exists.")
    parser.add_argument("--pending-only", action="store_true", help="Fetch only symbols missing raw fundamentals JSON.")
    parser.add_argument("--status-only", action="store_true", help="Write manifest/progress only; do not fetch fundamentals.")
    parser.add_argument(
        "--countries",
        default="JP,KR",
        help="Comma-separated country keys to include. Defaults to JP,KR.",
    )
    parser.add_argument(
        "--include-delisted",
        action="store_true",
        help="Also fetch currently delisted common stocks into the manifest.",
    )
    parser.add_argument(
        "--extra-exchange",
        action="append",
        default=[],
        metavar="COUNTRY:CODE[:NAME]",
        help=(
            "Manually add an exchange when EODHD exchanges-list omits it. "
            "Example: JP:TSE:Tokyo Stock Exchange. Can be repeated."
        ),
    )
    parser.add_argument(
        "--normalize-only",
        action="store_true",
        help="Rebuild normalized outputs from existing raw files without API calls.",
    )
    parser.add_argument("--skip-normalize", action="store_true", help="Fetch data and progress only; skip normalization.")
    return parser.parse_args()


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    for attempt in range(10):
        try:
            temp_path.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def atomic_write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(10):
        try:
            temp_path.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))


def clean_column(value: str) -> str:
    out: list[str] = []
    prev_underscore = False
    for char in value:
        if char.isalnum():
            out.append(char.lower())
            prev_underscore = False
        elif not prev_underscore:
            out.append("_")
            prev_underscore = True
    return "".join(out).strip("_")


def safe_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace("\\", "_").replace(":", "_")


class RateLimiter:
    def __init__(self, max_calls_per_minute: float) -> None:
        self.min_interval = 60.0 / max_calls_per_minute if max_calls_per_minute > 0 else 0.0
        self.last_call_at = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        wait_seconds = self.min_interval - (now - self.last_call_at)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        self.last_call_at = time.monotonic()


def request_json(url: str) -> tuple[int | None, Any, str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": "SignalValueInvestorWorkbench/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw), None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:1500]
        return exc.code, None, body
    except Exception as exc:  # noqa: BLE001 - preserve operational details for long pulls.
        return None, None, f"{type(exc).__name__}: {exc}"


def endpoint_url(base: str, path: str, token: str, params: dict[str, str] | None = None) -> str:
    query = {"api_token": token, "fmt": "json"}
    if params:
        query.update(params)
    return f"{base}/{path}?{urllib.parse.urlencode(query)}"


def fetch_with_cache(
    url: str,
    output_path: Path,
    force: bool,
    retries: int,
    sleep_seconds: float,
    rate_limiter: RateLimiter | None = None,
) -> dict[str, Any]:
    error_path = output_path.with_suffix(".error.json")
    if output_path.exists() and not force:
        try:
            cached = json.loads(output_path.read_text(encoding="utf-8"))
            return {
                "status": "cached",
                "items": len(cached) if isinstance(cached, (dict, list)) else None,
                "json_type": type(cached).__name__,
            }
        except json.JSONDecodeError:
            pass

    result: dict[str, Any] = {"status": "pending"}
    for attempt in range(retries + 1):
        if rate_limiter:
            rate_limiter.wait()
        http_status, data, error = request_json(url)
        result = {"http_status": http_status, "attempt": attempt + 1}
        if error is None:
            atomic_write_json(output_path, data)
            if error_path.exists():
                error_path.unlink()
            result.update(
                {
                    "status": "fetched",
                    "items": len(data) if isinstance(data, (dict, list)) else None,
                    "json_type": type(data).__name__,
                }
            )
            return result

        result["status"] = "error"
        result["error"] = error
        atomic_write_json(error_path, result)
        if attempt < retries:
            retry_floor = 30.0 if http_status == 429 else 0.5
            time.sleep(max(sleep_seconds, retry_floor) * (attempt + 1))
    return result


def country_matches(exchange: dict[str, Any], country_keys: set[str]) -> bool:
    for key in country_keys:
        target = TARGET_COUNTRIES[key]
        country = str(exchange.get("Country") or "").strip().lower()
        iso2 = str(exchange.get("CountryISO2") or "").strip().upper()
        iso3 = str(exchange.get("CountryISO3") or "").strip().upper()
        if iso2 == target["iso2"] or iso3 == target["iso3"] or country in target["names"]:
            return True
    return False


def row_country_key(row: dict[str, Any], exchanges_by_code: dict[str, dict[str, Any]]) -> str:
    exchange = exchanges_by_code.get(str(row.get("Exchange") or ""))
    country_text = " ".join(
        str(value or "")
        for value in [
            row.get("Country"),
            exchange.get("Country") if exchange else "",
            exchange.get("CountryISO2") if exchange else "",
            exchange.get("CountryISO3") if exchange else "",
        ]
    ).lower()
    if "japan" in country_text or " jp" in f" {country_text} " or "jpn" in country_text:
        return "JP"
    if "korea" in country_text or " kr" in f" {country_text} " or "kor" in country_text:
        return "KR"
    return ""


def parse_extra_exchanges(values: list[str], country_keys: set[str]) -> list[dict[str, Any]]:
    exchanges: list[dict[str, Any]] = []
    for value in values:
        parts = [part.strip() for part in value.split(":", 2)]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise SystemExit(f"Invalid --extra-exchange value: {value!r}. Expected COUNTRY:CODE[:NAME].")
        country_key = parts[0].upper()
        if country_key not in TARGET_COUNTRIES:
            raise SystemExit(f"Invalid --extra-exchange country {country_key!r}. Supported: {sorted(TARGET_COUNTRIES)}")
        if country_key not in country_keys:
            continue
        target = TARGET_COUNTRIES[country_key]
        code = parts[1].upper()
        name = parts[2] if len(parts) == 3 and parts[2] else code
        exchanges.append(
            {
                "Code": code,
                "Name": name,
                "Country": next(iter(target["names"])).title(),
                "CountryISO2": target["iso2"],
                "CountryISO3": target["iso3"],
                "Currency": "JPY" if country_key == "JP" else "KRW",
                "OperatingMIC": "",
                "_manual_extra_exchange": True,
            }
        )
    return exchanges


def discover_manifest(
    token: str,
    country_keys: set[str],
    include_delisted: bool,
    sleep_seconds: float,
    extra_exchanges: list[str] | None = None,
    rate_limiter: RateLimiter | None = None,
) -> list[dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exchanges_url = endpoint_url(BASE_URL, "exchanges-list/", token)
    if rate_limiter:
        rate_limiter.wait()
    exchanges_status, exchanges, exchange_error = request_json(exchanges_url)
    if exchange_error is not None:
        raise SystemExit(f"Could not fetch EODHD exchanges list: status={exchanges_status} error={exchange_error}")
    if not isinstance(exchanges, list):
        raise SystemExit("EODHD exchanges-list response was not a JSON array.")

    atomic_write_json(OUT_DIR / "exchanges_all.json", exchanges)
    manual_exchanges = parse_extra_exchanges(extra_exchanges or [], country_keys)
    selected_exchanges = [item for item in exchanges if isinstance(item, dict) and country_matches(item, country_keys)]
    selected_exchanges.extend(manual_exchanges)
    exchanges_by_code = {str(item.get("Code")): item for item in selected_exchanges if item.get("Code")}
    atomic_write_json(OUT_DIR / "exchanges_selected.json", selected_exchanges)
    availability = {}
    for key in sorted(country_keys):
        matching = [item for item in exchanges if isinstance(item, dict) and country_matches(item, {key})]
        manual_matching = [item for item in manual_exchanges if country_matches(item, {key})]
        availability[key] = {
            "requested": True,
            "exchange_count": len(matching) + len(manual_matching),
            "exchange_codes": [item.get("Code") for item in matching + manual_matching],
            "available_in_exchanges_list": bool(matching),
            "manual_exchange_codes": [item.get("Code") for item in manual_matching],
        }
    atomic_write_json(OUT_DIR / "country_availability.json", availability)

    all_symbols: list[dict[str, Any]] = []
    seen: set[str] = set()
    symbol_list_status: list[dict[str, Any]] = []
    for exchange in selected_exchanges:
        code = str(exchange.get("Code") or "").strip()
        if not code:
            continue
        for delisted_flag in ([False, True] if include_delisted else [False]):
            params = {"type": "common_stock"}
            if delisted_flag:
                params["delisted"] = "1"
            url = endpoint_url(BASE_URL, f"exchange-symbol-list/{urllib.parse.quote(code)}", token, params)
            if rate_limiter:
                rate_limiter.wait()
            status, symbols, error = request_json(url)
            status_row = {
                "exchange": code,
                "delisted": delisted_flag,
                "http_status": status,
                "error": error,
                "symbols": len(symbols) if isinstance(symbols, list) else None,
            }
            symbol_list_status.append(status_row)
            if error is not None or not isinstance(symbols, list):
                atomic_write_json(OUT_DIR / "symbol_list_status.json", symbol_list_status)
                time.sleep(sleep_seconds)
                continue
            atomic_write_json(OUT_DIR / "symbol_lists" / f"{safe_name(code)}{'_delisted' if delisted_flag else ''}.json", symbols)
            for item in symbols:
                if not isinstance(item, dict):
                    continue
                ticker_code = str(item.get("Code") or "").strip()
                exchange_code = str(item.get("Exchange") or code).strip()
                if not ticker_code or not exchange_code:
                    continue
                type_value = str(item.get("Type") or "").lower()
                if "common" not in type_value and type_value not in {"stock", "common stock"}:
                    continue
                symbol = f"{ticker_code}.{exchange_code}"
                if symbol in seen:
                    continue
                seen.add(symbol)
                country_key = row_country_key(item, exchanges_by_code)
                all_symbols.append(
                    {
                        "symbol": symbol,
                        "code": ticker_code,
                        "exchange": exchange_code,
                        "name": item.get("Name"),
                        "country": item.get("Country"),
                        "country_key": country_key,
                        "currency": item.get("Currency"),
                        "type": item.get("Type"),
                        "isin": item.get("Isin") or item.get("ISIN"),
                        "is_delisted_symbol_list": delisted_flag,
                        "exchange_name": exchange.get("Name"),
                        "exchange_country": exchange.get("Country"),
                        "exchange_country_iso2": exchange.get("CountryISO2"),
                        "exchange_country_iso3": exchange.get("CountryISO3"),
                    }
                )
            atomic_write_json(OUT_DIR / "symbol_list_status.json", symbol_list_status)
            time.sleep(sleep_seconds)

    filtered = [row for row in all_symbols if row["country_key"] in country_keys]
    filtered.sort(key=lambda row: (row["country_key"], row["exchange"], row["code"]))
    write_manifest(filtered)
    return filtered


def write_manifest(manifest: list[dict[str, Any]]) -> None:
    atomic_write_json(OUT_DIR / "stock_pull_manifest.json", manifest)
    if manifest:
        atomic_write_rows(OUT_DIR / "stock_pull_manifest.csv", manifest)


def read_manifest() -> list[dict[str, Any]]:
    path = OUT_DIR / "stock_pull_manifest.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def progress_for_symbol(item: dict[str, Any]) -> dict[str, Any]:
    symbol = item["symbol"]
    symbol_dir = RAW_DIR / safe_name(symbol)
    fundamentals_path = symbol_dir / "fundamentals.json"
    error_files = sorted(path.name for path in symbol_dir.glob("*.error.json")) if symbol_dir.exists() else []
    transcript_found = False
    transcript_keys = []
    if fundamentals_path.exists():
        try:
            payload = json.loads(fundamentals_path.read_text(encoding="utf-8"))
            transcript_keys = find_transcript_keys(payload)
            transcript_found = bool(transcript_keys)
        except json.JSONDecodeError:
            error_files.append("fundamentals.invalid_json")
    return {
        **item,
        "fundamentals_saved": fundamentals_path.exists(),
        "error_count": len(error_files),
        "error_files": ";".join(error_files),
        "transcript_like_keys_found": ";".join(transcript_keys),
        "transcript_available_in_payload": transcript_found,
    }


def write_progress(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [progress_for_symbol(item) for item in manifest]
    country_counts = Counter(row.get("country_key") for row in rows)
    complete_by_country = Counter(row.get("country_key") for row in rows if row["fundamentals_saved"])
    summary = {
        "updated_at_utc": now_iso(),
        "total_symbols": len(rows),
        "fundamentals_saved": sum(1 for row in rows if row["fundamentals_saved"]),
        "pending_fundamentals": sum(1 for row in rows if not row["fundamentals_saved"]),
        "symbols_with_errors": sum(1 for row in rows if row["error_count"]),
        "symbols_with_transcript_like_keys": sum(1 for row in rows if row["transcript_available_in_payload"]),
        "country_counts": dict(country_counts),
        "fundamentals_saved_by_country": dict(complete_by_country),
    }
    atomic_write_json(OUT_DIR / "progress_summary.json", summary)
    atomic_write_json(OUT_DIR / "progress_checklist.json", rows)
    if rows:
        atomic_write_rows(OUT_DIR / "progress_checklist.csv", rows)
    return summary


def find_transcript_keys(payload: Any, prefix: str = "", limit: int = 50) -> list[str]:
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_path = f"{prefix}.{key}" if prefix else str(key)
            if "transcript" in str(key).lower():
                found.append(key_path)
                if len(found) >= limit:
                    return found
            found.extend(find_transcript_keys(value, key_path, limit - len(found)))
            if len(found) >= limit:
                return found
    elif isinstance(payload, list):
        for index, item in enumerate(payload[:10]):
            found.extend(find_transcript_keys(item, f"{prefix}[{index}]", limit - len(found)))
            if len(found) >= limit:
                return found
    return found


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def flatten_scalar_section(payload: Any, prefix: str) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if not isinstance(payload, dict):
        return row
    for key, value in payload.items():
        if is_scalar(value):
            row[f"{prefix}_{clean_column(str(key))}"] = value
    return row


def iter_statement_rows(symbol: str, fundamentals: dict[str, Any], statement_key: str, statement_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    section = fundamentals.get("Financials", {}).get(statement_key, {})
    if not isinstance(section, dict):
        return rows
    for period in ["yearly", "quarterly"]:
        records = section.get(period, {})
        if not isinstance(records, dict):
            continue
        for date_key, record in sorted(records.items()):
            if not isinstance(record, dict):
                continue
            row = {"symbol": symbol, "statement": statement_name, "period": period, "date": date_key}
            for key, value in record.items():
                if is_scalar(value):
                    row[clean_column(str(key))] = value
            rows.append(row)
    return rows


def latest_five_yearly_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("period") != "yearly":
            continue
        symbol = str(row.get("symbol") or "")
        if not symbol:
            continue
        by_symbol.setdefault(symbol, []).append(row)

    latest_rows: list[dict[str, Any]] = []
    for symbol_rows in by_symbol.values():
        latest_rows.extend(sorted(symbol_rows, key=lambda row: str(row.get("date") or ""), reverse=True)[:5])
    latest_rows.sort(key=lambda row: (str(row.get("symbol") or ""), str(row.get("date") or "")), reverse=True)
    return latest_rows


def iter_earnings_rows(symbol: str, fundamentals: dict[str, Any]) -> list[dict[str, Any]]:
    earnings = fundamentals.get("Earnings", {})
    if not isinstance(earnings, dict):
        return []
    rows: list[dict[str, Any]] = []
    for section_name, records in earnings.items():
        if not isinstance(records, dict):
            continue
        for date_key, record in sorted(records.items()):
            if not isinstance(record, dict):
                continue
            row = {"symbol": symbol, "section": section_name, "date": date_key}
            for key, value in record.items():
                if is_scalar(value):
                    row[clean_column(str(key))] = value
            rows.append(row)
    return rows


def normalize(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    company_rows: list[dict[str, Any]] = []
    income_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    cash_flow_rows: list[dict[str, Any]] = []
    earnings_rows: list[dict[str, Any]] = []
    raw_jsonl_rows = 0
    transcript_key_hits: list[dict[str, Any]] = []

    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = NORMALIZED_DIR / "fundamentals_raw_payloads.jsonl"
    temp_jsonl = jsonl_path.with_name(f"{jsonl_path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temp_jsonl.open("w", encoding="utf-8") as jsonl:
        for item in manifest:
            symbol = item["symbol"]
            fundamentals_path = RAW_DIR / safe_name(symbol) / "fundamentals.json"
            if not fundamentals_path.exists():
                continue
            try:
                fundamentals = json.loads(fundamentals_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(fundamentals, dict):
                continue

            jsonl.write(json.dumps({"symbol": symbol, "fundamentals": fundamentals}, sort_keys=True) + "\n")
            raw_jsonl_rows += 1

            row = {f"manifest_{clean_column(key)}": value for key, value in item.items()}
            row["symbol"] = symbol
            row["raw_fundamentals_path"] = f"raw/{safe_name(symbol)}/fundamentals.json"
            for section in ["General", "Highlights", "Valuation", "SharesStats", "Technicals", "SplitsDividends"]:
                row.update(flatten_scalar_section(fundamentals.get(section), clean_column(section)))
            company_rows.append(row)

            income_rows.extend(iter_statement_rows(symbol, fundamentals, "Income_Statement", "income_statement"))
            balance_rows.extend(iter_statement_rows(symbol, fundamentals, "Balance_Sheet", "balance_sheet"))
            cash_flow_rows.extend(iter_statement_rows(symbol, fundamentals, "Cash_Flow", "cash_flow"))
            earnings_rows.extend(iter_earnings_rows(symbol, fundamentals))
            transcript_keys = find_transcript_keys(fundamentals)
            if transcript_keys:
                transcript_key_hits.append({"symbol": symbol, "transcript_like_keys": transcript_keys})
    temp_jsonl.replace(jsonl_path)

    income_latest_5y_rows = latest_five_yearly_rows(income_rows)
    balance_latest_5y_rows = latest_five_yearly_rows(balance_rows)
    cash_flow_latest_5y_rows = latest_five_yearly_rows(cash_flow_rows)

    outputs = {
        "companies.csv": company_rows,
        "income_statement.csv": income_rows,
        "balance_sheet.csv": balance_rows,
        "cash_flow.csv": cash_flow_rows,
        "income_statement_latest_5y.csv": income_latest_5y_rows,
        "balance_sheet_latest_5y.csv": balance_latest_5y_rows,
        "cash_flow_latest_5y.csv": cash_flow_latest_5y_rows,
        "earnings.csv": earnings_rows,
    }
    for filename, rows in outputs.items():
        if rows:
            atomic_write_rows(NORMALIZED_DIR / filename, rows)
        else:
            atomic_write_rows(NORMALIZED_DIR / filename, [])

    transcript_status = dict(TRANSCRIPT_DOC_STATUS)
    transcript_status["checked_at_utc"] = now_iso()
    transcript_status["transcript_like_keys_found_in_payloads"] = transcript_key_hits
    transcript_status["payloads_with_transcript_like_keys"] = len(transcript_key_hits)
    atomic_write_json(OUT_DIR / "earnings_transcript_availability.json", transcript_status)

    summary = {
        "updated_at_utc": now_iso(),
        "manifest_symbols": len(manifest),
        "raw_payload_jsonl_rows": raw_jsonl_rows,
        "company_rows": len(company_rows),
        "income_statement_rows": len(income_rows),
        "balance_sheet_rows": len(balance_rows),
        "cash_flow_rows": len(cash_flow_rows),
        "income_statement_latest_5y_rows": len(income_latest_5y_rows),
        "balance_sheet_latest_5y_rows": len(balance_latest_5y_rows),
        "cash_flow_latest_5y_rows": len(cash_flow_latest_5y_rows),
        "earnings_rows": len(earnings_rows),
        "payloads_with_transcript_like_keys": len(transcript_key_hits),
        "normalized_dir": str(NORMALIZED_DIR),
    }
    atomic_write_json(OUT_DIR / "normalization_summary.json", summary)
    write_readme(summary)
    return summary


def write_readme(summary: dict[str, Any]) -> None:
    text = f"""---
license: other
pretty_name: EODHD Japan Korea Fundamentals
task_categories:
- tabular-regression
- time-series-forecasting
language:
- en
tags:
- finance
- eodhd
- equities
- fundamentals
- japan
- south-korea
private-dataset: true
configs:
- config_name: companies
  data_files:
  - split: train
    path: normalized/companies.csv
- config_name: income_statement
  data_files:
  - split: train
    path: normalized/income_statement.csv
- config_name: income_statement_latest_5y
  data_files:
  - split: train
    path: normalized/income_statement_latest_5y.csv
- config_name: balance_sheet
  data_files:
  - split: train
    path: normalized/balance_sheet.csv
- config_name: balance_sheet_latest_5y
  data_files:
  - split: train
    path: normalized/balance_sheet_latest_5y.csv
- config_name: cash_flow
  data_files:
  - split: train
    path: normalized/cash_flow.csv
- config_name: cash_flow_latest_5y
  data_files:
  - split: train
    path: normalized/cash_flow_latest_5y.csv
- config_name: earnings
  data_files:
  - split: train
    path: normalized/earnings.csv
---

# EODHD Japan Korea Fundamentals

Private dataset of EODHD Fundamentals API v1.1 payloads for common stocks listed
in Japan and South Korea.

## Contents

- `raw/<symbol>/fundamentals.json`: original EODHD Fundamentals API response.
- `normalized/companies.csv`: company-level metadata, highlights, valuation,
  shares, technicals, and dividend scalar fields.
- `normalized/income_statement.csv`: annual and quarterly income statement rows.
- `normalized/balance_sheet.csv`: annual and quarterly balance sheet rows.
- `normalized/cash_flow.csv`: annual and quarterly cash flow rows.
- `normalized/income_statement_latest_5y.csv`: latest five annual income
  statement rows per symbol.
- `normalized/balance_sheet_latest_5y.csv`: latest five annual balance sheet
  rows per symbol.
- `normalized/cash_flow_latest_5y.csv`: latest five annual cash flow rows per
  symbol.
- `normalized/earnings.csv`: earnings history/trend rows from the fundamentals
  payload.
- `normalized/fundamentals_raw_payloads.jsonl`: raw payloads in JSONL form for
  screening and downstream processing.
- `stock_pull_manifest.csv`: discovered common-stock universe.
- `progress_summary.json` and `progress_checklist.csv`: resumable run status.
- `earnings_transcript_availability.json`: transcript availability check.

## Current Local Summary

```json
{json.dumps(summary, indent=2, sort_keys=True)}
```

## Notes

EODHD documents earnings history/trends and financial statements in the
Fundamentals API, but earnings transcript text is not documented in the public
Fundamentals or Calendar API docs. This pipeline scans returned payload keys for
transcript-like fields and records any hits.
"""
    atomic_write_text(OUT_DIR / "README.md", text)


def fetch_fundamentals(
    manifest: list[dict[str, Any]],
    args: argparse.Namespace,
    token: str,
    rate_limiter: RateLimiter | None = None,
) -> dict[str, Any]:
    selected = manifest[args.offset :]
    if args.pending_only and not args.force:
        selected = [
            item
            for item in selected
            if not (RAW_DIR / safe_name(item["symbol"]) / "fundamentals.json").exists()
        ]
    if args.limit is not None:
        selected = selected[: args.limit]

    run_state = {
        "started_at_utc": now_iso(),
        "offset": args.offset,
        "limit": args.limit,
        "sleep_seconds": args.sleep,
        "max_calls_per_minute": args.max_calls_per_minute,
        "pending_only": args.pending_only,
        "selected_symbols": [item["symbol"] for item in selected],
        "results": [],
    }
    checkpoint_path = OUT_DIR / f"checkpoint_offset_{args.offset}_limit_{args.limit or 'all'}.json"
    atomic_write_json(checkpoint_path, run_state)

    for index, item in enumerate(selected, start=1):
        symbol = item["symbol"]
        encoded_symbol = urllib.parse.quote(symbol)
        url = endpoint_url(FUNDAMENTALS_BASE_URL, f"fundamentals/{encoded_symbol}", token)
        output_path = RAW_DIR / safe_name(symbol) / "fundamentals.json"
        result = fetch_with_cache(url, output_path, args.force, args.retries, args.sleep, rate_limiter)
        run_state["results"].append({"symbol": symbol, "fundamentals": result})
        run_state["updated_at_utc"] = now_iso()
        atomic_write_json(checkpoint_path, run_state)
        atomic_write_json(OUT_DIR / "latest_checkpoint.json", run_state)
        if not args.quiet:
            print(json.dumps({"symbol": symbol, **result}, sort_keys=True))
        if args.progress_every > 0 and index % args.progress_every == 0:
            run_state["progress_summary"] = write_progress(manifest)
            atomic_write_json(checkpoint_path, run_state)
            atomic_write_json(OUT_DIR / "latest_checkpoint.json", run_state)
        time.sleep(args.sleep)

    run_state["completed_at_utc"] = now_iso()
    atomic_write_json(checkpoint_path, run_state)
    atomic_write_json(OUT_DIR / "latest_checkpoint.json", run_state)
    return write_progress(manifest)


def main() -> int:
    args = parse_args()
    country_keys = {key.strip().upper() for key in args.countries.split(",") if key.strip()}
    unknown = sorted(country_keys - set(TARGET_COUNTRIES))
    if unknown:
        raise SystemExit(f"Unsupported country keys: {unknown}. Supported: {sorted(TARGET_COUNTRIES)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.normalize_only:
        manifest = read_manifest()
        if not manifest:
            raise SystemExit("No manifest found. Run discovery/fetch first.")
        print(json.dumps(normalize(manifest), indent=2, sort_keys=True))
        return 0

    token = os.environ.get("EODHD_API_TOKEN")
    if not token:
        raise SystemExit("Set EODHD_API_TOKEN before running.")

    rate_limiter = RateLimiter(args.max_calls_per_minute)
    manifest = discover_manifest(
        token,
        country_keys,
        args.include_delisted,
        args.sleep,
        args.extra_exchange,
        rate_limiter,
    )
    progress = write_progress(manifest)
    if args.status_only:
        print(json.dumps(progress, indent=2, sort_keys=True))
        return 0

    progress = fetch_fundamentals(manifest, args, token, rate_limiter)
    if args.skip_normalize:
        print(json.dumps({"progress": progress, "normalization": "skipped"}, indent=2, sort_keys=True))
        return 0
    normalization_summary = normalize(manifest)
    print(json.dumps({"progress": progress, "normalization": normalization_summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
