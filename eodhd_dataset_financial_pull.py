#!/usr/bin/env python3
"""Fetch EODHD financial data for symbols in the local investment dataset.

The script reads data/processed/investment_canonical.csv, builds a symbol
manifest with publication-date windows, then fetches EOD prices and
fundamentals for a resumable batch. Tokens are read from EODHD_API_TOKEN and
are never written to output files.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT / "data" / "processed" / "investment_canonical.csv"
OUT_DIR = ROOT / "eodhd_output" / "dataset_financial_pull"
RAW_DIR = OUT_DIR / "raw"
BASE_URL = "https://eodhd.com/api"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Number of symbols to fetch from the sorted manifest.")
    parser.add_argument("--offset", type=int, default=0, help="Start offset into the sorted manifest.")
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay between API calls.")
    parser.add_argument("--force", action="store_true", help="Refetch files even when local raw JSON exists.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per API call before recording an error.")
    parser.add_argument(
        "--checklist-every",
        type=int,
        default=10,
        help="Refresh the all-symbol progress checklist after this many symbols.",
    )
    parser.add_argument("--news-limit", type=int, default=0, help="Fetch per-symbol news for the first N fetched symbols.")
    parser.add_argument("--earnings-limit", type=int, default=0, help="Fetch earnings calendar for the first N fetched symbols.")
    parser.add_argument("--status-only", action="store_true", help="Write manifest/checklist and exit without API calls.")
    return parser.parse_args()


def add_years(date_value: dt.date, years: int) -> dt.date:
    try:
        return date_value.replace(year=date_value.year + years)
    except ValueError:
        return date_value.replace(month=2, day=28, year=date_value.year + years)


def build_manifest() -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "min_date": None,
            "max_date": None,
            "horizons": Counter(),
            "companies": Counter(),
            "directions": Counter(),
        }
    )
    with SOURCE_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            symbol = (row.get("eodhd_symbol") or "").strip()
            published = (row.get("publication_date") or "").strip()
            if not symbol or not published:
                continue
            bucket = grouped[symbol]
            bucket["count"] += 1
            bucket["min_date"] = published if bucket["min_date"] is None or published < bucket["min_date"] else bucket["min_date"]
            bucket["max_date"] = published if bucket["max_date"] is None or published > bucket["max_date"] else bucket["max_date"]
            bucket["companies"][row.get("company_name") or ""] += 1
            bucket["directions"]["short" if row.get("is_short") == "True" else "long"] += 1
            for horizon in ("1y", "3y", "5y", "10y", "20y"):
                if row.get(f"raw_perf_{horizon}"):
                    bucket["horizons"][horizon] += 1

    today = dt.date.today()
    manifest: list[dict[str, Any]] = []
    for symbol, bucket in grouped.items():
        max_years = 0
        for horizon, count in bucket["horizons"].items():
            if count:
                max_years = max(max_years, int(horizon[:-1]))
        if max_years == 0:
            max_years = 3
        start_date = dt.date.fromisoformat(bucket["min_date"]) - dt.timedelta(days=30)
        end_date = add_years(dt.date.fromisoformat(bucket["max_date"]), max_years) + dt.timedelta(days=30)
        if end_date > today:
            end_date = today
        manifest.append(
            {
                "symbol": symbol,
                "company_name": bucket["companies"].most_common(1)[0][0],
                "idea_count": bucket["count"],
                "first_publication_date": bucket["min_date"],
                "last_publication_date": bucket["max_date"],
                "max_horizon_years": max_years,
                "eod_from": start_date.isoformat(),
                "eod_to": end_date.isoformat(),
                "long_ideas": bucket["directions"]["long"],
                "short_ideas": bucket["directions"]["short"],
            }
        )
    manifest.sort(key=lambda item: (-item["idea_count"], item["symbol"]))
    return manifest


def write_manifest(manifest: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stock_pull_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (OUT_DIR / "stock_pull_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)


def request_json(url: str) -> tuple[int | None, Any, str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": "SignalValueInvestorWorkbench/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read()
            return response.status, json.loads(raw.decode("utf-8")), None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:1000]
        return exc.code, None, body
    except Exception as exc:  # noqa: BLE001 - preserve operational error details.
        return None, None, f"{type(exc).__name__}: {exc}"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(10):
        try:
            temp_path.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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


def safe_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace("\\", "_").replace(":", "_")


def progress_for_symbol(item: dict[str, Any]) -> dict[str, Any]:
    symbol_dir = RAW_DIR / safe_name(item["symbol"])
    error_files = sorted(path.name for path in symbol_dir.glob("*.error.json")) if symbol_dir.exists() else []
    eod_path = symbol_dir / "eod_daily.json"
    fundamentals_path = symbol_dir / "fundamentals.json"
    news_path = symbol_dir / "news.json"
    earnings_path = symbol_dir / "earnings.json"
    return {
        **item,
        "eod_saved": eod_path.exists(),
        "fundamentals_saved": fundamentals_path.exists(),
        "core_complete": eod_path.exists() and fundamentals_path.exists(),
        "news_saved": news_path.exists(),
        "earnings_saved": earnings_path.exists(),
        "error_count": len(error_files),
        "error_files": ";".join(error_files),
    }


def write_progress_checklist(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [progress_for_symbol(item) for item in manifest]
    total = len(rows)
    core_complete = sum(1 for row in rows if row["core_complete"])
    summary = {
        "updated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "total_symbols": total,
        "core_complete": core_complete,
        "pending_core": total - core_complete,
        "eod_saved": sum(1 for row in rows if row["eod_saved"]),
        "fundamentals_saved": sum(1 for row in rows if row["fundamentals_saved"]),
        "news_saved": sum(1 for row in rows if row["news_saved"]),
        "earnings_saved": sum(1 for row in rows if row["earnings_saved"]),
        "symbols_with_errors": sum(1 for row in rows if row["error_count"]),
    }
    atomic_write_json(OUT_DIR / "progress_summary.json", summary)
    atomic_write_json(OUT_DIR / "progress_checklist.json", rows)
    atomic_write_csv(OUT_DIR / "progress_checklist.csv", rows)
    return summary


def endpoint_url(path: str, token: str, params: dict[str, str]) -> str:
    query = {"api_token": token, "fmt": "json", **params}
    return f"{BASE_URL}/{path}?{urllib.parse.urlencode(query)}"


def maybe_fetch(name: str, url: str, output_path: Path, force: bool, retries: int, sleep_seconds: float) -> dict[str, Any]:
    error_path = output_path.with_suffix(".error.json")
    if output_path.exists() and not force:
        try:
            data = json.loads(output_path.read_text(encoding="utf-8"))
            return {"name": name, "status": "cached", "items": len(data) if isinstance(data, (list, dict)) else None}
        except json.JSONDecodeError:
            pass

    result: dict[str, Any] = {"name": name}
    for attempt in range(retries + 1):
        status_code, data, error = request_json(url)
        result = {"name": name, "http_status": status_code, "attempt": attempt + 1}
        if error is None:
            break
        result["error"] = error
        atomic_write_json(error_path, result)
        if attempt < retries:
            time.sleep(max(sleep_seconds, 0.5) * (attempt + 1))
    else:
        return result

    if data is None:
        return result
    atomic_write_json(output_path, data)
    if error_path.exists():
        error_path.unlink()
    result["items"] = len(data) if isinstance(data, (list, dict)) else None
    result["json_type"] = type(data).__name__
    return result


def checkpoint(run_summary: dict[str, Any], path: Path) -> None:
    run_summary["updated_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
    atomic_write_json(path, run_summary)
    atomic_write_json(OUT_DIR / "latest_checkpoint.json", run_summary)


def main() -> int:
    args = parse_args()
    manifest = build_manifest()
    write_manifest(manifest)
    progress_summary = write_progress_checklist(manifest)
    if args.status_only:
        print(json.dumps(progress_summary, indent=2))
        return 0

    token = os.environ.get("EODHD_API_TOKEN")
    if not token:
        raise SystemExit("Set EODHD_API_TOKEN before running.")

    selected = manifest[args.offset : args.offset + args.limit]

    run_summary: dict[str, Any] = {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "source_csv": str(SOURCE_CSV),
        "unique_symbols_total": len(manifest),
        "offset": args.offset,
        "limit": args.limit,
        "selected_symbols": [item["symbol"] for item in selected],
        "results": [],
    }
    checkpoint_path = OUT_DIR / f"checkpoint_offset_{args.offset}_limit_{args.limit}.json"
    checkpoint(run_summary, checkpoint_path)

    for index, item in enumerate(selected):
        symbol = item["symbol"]
        symbol_dir = RAW_DIR / safe_name(symbol)
        encoded_symbol = urllib.parse.quote(symbol)
        eod_url = endpoint_url(
            f"eod/{encoded_symbol}",
            token,
            {"period": "d", "from": item["eod_from"], "to": item["eod_to"]},
        )
        fundamentals_url = endpoint_url(f"fundamentals/{encoded_symbol}", token, {})
        symbol_result: dict[str, Any] = {"symbol": symbol, "company_name": item["company_name"], "window": item}
        symbol_result["eod"] = maybe_fetch("eod", eod_url, symbol_dir / "eod_daily.json", args.force, args.retries, args.sleep)
        run_summary["results"].append(symbol_result)
        checkpoint(run_summary, checkpoint_path)
        time.sleep(args.sleep)
        symbol_result["fundamentals"] = maybe_fetch(
            "fundamentals", fundamentals_url, symbol_dir / "fundamentals.json", args.force, args.retries, args.sleep
        )
        checkpoint(run_summary, checkpoint_path)
        time.sleep(args.sleep)

        if index < args.news_limit:
            news_url = endpoint_url("news", token, {"s": symbol, "limit": "10"})
            symbol_result["news"] = maybe_fetch("news", news_url, symbol_dir / "news.json", args.force, args.retries, args.sleep)
            checkpoint(run_summary, checkpoint_path)
            time.sleep(args.sleep)
        if index < args.earnings_limit:
            earnings_url = endpoint_url(
                "calendar/earnings",
                token,
                {"symbols": symbol, "from": item["eod_from"], "to": item["eod_to"]},
            )
            symbol_result["earnings"] = maybe_fetch(
                "earnings", earnings_url, symbol_dir / "earnings.json", args.force, args.retries, args.sleep
            )
            checkpoint(run_summary, checkpoint_path)
            time.sleep(args.sleep)

        print(json.dumps({"done": len(run_summary["results"]), "symbol": symbol, "eod": symbol_result["eod"], "fundamentals": symbol_result["fundamentals"]}))
        if args.checklist_every > 0 and len(run_summary["results"]) % args.checklist_every == 0:
            write_progress_checklist(manifest)

    summary_path = OUT_DIR / f"pull_summary_offset_{args.offset}_limit_{args.limit}.json"
    run_summary["completed_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
    atomic_write_json(summary_path, run_summary)
    checkpoint(run_summary, checkpoint_path)
    progress_summary = write_progress_checklist(manifest)
    print(json.dumps(progress_summary, indent=2))
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
