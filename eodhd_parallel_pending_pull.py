#!/usr/bin/env python3
"""Parallel pending-only EODHD pull for the local investment dataset.

This is a supervisor for the long all-symbol run. It reads the same manifest as
eodhd_dataset_financial_pull.py, fetches only symbols missing either daily EOD
or fundamentals JSON, and refreshes durable progress/checkpoint files
throughout the run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

from eodhd_dataset_financial_pull import (
    OUT_DIR,
    RAW_DIR,
    atomic_write_json,
    build_manifest,
    endpoint_url,
    maybe_fetch,
    progress_for_symbol,
    safe_name,
    write_manifest,
    write_progress_checklist,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start offset in the manifest.")
    parser.add_argument("--end", type=int, default=None, help="End offset in the manifest.")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel symbol workers.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.05, help="Delay between EOD and fundamentals calls per symbol.")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Refresh checklist after this many completed symbols.")
    parser.add_argument("--max-symbols", type=int, default=None, help="Optional cap for one supervised run.")
    return parser.parse_args()


def raw_symbol_dir(symbol: str) -> Path:
    return RAW_DIR / safe_name(symbol)


def fetch_symbol(item: dict[str, Any], token: str, retries: int, sleep_seconds: float) -> dict[str, Any]:
    symbol = item["symbol"]
    symbol_dir = raw_symbol_dir(symbol)
    encoded_symbol = urllib.parse.quote(symbol)
    eod_url = endpoint_url(
        f"eod/{encoded_symbol}",
        token,
        {"period": "d", "from": item["eod_from"], "to": item["eod_to"]},
    )
    fundamentals_url = endpoint_url(f"fundamentals/{encoded_symbol}", token, {})
    result: dict[str, Any] = {
        "symbol": symbol,
        "company_name": item["company_name"],
        "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    result["eod"] = maybe_fetch("eod", eod_url, symbol_dir / "eod_daily.json", False, retries, sleep_seconds)
    time.sleep(sleep_seconds)
    result["fundamentals"] = maybe_fetch(
        "fundamentals",
        fundamentals_url,
        symbol_dir / "fundamentals.json",
        False,
        retries,
        sleep_seconds,
    )
    result["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
    progress = progress_for_symbol(item)
    result["core_complete"] = progress["core_complete"]
    result["error_count"] = progress["error_count"]
    return result


def main() -> int:
    args = parse_args()
    token = os.environ.get("EODHD_API_TOKEN")
    if not token:
        raise SystemExit("Set EODHD_API_TOKEN before running.")

    manifest = build_manifest()
    write_manifest(manifest)
    end = args.end if args.end is not None else len(manifest)
    scoped = manifest[args.start : end]
    pending = [item for item in scoped if not progress_for_symbol(item)["core_complete"]]
    if args.max_symbols is not None:
        pending = pending[: args.max_symbols]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUT_DIR / "parallel_pending_checkpoint.json"
    results_path = OUT_DIR / "parallel_pending_results.jsonl"
    lock = threading.Lock()
    completed = 0
    failures = 0

    run_state: dict[str, Any] = {
        "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "start": args.start,
        "end": end,
        "workers": args.workers,
        "pending_at_start": len(pending),
        "completed": 0,
        "failures": 0,
        "running": True,
    }
    atomic_write_json(checkpoint_path, run_state)
    write_progress_checklist(manifest)

    def persist(result: dict[str, Any]) -> None:
        nonlocal completed, failures
        with lock:
            completed += 1
            if not result.get("core_complete"):
                failures += 1
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
            run_state.update(
                {
                    "updated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                    "completed": completed,
                    "failures": failures,
                    "last_symbol": result.get("symbol"),
                }
            )
            atomic_write_json(checkpoint_path, run_state)
            if completed % args.checkpoint_every == 0:
                run_state["progress_summary"] = write_progress_checklist(manifest)
                atomic_write_json(checkpoint_path, run_state)

    print(json.dumps({"pending_at_start": len(pending), "workers": args.workers, "start": args.start, "end": end}))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_symbol = {
            executor.submit(fetch_symbol, item, token, args.retries, args.sleep): item["symbol"] for item in pending
        }
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - preserve run details.
                result = {
                    "symbol": symbol,
                    "core_complete": False,
                    "error_count": 1,
                    "exception": f"{type(exc).__name__}: {exc}",
                    "finished_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                }
            persist(result)
            print(json.dumps({"completed": completed, "pending_at_start": len(pending), "symbol": symbol, "core_complete": result.get("core_complete")}))

    run_state["running"] = False
    run_state["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
    run_state["progress_summary"] = write_progress_checklist(manifest)
    atomic_write_json(checkpoint_path, run_state)
    print(json.dumps(run_state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
