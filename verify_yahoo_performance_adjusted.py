"""
Audit public.performance_yahoo against a fresh Yahoo Finance chart fetch.

This is intentionally read-only. It samples rows marked source_status='ok',
re-fetches Yahoo adjusted daily bars for the publication date through five
years later, and verifies that stored multipliers equal:

    future_adjusted_close / base_adjusted_close

Required environment:
  SUPABASE_SERVICE_ROLE_KEY=<service role key>

Optional environment:
  SUPABASE_URL=https://<project-ref>.supabase.co
  SUPABASE_PROJECT_REF=<project-ref>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Any

from dateutil.relativedelta import relativedelta

from fetch_yahoo_performance_adjusted import (
    Bar,
    bar_on_or_after,
    candidate_symbols,
    parse_bars,
    parse_date,
    unix_day,
    yahoo_chart,
)


PROJECT_REF = os.environ.get("SUPABASE_PROJECT_REF", "aqfpldvpcoyipkyihuea")
SUPABASE_URL = os.environ.get("SUPABASE_URL", f"https://{PROJECT_REF}.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

TIMEFRAMES: dict[str, Any] = {
    "1w": timedelta(weeks=1),
    "2w": timedelta(weeks=2),
    "1m": relativedelta(months=1),
    "3m": relativedelta(months=3),
    "6m": relativedelta(months=6),
    "1y": relativedelta(years=1),
    "2y": relativedelta(years=2),
    "3y": relativedelta(years=3),
    "5y": relativedelta(years=5),
}

SELECT_COLUMNS = ",".join(
    [
        "idea_id",
        "raw_symbol",
        "yahoo_symbol",
        "publication_date",
        "base_trade_date",
        "base_adj_close",
        "next_trade_date",
        "next_adj_close",
    ]
    + [
        item
        for label in TIMEFRAMES
        for item in (f"adj_price_{label}", f"perf_{label}", f"short_perf_{label}", f"trade_date_{label}")
    ]
)


def maybe_load_local_service_key() -> None:
    """Keep a single secret source: SUPABASE_SERVICE_ROLE_KEY."""
    global SUPABASE_KEY
    SUPABASE_KEY = SUPABASE_KEY or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")


def supabase_get(path: str, query: dict[str, str], headers: dict[str, str] | None = None) -> tuple[Any, dict[str, str]]:
    if not SUPABASE_KEY:
        raise RuntimeError("Set SUPABASE_SERVICE_ROLE_KEY before running this verifier.")
    url = f"{SUPABASE_URL}/rest/v1/{path}?{urllib.parse.urlencode(query)}"
    req_headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = resp.read().decode("utf-8")
        return json.loads(payload) if payload else None, {k.lower(): v for k, v in resp.headers.items()}


def count_ok_rows() -> int:
    _, headers = supabase_get(
        "performance_yahoo",
        {"select": "idea_id", "source_status": "eq.ok", "limit": "1"},
        {"Prefer": "count=exact"},
    )
    content_range = headers.get("content-range", "")
    if "/" not in content_range:
        raise RuntimeError(f"Could not read row count from content-range: {content_range!r}")
    return int(content_range.rsplit("/", 1)[1])


def fetch_sample(sample_size: int, seed: int) -> list[dict[str, Any]]:
    total = count_ok_rows()
    if total == 0:
        return []
    rng = random.Random(seed)
    offsets = sorted(rng.sample(range(total), min(sample_size, total)))
    rows: list[dict[str, Any]] = []
    for offset in offsets:
        page, _ = supabase_get(
            "performance_yahoo",
            {
                "select": SELECT_COLUMNS,
                "source_status": "eq.ok",
                "order": "idea_id.asc",
                "limit": "1",
                "offset": str(offset),
            },
        )
        if page:
            rows.append(page[0])
    return rows


def fetch_all_ok(page_size: int = 1000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page, _ = supabase_get(
            "performance_yahoo",
            {
                "select": SELECT_COLUMNS,
                "source_status": "eq.ok",
                "order": "idea_id.asc",
                "limit": str(page_size),
                "offset": str(offset),
            },
        )
        if not page:
            break
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return rows


def fresh_bars(symbol: str, publication_date: date) -> tuple[str, list[Bar]]:
    start = publication_date - timedelta(days=10)
    end = min(publication_date + relativedelta(years=5, days=14), date.today())
    last_error: Exception | None = None
    for candidate in candidate_symbols(symbol):
        try:
            bars = parse_bars(yahoo_chart(candidate, start, end))
            if bars:
                return candidate, bars
            last_error = RuntimeError("empty_yahoo_result")
        except Exception as exc:
            last_error = exc
    raise RuntimeError(str(last_error or "empty_yahoo_result"))


def close_enough(left: float | None, right: float | None, tolerance: float) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if not (math.isfinite(left) and math.isfinite(right)):
        return False
    return abs(left - right) <= max(tolerance, tolerance * abs(right))


def audit_row(row: dict[str, Any], tolerance: float) -> dict[str, Any]:
    pub_date = parse_date(row["publication_date"])
    resolved_symbol, bars = fresh_bars(row["yahoo_symbol"], pub_date)
    base = bar_on_or_after(bars, pub_date)
    errors: list[str] = []
    checked = 0

    if base is None or base.adj_close is None or base.adj_close <= 0:
        return {
            "idea_id": row["idea_id"],
            "symbol": row["yahoo_symbol"],
            "resolved_symbol": resolved_symbol,
            "ok": False,
            "checked": checked,
            "errors": ["fresh_missing_base_price"],
        }

    checked += 1
    if str(base.trade_date) != str(row.get("base_trade_date")):
        errors.append(f"base_trade_date stored={row.get('base_trade_date')} fresh={base.trade_date}")
    if not close_enough(row.get("base_adj_close"), base.adj_close, tolerance):
        errors.append(f"base_adj_close stored={row.get('base_adj_close')} fresh={base.adj_close}")

    for label, delta in TIMEFRAMES.items():
        future = bar_on_or_after(bars, pub_date + delta)
        stored_adj = row.get(f"adj_price_{label}")
        stored_perf = row.get(f"perf_{label}")
        stored_short = row.get(f"short_perf_{label}")
        stored_date = row.get(f"trade_date_{label}")

        if future is None or future.adj_close is None or future.adj_close <= 0:
            expected_adj = None
            expected_perf = None
            expected_short = None
            expected_date = None
        else:
            expected_adj = future.adj_close
            expected_perf = future.adj_close / base.adj_close
            expected_short = 1 / expected_perf if expected_perf else None
            expected_date = str(future.trade_date)

        checked += 3
        if str(stored_date) != str(expected_date):
            errors.append(f"trade_date_{label} stored={stored_date} fresh={expected_date}")
        if not close_enough(stored_adj, expected_adj, tolerance):
            errors.append(f"adj_price_{label} stored={stored_adj} fresh={expected_adj}")
        if not close_enough(stored_perf, expected_perf, tolerance):
            errors.append(f"perf_{label} stored={stored_perf} fresh={expected_perf}")
        if not close_enough(stored_short, expected_short, tolerance):
            errors.append(f"short_perf_{label} stored={stored_short} fresh={expected_short}")

    return {
        "idea_id": row["idea_id"],
        "raw_symbol": row.get("raw_symbol"),
        "symbol": row["yahoo_symbol"],
        "resolved_symbol": resolved_symbol,
        "publication_date": row["publication_date"],
        "ok": not errors,
        "checked": checked,
        "errors": errors,
    }


def audit_row_internal(row: dict[str, Any], tolerance: float) -> dict[str, Any]:
    errors: list[str] = []
    checked = 0
    base_adj = row.get("base_adj_close")
    if base_adj is None or base_adj <= 0:
        return {
            "idea_id": row["idea_id"],
            "raw_symbol": row.get("raw_symbol"),
            "symbol": row.get("yahoo_symbol"),
            "publication_date": row.get("publication_date"),
            "ok": False,
            "checked": checked,
            "errors": [f"invalid_base_adj_close: {base_adj}"],
        }

    for label in TIMEFRAMES:
        stored_adj = row.get(f"adj_price_{label}")
        stored_perf = row.get(f"perf_{label}")
        stored_short = row.get(f"short_perf_{label}")
        checked += 2

        if stored_adj is None and stored_perf is None and stored_short is None:
            continue
        expected_perf = stored_adj / base_adj if stored_adj is not None else None
        expected_short = 1 / expected_perf if expected_perf else None
        if not close_enough(stored_perf, expected_perf, tolerance):
            errors.append(f"perf_{label} stored={stored_perf} expected={expected_perf}")
        if not close_enough(stored_short, expected_short, tolerance):
            errors.append(f"short_perf_{label} stored={stored_short} expected={expected_short}")

    return {
        "idea_id": row["idea_id"],
        "raw_symbol": row.get("raw_symbol"),
        "symbol": row.get("yahoo_symbol"),
        "publication_date": row.get("publication_date"),
        "ok": not errors,
        "checked": checked,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--internal-only", action="store_true", help="Do not call Yahoo; verify stored formula consistency only.")
    parser.add_argument("--all", action="store_true", help="Audit every source_status='ok' row instead of a sample.")
    args = parser.parse_args()

    maybe_load_local_service_key()
    started = datetime.now(timezone.utc)
    rows = fetch_all_ok() if args.all else fetch_sample(args.sample_size, args.seed)
    results = []
    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {row['yahoo_symbol']} {row['publication_date']} {row['idea_id']}")
        try:
            if args.internal_only:
                results.append(audit_row_internal(row, args.tolerance))
            else:
                results.append(audit_row(row, args.tolerance))
        except Exception as exc:
            results.append(
                {
                    "idea_id": row.get("idea_id"),
                    "raw_symbol": row.get("raw_symbol"),
                    "symbol": row.get("yahoo_symbol"),
                    "publication_date": row.get("publication_date"),
                    "ok": False,
                    "checked": 0,
                    "errors": [f"verification_fetch_error: {exc}"],
                }
            )

    failed = [result for result in results if not result["ok"]]
    checked_values = sum(result["checked"] for result in results)
    summary = {
        "started_at": started.isoformat(),
        "sample_size": len(rows),
        "passed_rows": len(rows) - len(failed),
        "failed_rows": len(failed),
        "checked_values": checked_values,
        "mode": "internal_only" if args.internal_only else "fresh_yahoo",
        "tolerance": args.tolerance,
        "failures": failed[:20],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
