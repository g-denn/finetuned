"""
Rebuild split-adjusted Yahoo Finance performance labels for VIC ideas.

This script does not trust or overwrite public.performance. It writes an audit
layer to public.performance_yahoo, using Yahoo chart API adjusted closes.

Required environment:
  SUPABASE_SERVICE_ROLE_KEY=<service role key>

Optional environment:
  SUPABASE_URL=https://<project-ref>.supabase.co
  SUPABASE_PROJECT_REF=<project-ref>

Examples:
  python fetch_yahoo_performance_adjusted.py --limit 20 --dry-run
  python fetch_yahoo_performance_adjusted.py --apply --only-missing
  python fetch_yahoo_performance_adjusted.py --apply --refresh-errors --batch-size 100
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from dateutil.relativedelta import relativedelta


PROJECT_REF = os.environ.get("SUPABASE_PROJECT_REF", "aqfpldvpcoyipkyihuea")
SUPABASE_URL = os.environ.get("SUPABASE_URL", f"https://{PROJECT_REF}.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

FETCH_PAGE = 1000
DEFAULT_BATCH_SIZE = 100
DEFAULT_SLEEP = 0.25
MAX_RETRIES = 3

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

UPSERT_COLUMNS = [
    "idea_id",
    "raw_symbol",
    "yahoo_symbol",
    "source_status",
    "source_error",
    "publication_date",
    "base_trade_date",
    "base_close",
    "base_adj_close",
    "base_adjustment_factor",
    "next_trade_date",
    "next_open",
    "next_close",
    "next_adj_close",
]
for _label in TIMEFRAMES:
    UPSERT_COLUMNS.extend(
        [
            f"price_{_label}",
            f"adj_price_{_label}",
            f"perf_{_label}",
            f"short_perf_{_label}",
            f"trade_date_{_label}",
        ]
    )
UPSERT_COLUMNS.extend(
    [
        "split_events",
        "dividend_events",
        "checked_at",
        "yahoo_payload_range",
        "updated_at",
    ]
)

BLOOMBERG_TO_YAHOO = {
    "US": "",
    "UN": "",
    "UQ": "",
    "UW": "",
    "UA": "",
    "KS": ".KS",
    "KQ": ".KQ",
    "KP": ".KS",
    "JT": ".T",
    "JP": ".T",
    "HK": ".HK",
    "AU": ".AX",
    "NZ": ".NZ",
    "SP": ".SI",
    "MK": ".KL",
    "IJ": ".JK",
    "PM": ".PS",
    "TB": ".BK",
    "TT": ".TW",
    "CH": ".SS",
    "CZ": ".SZ",
    "IN": ".NS",
    "IB": ".BO",
    "LN": ".L",
    "LI": ".L",
    "GR": ".DE",
    "GY": ".DE",
    "FP": ".PA",
    "NA": ".AS",
    "BB": ".BR",
    "SW": ".SW",
    "SE": ".ST",
    "SS": ".ST",
    "NO": ".OL",
    "DC": ".CO",
    "FH": ".HE",
    "SM": ".MC",
    "IM": ".MI",
    "IT": ".MI",
    "PW": ".LS",
    "PL": ".WA",
    "IR": ".IR",
    "ID": ".IR",
    "AT": ".VI",
    "CN": ".TO",
    "CT": ".TO",
    "CV": ".V",
    "CF": ".CN",
    "MX": ".MX",
    "BZ": ".SA",
    "IS": ".IS",
    "SA": ".SR",
    "DU": ".DU",
    "AD": ".AE",
    "EY": ".CA",
    "SJ": ".JO",
}

SEPARATOR_EXCHANGE_ALIASES = {
    "ASX": ".AX",
    "AU": ".AX",
    "CN": ".TO",
    "HK": ".HK",
    "JP": ".T",
    "JT": ".T",
    "KQ": ".KQ",
    "KRX": ".KS",
    "KS": ".KS",
    "LN": ".L",
    "LON": ".L",
    "LSE": ".L",
    "SIX": ".SW",
    "TK": ".T",
    "TKS": ".T",
    "TO": ".TO",
    "TSX": ".TO",
    "TW": ".TW",
}

KNOWN_DOT_SUFFIX_FIXES = {
    ".LN": ".L",
    ".JT": ".T",
    ".JP": ".T",
    ".AU": ".AX",
    ".CN": ".TO",
    ".SS": ".ST",
}

SKIP_TOKENS = {
    "BOND",
    "CORP",
    "CREDIT",
    "GOVT",
    "MUNI",
    "PFD",
    "PREF",
    "RIGHT",
    "RT",
    "WARRANT",
    "WT",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("fetch_yahoo_performance_adjusted.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Bar:
    trade_date: date
    open: float | None
    close: float | None
    adj_close: float | None


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def parse_date(value: str) -> date:
    return date.fromisoformat(value[:10])


def unix_day(value: date) -> int:
    return int(datetime(value.year, value.month, value.day, tzinfo=timezone.utc).timestamp())


def http_json(method: str, url: str, headers: dict[str, str] | None = None, body: Any = None) -> tuple[Any, dict[str, str]]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, method=method, data=data, headers=headers or {})
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = resp.read().decode("utf-8")
                parsed = json.loads(payload) if payload else None
                return parsed, {k.lower(): v for k, v in resp.headers.items()}
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            if exc.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"{method} {url} failed HTTP {exc.code}: {text[:500]}") from exc
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(2**attempt)
                continue
    raise RuntimeError(f"{method} {url} failed: {last_error}") from last_error


def supabase_headers(prefer: str | None = None) -> dict[str, str]:
    if not SUPABASE_KEY:
        raise SystemExit("Set SUPABASE_SERVICE_ROLE_KEY before running this script.")
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "apikey": SUPABASE_KEY,
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def rest_url(table: str, query: dict[str, str]) -> str:
    return f"{SUPABASE_URL}/rest/v1/{table}?" + urllib.parse.urlencode(query, safe="(),.*")


def fetch_existing_yahoo_ids(status_filter: str | None = None) -> set[str]:
    ids: set[str] = set()
    offset = 0
    while True:
        query = {
            "select": "idea_id,source_status",
            "order": "idea_id.asc",
            "limit": str(FETCH_PAGE),
            "offset": str(offset),
        }
        if status_filter:
            query["source_status"] = status_filter
        rows, _ = http_json("GET", rest_url("performance_yahoo", query), supabase_headers())
        if not rows:
            break
        ids.update(row["idea_id"] for row in rows)
        offset += len(rows)
        if len(rows) < FETCH_PAGE:
            break
    return ids


def fetch_ideas(limit: int | None, only_ids: set[str] | None) -> list[dict[str, Any]]:
    ideas: list[dict[str, Any]] = []
    offset = 0
    while True:
        page_size = min(FETCH_PAGE, limit - len(ideas)) if limit else FETCH_PAGE
        if page_size <= 0:
            break
        rows, _ = http_json(
            "GET",
            rest_url(
                "ideas",
                {
                    "select": "id,company_id,date,is_short",
                    "order": "company_id.asc,date.asc,id.asc",
                    "limit": str(page_size),
                    "offset": str(offset),
                },
            ),
            supabase_headers(),
        )
        if not rows:
            break
        for row in rows:
            if only_ids is None or row["id"] in only_ids:
                ideas.append(row)
        offset += len(rows)
        log.info("Fetched %s ideas so far...", len(ideas))
        if len(rows) < page_size or (limit and len(ideas) >= limit):
            break
    return ideas[:limit] if limit else ideas


def normalize_symbol(raw: str | None) -> tuple[str | None, str | None]:
    if not raw:
        return None, "missing_symbol"
    value = " ".join(str(raw).strip().split())
    if not value:
        return None, "missing_symbol"

    upper = value.upper()
    if any(token in upper.split() for token in SKIP_TOKENS):
        return None, "unsupported_instrument"

    if " " in value:
        base, exchange = value.rsplit(" ", 1)
        base = clean_symbol_base(base)
        exchange = exchange.strip().upper()
        suffix = BLOOMBERG_TO_YAHOO.get(exchange)
        if suffix is None:
            return None, f"unknown_exchange:{exchange}"
        if exchange == "HK" and base.isdigit():
            base = base.zfill(4)
        return base + suffix, None

    separator_match = re.fullmatch(r"(.+?)[-:](\(?[A-Za-z]{2,5}\)?)", value.strip())
    if separator_match:
        base = clean_symbol_base(separator_match.group(1))
        exchange = separator_match.group(2).strip("()").upper()
        suffix = SEPARATOR_EXCHANGE_ALIASES.get(exchange)
        if suffix:
            if suffix == ".HK" and base.isdigit():
                base = base.zfill(4)
            return base + suffix, None

    if looks_like_cusip(value):
        return None, "unsupported_identifier"

    symbol = clean_symbol_base(value)
    compact_hk = re.fullmatch(r"(\d{1,4})HK", symbol)
    if compact_hk:
        return compact_hk.group(1).zfill(4) + ".HK", None

    if symbol.isdigit():
        return None, "missing_exchange_for_numeric_symbol"
    for wrong, right in KNOWN_DOT_SUFFIX_FIXES.items():
        if symbol.endswith(wrong):
            symbol = symbol[: -len(wrong)] + right
            break
    if symbol.endswith(".HK"):
        base = symbol[:-3]
        if base.isdigit():
            symbol = base.zfill(4) + ".HK"
    return symbol, None


def clean_symbol_base(value: str) -> str:
    symbol = value.strip().upper()
    symbol = symbol.strip("$")
    if symbol.startswith("(") and symbol.endswith(")"):
        symbol = symbol[1:-1]
    return symbol.replace("/", "-").rstrip(".")


def looks_like_cusip(value: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    if not re.fullmatch(r"[A-Z0-9]{9}", compact):
        return False
    return any(ch.isdigit() for ch in compact) and any(ch.isalpha() for ch in compact)


def yahoo_chart(symbol: str, start: date, end: date) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "period1": unix_day(start),
            "period2": unix_day(end + timedelta(days=1)),
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?{params}"
    payload, _ = http_json("GET", url, {"User-Agent": "vic-finetuning-data-repair/1.0"})
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(json.dumps(chart["error"]))
    results = chart.get("result") or []
    if not results:
        raise RuntimeError("empty_yahoo_result")
    return results[0]


def candidate_symbols(symbol: str) -> list[str]:
    candidates = [symbol]
    if symbol.endswith(".KS"):
        candidates.append(symbol[:-3] + ".KQ")
    elif symbol.endswith(".KQ"):
        candidates.append(symbol[:-3] + ".KS")
    return candidates


def parse_bars(result: dict[str, Any]) -> list[Bar]:
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    adj = ((result.get("indicators") or {}).get("adjclose") or [{}])[0]
    opens = quote.get("open") or []
    closes = quote.get("close") or []
    adj_closes = adj.get("adjclose") or []
    bars: list[Bar] = []
    for idx, ts in enumerate(timestamps):
        bars.append(
            Bar(
                trade_date=datetime.fromtimestamp(ts, timezone.utc).date(),
                open=clean_float(opens[idx] if idx < len(opens) else None),
                close=clean_float(closes[idx] if idx < len(closes) else None),
                adj_close=clean_float(adj_closes[idx] if idx < len(adj_closes) else None),
            )
        )
    return [bar for bar in bars if bar.close is not None and bar.adj_close is not None]


def parse_events(result: dict[str, Any], event_name: str) -> list[dict[str, Any]]:
    events = ((result.get("events") or {}).get(event_name) or {}).values()
    parsed = []
    for event in events:
        event_date = datetime.fromtimestamp(event["date"], timezone.utc).date().isoformat()
        item = {"date": event_date}
        for key, value in event.items():
            if key != "date":
                item[key] = value
        parsed.append(item)
    return sorted(parsed, key=lambda row: row["date"])


def bar_on_or_after(bars: list[Bar], target: date) -> Bar | None:
    for bar in bars:
        if bar.trade_date >= target:
            return bar
    return None


def compute_row(idea: dict[str, Any], symbol: str, bars: list[Bar], result: dict[str, Any]) -> dict[str, Any]:
    pub_date = parse_date(idea["date"])
    base = bar_on_or_after(bars, pub_date)
    if base is None or base.adj_close is None or base.adj_close <= 0:
        return error_row(idea, symbol, "missing_base_price", None)

    next_bar = bar_on_or_after(bars, pub_date + timedelta(days=1))
    row: dict[str, Any] = {
        "idea_id": idea["id"],
        "raw_symbol": idea.get("company_id"),
        "yahoo_symbol": symbol,
        "source_status": "ok",
        "source_error": None,
        "publication_date": pub_date.isoformat(),
        "base_trade_date": base.trade_date.isoformat(),
        "base_close": base.close,
        "base_adj_close": base.adj_close,
        "base_adjustment_factor": base.adj_close / base.close if base.close else None,
        "next_trade_date": next_bar.trade_date.isoformat() if next_bar else None,
        "next_open": next_bar.open if next_bar else None,
        "next_close": next_bar.close if next_bar else None,
        "next_adj_close": next_bar.adj_close if next_bar else None,
        "split_events": parse_events(result, "splits"),
        "dividend_events": parse_events(result, "dividends"),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "yahoo_payload_range": None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    for label, delta in TIMEFRAMES.items():
        target = pub_date + delta
        future = bar_on_or_after(bars, target)
        if future is None or future.adj_close is None or future.adj_close <= 0:
            row[f"price_{label}"] = None
            row[f"adj_price_{label}"] = None
            row[f"perf_{label}"] = None
            row[f"short_perf_{label}"] = None
            row[f"trade_date_{label}"] = None
            continue
        ratio = future.adj_close / base.adj_close
        row[f"price_{label}"] = future.close
        row[f"adj_price_{label}"] = future.adj_close
        row[f"perf_{label}"] = ratio
        row[f"short_perf_{label}"] = 1 / ratio if ratio else None
        row[f"trade_date_{label}"] = future.trade_date.isoformat()
    return row


def error_row(idea: dict[str, Any], symbol: str | None, status: str, detail: str | None) -> dict[str, Any]:
    return {
        "idea_id": idea["id"],
        "raw_symbol": idea.get("company_id"),
        "yahoo_symbol": symbol,
        "source_status": status,
        "source_error": detail[:1000] if detail else None,
        "publication_date": parse_date(idea["date"]).isoformat() if idea.get("date") else None,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def upsert_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    normalized = [normalize_upsert_row(row) for row in rows]
    _, _ = http_json(
        "POST",
        f"{SUPABASE_URL}/rest/v1/performance_yahoo",
        supabase_headers("resolution=merge-duplicates,return=minimal"),
        normalized,
    )


def normalize_upsert_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {column: row.get(column) for column in UPSERT_COLUMNS}
    normalized["split_events"] = normalized["split_events"] or []
    normalized["dividend_events"] = normalized["dividend_events"] or []
    return normalized


def flush(rows: list[dict[str, Any]], batch_size: int, dry_run: bool) -> int:
    written = 0
    while len(rows) >= batch_size:
        batch = rows[:batch_size]
        del rows[:batch_size]
        if dry_run:
            log.info("Dry run batch sample: %s", json.dumps(batch[:2], indent=2)[:3000])
        else:
            upsert_rows(batch)
        written += len(batch)
    return written


def group_ideas(ideas: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    immediate_errors: list[dict[str, Any]] = []
    for idea in ideas:
        symbol, error = normalize_symbol(idea.get("company_id"))
        if error:
            immediate_errors.append(error_row(idea, symbol, error, None))
        else:
            idea["_yahoo_symbol"] = symbol
            grouped[symbol].append(idea)
    return grouped, immediate_errors


def select_ids(args: argparse.Namespace) -> set[str] | None:
    if args.refresh_errors:
        statuses = [
            "empty_yahoo_result",
            "missing_base_price",
            "yahoo_error",
            "network_error",
        ]
        ids: set[str] = set()
        for status in statuses:
            ids.update(fetch_existing_yahoo_ids(f"eq.{status}"))
        ids.update(fetch_existing_yahoo_ids("like.unknown_exchange:%"))
        ids.update(fetch_existing_yahoo_ids("eq.missing_exchange_for_numeric_symbol"))
        log.info("Refreshing %s prior error rows.", len(ids))
        return ids
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write to Supabase. Default is dry-run.")
    parser.add_argument("--dry-run", action="store_true", help="Print sample rows without writing.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--refresh-errors", action="store_true")
    args = parser.parse_args()

    dry_run = not args.apply or args.dry_run
    if args.apply and args.dry_run:
        raise SystemExit("Use either --apply or --dry-run, not both.")

    if not dry_run and not SUPABASE_KEY:
        raise SystemExit("Set SUPABASE_SERVICE_ROLE_KEY before using --apply.")

    only_ids = select_ids(args)
    ideas = fetch_ideas(args.limit, only_ids)
    if args.only_missing:
        existing = fetch_existing_yahoo_ids()
        ideas = [idea for idea in ideas if idea["id"] not in existing]

    grouped, pending = group_ideas(ideas)
    log.info("Ideas selected: %s", len(ideas))
    log.info("Yahoo symbols selected: %s", len(grouped))
    log.info("Immediate symbol errors: %s", len(pending))

    written = flush(pending, args.batch_size, dry_run)
    today = date.today()

    for index, (symbol, symbol_ideas) in enumerate(sorted(grouped.items()), start=1):
        dates = [parse_date(idea["date"]) for idea in symbol_ideas]
        start = min(dates) - timedelta(days=10)
        end = min(max(dates) + relativedelta(years=5, days=14), today)
        log.info("[%s/%s] %s: %s ideas, %s to %s", index, len(grouped), symbol, len(symbol_ideas), start, end)
        try:
            result = None
            bars: list[Bar] = []
            resolved_symbol = symbol
            last_error: Exception | None = None
            for candidate in candidate_symbols(symbol):
                try:
                    candidate_result = yahoo_chart(candidate, start, end)
                    candidate_bars = parse_bars(candidate_result)
                    if candidate_bars:
                        result = candidate_result
                        bars = candidate_bars
                        resolved_symbol = candidate
                        break
                    last_error = RuntimeError("empty_yahoo_result")
                except Exception as exc:
                    last_error = exc
            if result is None or not bars:
                raise last_error or RuntimeError("empty_yahoo_result")
            for idea in symbol_ideas:
                pending.append(compute_row(idea, resolved_symbol, bars, result))
        except urllib.error.URLError as exc:
            for idea in symbol_ideas:
                pending.append(error_row(idea, symbol, "network_error", str(exc)))
        except Exception as exc:
            for idea in symbol_ideas:
                pending.append(error_row(idea, symbol, "yahoo_error", str(exc)))

        written += flush(pending, args.batch_size, dry_run)
        time.sleep(args.sleep)

    if pending:
        if dry_run:
            log.info("Dry run final sample: %s", json.dumps(pending[:2], indent=2)[:3000])
        else:
            upsert_rows(pending)
        written += len(pending)

    log.info("Done. %s rows %s.", written, "prepared" if dry_run else "written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
