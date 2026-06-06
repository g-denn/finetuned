#!/usr/bin/env python3
"""Build high-confidence provider-error repair ideas from cached delisted archives."""

from __future__ import annotations

import csv
import difflib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path("eodhd_output/full_run")
VALIDATION_CSV = BASE_DIR / "validation_results_with_provider_error_yahoo_identity.csv"
IDEAS_JSON = Path("eodhd_output/all_ideas.json")
OUT_JSON = Path("eodhd_output/provider_error_delisted_archive_repair_ideas.json")
SUMMARY_JSON = BASE_DIR / "provider_error_delisted_archive_repair_summary.json"

MIN_MATCH_SCORE = 0.72
COMMON_TYPES = {"common stock", "common share", "ordinary share", "ordinary shares"}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_name(value: str | None) -> str:
    text = (value or "").lower()
    text = re.sub(
        r"\b(inc|corp|corporation|co|company|ltd|limited|plc|sa|ag|nv|se|ab|adr|holdings?|group|the|class|cl|ord|common|stock)\b",
        " ",
        text,
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def name_score(left: str | None, right: str | None) -> float:
    a = normalize_name(left)
    b = normalize_name(right)
    if not a or not b:
        return 0.0
    sequence = difflib.SequenceMatcher(None, a, b).ratio()
    left_tokens = set(a.split())
    right_tokens = set(b.split())
    jaccard = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    return max(sequence, jaccard)


def code_from_raw(raw_symbol: str | None, eodhd_symbol: str | None) -> str:
    raw = (raw_symbol or eodhd_symbol or "").upper().strip()
    raw = raw.replace("-", " ").replace(".", " ")
    parts = raw.split()
    return parts[0] if parts else ""


def hint_exchanges(raw_symbol: str | None, yahoo_symbol: str | None) -> set[str]:
    combo = f"{raw_symbol or ''} {yahoo_symbol or ''}".upper()
    mapping = {
        " KS": {"KO", "KQ"},
        ".KS": {"KO", "KQ"},
        " JP": {"T", "TSE"},
        ".T": {"T", "TSE"},
        " JT": {"T", "TSE"},
        " LN": {"LSE"},
        ".L": {"LSE"},
        " HK": {"HK"},
        ".HK": {"HK"},
        " CN": {"TO", "V"},
        ".TO": {"TO"},
        ".V": {"V"},
        " GR": {"F", "XETRA"},
        ".DE": {"F", "XETRA"},
        " GY": {"F", "XETRA"},
        " SJ": {"JSE"},
        ".JO": {"JSE"},
        " AU": {"AU"},
        ".AX": {"AU"},
    }
    hints: set[str] = set()
    for marker, exchanges in mapping.items():
        if marker in combo:
            hints.update(exchanges)
    return hints


def eodhd_symbol_from_record(record: dict[str, Any], archive_exchange: str) -> str | None:
    code = str(record.get("Code") or "").upper().strip()
    if not code:
        return None
    country = str(record.get("Country") or "").upper()
    exchange = str(record.get("Exchange") or archive_exchange).upper().strip()
    if country == "USA" or archive_exchange == "US" or exchange in {"NYSE", "NASDAQ", "NYSE MKT", "NYSE ARCA", "OTC", "PINK"}:
        return f"{code}.US"
    return f"{code}.{exchange}"


def load_archives() -> dict[str, list[tuple[str, dict[str, Any]]]]:
    by_code: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for path in BASE_DIR.glob("delisted_symbols_*.json"):
        if path.name.endswith(".error.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, list):
            continue
        archive_exchange = path.stem.replace("delisted_symbols_", "").upper()
        for record in payload:
            if not isinstance(record, dict):
                continue
            code = str(record.get("Code") or "").upper().strip()
            if code:
                by_code[code].append((archive_exchange, record))
    return by_code


def main() -> int:
    rows = load_csv(VALIDATION_CSV)
    ideas = json.loads(IDEAS_JSON.read_text(encoding="utf-8"))
    ideas_by_id = {str(row.get("idea_id") or ""): row for row in ideas if isinstance(row, dict)}
    archives_by_code = load_archives()

    repair_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for row in rows:
        if row.get("math_validation_status") != "provider_error":
            continue
        idea = ideas_by_id.get(row.get("idea_id") or "") or {}
        code = code_from_raw(row.get("raw_symbol"), row.get("eodhd_symbol"))
        if not code:
            reason_counts["missing_code"] += 1
            continue
        candidates = archives_by_code.get(code) or []
        if not candidates:
            reason_counts["no_delisted_archive_code_match"] += 1
            continue
        company_name = (
            idea.get("company_name")
            or row.get("fundamentals_name")
            or row.get("delisted_provider_name")
            or ""
        )
        hints = hint_exchanges(row.get("raw_symbol"), idea.get("yahoo_symbol"))
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for archive_exchange, record in candidates:
            instrument_type = str(record.get("Type") or "").strip().lower()
            if instrument_type and instrument_type not in COMMON_TYPES:
                continue
            exchange = str(record.get("Exchange") or archive_exchange).upper()
            score = name_score(company_name, str(record.get("Name") or ""))
            if hints and exchange in hints:
                score += 0.15
            repaired_symbol = eodhd_symbol_from_record(record, archive_exchange)
            if repaired_symbol:
                scored.append((score, repaired_symbol, record))
        if not scored:
            reason_counts["no_common_stock_archive_candidate"] += 1
            continue
        scored.sort(key=lambda item: item[0], reverse=True)
        score, repaired_symbol, record = scored[0]
        if score < MIN_MATCH_SCORE:
            reason_counts["archive_name_match_too_weak"] += 1
            continue
        repaired = dict(idea)
        repaired.update(
            {
                "idea_id": row.get("idea_id"),
                "raw_symbol": row.get("raw_symbol"),
                "original_eodhd_symbol": row.get("eodhd_symbol"),
                "eodhd_symbol": repaired_symbol,
                "archive_match_score": score,
                "archive_exchange": record.get("Exchange"),
                "archive_name": record.get("Name"),
                "archive_type": record.get("Type"),
                "archive_isin": record.get("Isin"),
            }
        )
        repair_rows.append(repaired)
        reason_counts["repair_candidate"] += 1

    OUT_JSON.write_text(json.dumps(repair_rows, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "input_rows": len(rows),
        "repair_rows": len(repair_rows),
        "unique_repair_symbols": len({row.get("eodhd_symbol") for row in repair_rows}),
        "reason_counts": dict(reason_counts),
        "top_repair_exchanges": dict(Counter(str(row.get("eodhd_symbol") or "").rsplit(".", 1)[-1] for row in repair_rows).most_common(25)),
        "output": str(OUT_JSON.resolve()),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
