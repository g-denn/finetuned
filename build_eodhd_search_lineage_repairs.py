#!/usr/bin/env python3
"""Use EODHD Search to propose high-confidence current-ticker lineage repairs."""

from __future__ import annotations

import csv
import difflib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from salvage_partial_horizon_labels import looks_non_common_instrument, split_flags


BASE_DIR = Path("eodhd_output/full_run")
VALIDATION_CSV = BASE_DIR / "validation_results_with_reverse_split_salvage.csv"
TRAINING_READY_CSV = BASE_DIR / "training_ready_after_reverse_split_salvage.csv"
IDEAS_JSON = Path("eodhd_output/all_ideas.json")
SEARCH_CACHE = BASE_DIR / "eodhd_search_cache"
OUT_JSON = Path("eodhd_output/eodhd_search_lineage_repair_ideas.json")
SUMMARY_JSON = BASE_DIR / "eodhd_search_lineage_repair_summary.json"

API_ROOT = "https://eodhd.com/api"
MIN_NAME_SCORE = 0.9
COMMON_TYPE = "common stock"
BAD_CODE_SUFFIXES = ("-WS", "-WT", "-W", "-PF", "-P", "-PR", "-U", "-UN", "/WS", "/WT", "/W", "/U")


def require_token() -> str:
    token = os.environ.get("EODHD_API_TOKEN")
    if not token:
        raise SystemExit("Set EODHD_API_TOKEN before running this script.")
    return token


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("idea_id") or "", row.get("eodhd_symbol") or "", row.get("publication_date") or "")


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


def safe_cache_token(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value)[:120]


def search_query_variants(company_name: str) -> list[str]:
    clean = re.sub(r"\s+", " ", company_name or "").strip()
    variants = [clean]
    without_adr = re.sub(r"\bADR\b", "", clean, flags=re.I).strip()
    if without_adr and without_adr != clean:
        variants.append(without_adr)
    without_suffix = re.sub(r"\b(Inc\.?|Corporation|Corp\.?|Ltd\.?|Limited|PLC|S\.A\.|AG|NV|SE)\b", "", without_adr, flags=re.I)
    without_suffix = re.sub(r"\s+", " ", without_suffix).strip()
    if without_suffix and without_suffix not in variants:
        variants.append(without_suffix)
    return [variant for variant in variants if variant]


def request_search(query: str, token: str) -> list[dict[str, Any]]:
    SEARCH_CACHE.mkdir(parents=True, exist_ok=True)
    path = SEARCH_CACHE / f"{safe_cache_token(query)}.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = []
        return payload if isinstance(payload, list) else []
    params = urllib.parse.urlencode({"fmt": "json", "api_token": token})
    url = f"{API_ROOT}/search/{urllib.parse.quote(query)}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "financial-validation-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, list):
        payload = []
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    time.sleep(0.05)
    return payload


def exchange_hints(row: dict[str, str], idea: dict[str, Any]) -> set[str]:
    combo = f"{row.get('raw_symbol') or ''} {row.get('eodhd_symbol') or ''} {idea.get('yahoo_symbol') or ''}".upper()
    hints: set[str] = set()
    mapping = {
        ".US": {"US"},
        " US": {"US"},
        ".L": {"LSE"},
        " LN": {"LSE"},
        ".TO": {"TO"},
        " CN": {"TO", "V"},
        ".V": {"V"},
        ".HK": {"HK"},
        " HK": {"HK"},
        ".T": {"TSE", "T"},
        " JP": {"TSE", "T"},
        ".KS": {"KO", "KQ"},
        " KS": {"KO", "KQ"},
        ".DE": {"XETRA", "F"},
        " GR": {"XETRA", "F"},
        " GY": {"XETRA", "F"},
    }
    for marker, exchanges in mapping.items():
        if marker in combo:
            hints.update(exchanges)
    return hints


def candidate_symbol(record: dict[str, Any]) -> str | None:
    code = str(record.get("Code") or "").upper().strip()
    exchange = str(record.get("Exchange") or "").upper().strip()
    if not code or not exchange:
        return None
    return f"{code}.{exchange}"


def bad_candidate_symbol(symbol: str) -> bool:
    code = symbol.upper().rsplit(".", 1)[0]
    return any(code.endswith(suffix) or suffix in code for suffix in BAD_CODE_SUFFIXES)


def pick_candidate(row: dict[str, str], idea: dict[str, Any], token: str) -> tuple[dict[str, Any] | None, str]:
    idea_name = str(idea.get("company_name") or "").strip()
    row_name = str(row.get("fundamentals_name") or "").strip()
    company_name = idea_name or row_name
    hints = exchange_hints(row, idea)
    best: tuple[float, dict[str, Any]] | None = None
    for query in search_query_variants(company_name):
        try:
            records = request_search(query, token)
        except Exception as exc:  # noqa: BLE001 - per-query failure should not stop the batch.
            return None, f"search_failed:{type(exc).__name__}"
        for record in records:
            if str(record.get("Type") or "").strip().lower() != COMMON_TYPE:
                continue
            symbol = candidate_symbol(record)
            if not symbol or symbol == row.get("eodhd_symbol"):
                continue
            if bad_candidate_symbol(symbol):
                continue
            candidate_name = str(record.get("Name") or "")
            idea_score = name_score(idea_name, candidate_name)
            row_score = name_score(row_name, candidate_name)
            row_to_idea_score = name_score(idea_name, row_name)
            if idea_name:
                if idea_score >= 0.82:
                    score = idea_score
                elif row_score >= 0.9 and row_to_idea_score >= 0.75:
                    score = min(row_score, row_to_idea_score)
                else:
                    continue
            else:
                score = row_score
            exchange = str(record.get("Exchange") or "").upper()
            if hints and exchange in hints:
                score += 0.08
            if not hints and exchange != "US" and str(row.get("eodhd_symbol") or "").endswith(".US"):
                score -= 0.15
            if best is None or score > best[0]:
                best = (score, record)
    if best is None:
        return None, "no_search_candidate"
    score, record = best
    if score < MIN_NAME_SCORE:
        return None, "search_name_match_too_weak"
    result = dict(record)
    result["match_score"] = score
    return result, "repair_candidate"


def select_rows(rows: list[dict[str, str]], training_keys: set[tuple[str, str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        if row_key(row) in training_keys:
            continue
        if row.get("fundamentals_status") != "fetched":
            continue
        if row.get("fundamentals_type") != "Common Stock":
            continue
        if looks_non_common_instrument(row):
            continue
        flags = split_flags(row)
        if row.get("math_validation_status") == "provider_error":
            selected.append(row)
        elif row.get("math_validation_status") == "math_incomplete" and (
            "first_price_far_after_publication" in flags or "invalid_start_price_for_publication_date" in flags
        ):
            selected.append(row)
    return selected


def main() -> int:
    token = require_token()
    rows = load_csv(VALIDATION_CSV)
    training_rows = load_csv(TRAINING_READY_CSV)
    training_keys = {row_key(row) for row in training_rows}
    ideas = json.loads(IDEAS_JSON.read_text(encoding="utf-8"))
    ideas_by_id = {str(row.get("idea_id") or ""): row for row in ideas if isinstance(row, dict)}

    selected = select_rows(rows, training_keys)
    repairs: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for row in selected:
        idea = ideas_by_id.get(row.get("idea_id") or "") or {}
        candidate, reason = pick_candidate(row, idea, token)
        reasons[reason] += 1
        if not candidate:
            continue
        repaired = dict(idea)
        repaired.update(
            {
                "idea_id": row.get("idea_id"),
                "raw_symbol": row.get("raw_symbol"),
                "original_eodhd_symbol": row.get("eodhd_symbol"),
                "eodhd_symbol": candidate_symbol(candidate),
                "search_match_score": candidate.get("match_score"),
                "search_name": candidate.get("Name"),
                "search_type": candidate.get("Type"),
                "search_exchange": candidate.get("Exchange"),
            }
        )
        repairs.append(repaired)

    OUT_JSON.write_text(json.dumps(repairs, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "repair_rows": len(repairs),
        "unique_repair_symbols": len({row.get("eodhd_symbol") for row in repairs}),
        "reason_counts": dict(reasons),
        "top_repair_exchanges": dict(Counter(str(row.get("eodhd_symbol") or "").rsplit(".", 1)[-1] for row in repairs).most_common(25)),
        "output": str(OUT_JSON.resolve()),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
