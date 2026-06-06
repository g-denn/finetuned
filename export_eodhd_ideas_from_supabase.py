#!/usr/bin/env python3
"""Export performance rows from linked Supabase into EODHD backfill input JSON."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


YAHOO_TO_EODHD_EXCHANGE = {
    "AX": "AU",
    "KS": "KO",
    "KQ": "KQ",
    "L": "LSE",
    "TO": "TO",
    "V": "V",
    "PA": "PA",
    "AS": "AS",
    "BR": "BR",
    "SW": "SW",
    "MI": "MI",
    "MC": "MC",
    "DE": "XETRA",
    "F": "F",
    "HK": "HK",
    "SI": "SG",
    "SS": "SHG",
    "SZ": "SHE",
    "T": "TSE",
    "TW": "TW",
    "TWO": "TWO",
}


def resolve_eodhd_symbol(raw_symbol: str | None, yahoo_symbol: str | None) -> tuple[str | None, str | None]:
    candidate = clean_symbol(yahoo_symbol) or clean_symbol(raw_symbol)
    if not candidate:
        return None, "missing_symbol"

    if " " in candidate and "." not in candidate:
        parts = candidate.split()
        if len(parts) == 2 and parts[1].upper() in YAHOO_TO_EODHD_EXCHANGE:
            return f"{parts[0].upper()}.{YAHOO_TO_EODHD_EXCHANGE[parts[1].upper()]}", None
        return None, "unsupported_symbol_format"

    if any(char in candidate for char in [" ", "/", "\\", "\t", "\r", "\n"]):
        return None, "unsupported_symbol_format"

    if "." in candidate:
        code, suffix = candidate.rsplit(".", 1)
        suffix = suffix.upper()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,14}", code.upper()):
            return None, "unsupported_symbol_format"
        if not re.fullmatch(r"[A-Z0-9]{1,8}", suffix):
            return None, "unsupported_symbol_format"
        exchange = YAHOO_TO_EODHD_EXCHANGE.get(suffix, suffix)
        return f"{code.upper()}.{exchange}", None

    if re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,14}", candidate.upper()):
        return f"{candidate.upper()}.US", None

    return None, "unsupported_symbol_format"


def clean_symbol(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eodhd_output/all_ideas.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    sql = """
        select
            py.idea_id,
            py.raw_symbol,
            py.yahoo_symbol,
            py.publication_date::text as publication_date,
            coalesce(c.company_name, py.raw_symbol, py.yahoo_symbol) as company_name
        from public.performance_yahoo py
        left join public.companies c on c.ticker = py.raw_symbol
        where py.publication_date is not null
        order by py.raw_symbol nulls last, py.publication_date;
    """
    proc = subprocess.run(
        ["supabase", "db", "query", "--linked", sql, "--output", "json"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    rows = payload.get("rows", [])
    ideas: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        symbol, reason = resolve_eodhd_symbol(row.get("raw_symbol"), row.get("yahoo_symbol"))
        item = {
            "idea_id": str(row["idea_id"]),
            "raw_symbol": row.get("raw_symbol") or row.get("yahoo_symbol"),
            "yahoo_symbol": row.get("yahoo_symbol"),
            "eodhd_symbol": symbol,
            "company_name": row.get("company_name"),
            "publication_date": row["publication_date"],
        }
        if symbol:
            ideas.append(item)
        else:
            item["skip_reason"] = reason
            skipped.append(item)

    out.write_text(json.dumps(ideas, indent=2, sort_keys=True), encoding="utf-8")
    skipped_path = out.with_name(out.stem + "_skipped.json")
    skipped_path.write_text(json.dumps(skipped, indent=2, sort_keys=True), encoding="utf-8")
    unique_symbols = {idea["eodhd_symbol"] for idea in ideas}
    print(f"exported_ideas={len(ideas)}")
    print(f"unique_eodhd_symbols={len(unique_symbols)}")
    print(f"skipped={len(skipped)}")
    print(f"output={out.resolve()}")
    print(f"skipped_output={skipped_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
