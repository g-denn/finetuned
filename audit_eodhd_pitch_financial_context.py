from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
CANONICAL_CSV = ROOT / "data" / "processed" / "investment_canonical.csv"
COMBINED_TABLE_CSV = ROOT / "eodhd_output" / "dataset_financial_pull" / "eodhd_combined_stock_table.csv"
RAW_DIR = ROOT / "eodhd_output" / "dataset_financial_pull" / "raw"
OUT_DIR = ROOT / "eodhd_output" / "pitch_financial_context_audit"
OUT_CSV = OUT_DIR / "pitch_financial_context_coverage.csv"
OUT_JSON = OUT_DIR / "pitch_financial_context_coverage_summary.json"

FILING_DATE_RE = re.compile(r'"filing_date"\s*:\s*"(\d{4}-\d{2}-\d{2})"')
FISCAL_DATE_RE = re.compile(r'"date"\s*:\s*"(\d{4}-\d{2}-\d{2})"')


def parse_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NONE", "NULL"}:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def scan_fundamentals_dates(symbol: str) -> tuple[str, dict[str, Any]]:
    path = RAW_DIR / symbol / "fundamentals.json"
    if not path.exists():
        return symbol, {"has_fundamentals_file": False, "filing_dates": [], "fiscal_dates": []}

    text = path.read_text(encoding="utf-8", errors="ignore")
    filing_dates = sorted({parsed for parsed in (parse_date(match) for match in FILING_DATE_RE.findall(text)) if parsed})
    fiscal_dates = sorted({parsed for parsed in (parse_date(match) for match in FISCAL_DATE_RE.findall(text)) if parsed})
    return symbol, {
        "has_fundamentals_file": True,
        "fundamentals_bytes": path.stat().st_size,
        "filing_dates": filing_dates,
        "fiscal_dates": fiscal_dates,
    }


def summarize_dates(values: list[date]) -> tuple[str | None, str | None]:
    if not values:
        return None, None
    return values[0].isoformat(), values[-1].isoformat()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ideas = pd.read_csv(
        CANONICAL_CSV,
        dtype=str,
        keep_default_na=False,
        usecols=["idea_id", "raw_symbol", "eodhd_symbol", "company_name", "publication_date"],
    )
    combined = pd.read_csv(
        COMBINED_TABLE_CSV,
        dtype=str,
        keep_default_na=False,
        usecols=["symbol", "eod_first_date", "eod_latest_date"],
    )
    symbol_meta = combined.set_index("symbol").to_dict(orient="index")
    symbols = sorted({symbol for symbol in ideas["eodhd_symbol"].astype(str) if symbol})

    scanned: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(scan_fundamentals_dates, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol, payload = future.result()
            scanned[symbol] = payload

    rows: list[dict[str, Any]] = []
    for idea in ideas.to_dict(orient="records"):
        symbol = str(idea.get("eodhd_symbol") or "").strip()
        pub_date = parse_date(idea.get("publication_date"))
        meta = symbol_meta.get(symbol, {})
        scan = scanned.get(symbol, {"has_fundamentals_file": False, "filing_dates": [], "fiscal_dates": []})

        eod_first = parse_date(meta.get("eod_first_date"))
        eod_latest = parse_date(meta.get("eod_latest_date"))
        filing_dates: list[date] = scan["filing_dates"]
        fiscal_dates: list[date] = scan["fiscal_dates"]
        filing_min, filing_max = summarize_dates(filing_dates)
        fiscal_min, fiscal_max = summarize_dates(fiscal_dates)

        before_filing: list[date] = []
        next_3y_filing: list[date] = []
        latest_before = None
        earliest_next = None
        if pub_date:
            end_3y = pub_date + timedelta(days=365 * 3)
            before_filing = [value for value in filing_dates if value <= pub_date]
            next_3y_filing = [value for value in filing_dates if pub_date < value <= end_3y]
            latest_before = before_filing[-1] if before_filing else None
            earliest_next = next_3y_filing[0] if next_3y_filing else None

        rows.append(
            {
                "idea_id": idea.get("idea_id"),
                "raw_symbol": idea.get("raw_symbol"),
                "eodhd_symbol": symbol,
                "company_name": idea.get("company_name"),
                "publication_date": pub_date.isoformat() if pub_date else None,
                "has_eodhd_symbol": bool(symbol),
                "has_publication_date": pub_date is not None,
                "has_fundamentals_file": bool(scan["has_fundamentals_file"]),
                "fundamentals_bytes": scan.get("fundamentals_bytes"),
                "unique_filing_dates": len(filing_dates),
                "filing_date_min": filing_min,
                "filing_date_max": filing_max,
                "unique_fiscal_dates": len(fiscal_dates),
                "fiscal_date_min": fiscal_min,
                "fiscal_date_max": fiscal_max,
                "unique_filing_dates_on_or_before_pub": len(before_filing),
                "has_point_in_time_filing_on_or_before_pub": bool(before_filing),
                "latest_filing_on_or_before_pub": latest_before.isoformat() if latest_before else None,
                "days_between_pub_and_latest_filing": (pub_date - latest_before).days
                if pub_date and latest_before
                else None,
                "unique_filing_dates_pub_to_3y": len(next_3y_filing),
                "has_filing_pub_to_3y": bool(next_3y_filing),
                "earliest_filing_after_pub_within_3y": earliest_next.isoformat() if earliest_next else None,
                "eod_first_date": eod_first.isoformat() if eod_first else None,
                "eod_latest_date": eod_latest.isoformat() if eod_latest else None,
                "eod_covers_publication_date": bool(
                    pub_date and eod_first and eod_latest and eod_first <= pub_date <= eod_latest
                ),
                "eod_covers_publication_to_3y": bool(
                    pub_date
                    and eod_first
                    and eod_latest
                    and eod_first <= pub_date
                    and eod_latest >= pub_date + timedelta(days=365 * 3)
                ),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False)

    valid = frame[(frame["has_eodhd_symbol"] == True) & (frame["has_publication_date"] == True)]  # noqa: E712
    with_fundamentals = valid[valid["has_fundamentals_file"] == True]  # noqa: E712

    def count_true(column: str, source: pd.DataFrame = valid) -> int:
        return int((source[column] == True).sum())  # noqa: E712

    def pct(value: int, total: int) -> float | None:
        return round(value / total, 6) if total else None

    point_in_time = count_true("has_point_in_time_filing_on_or_before_pub", with_fundamentals)
    future_3y = count_true("has_filing_pub_to_3y", with_fundamentals)
    eod_at_pub = count_true("eod_covers_publication_date")
    eod_to_3y = count_true("eod_covers_publication_to_3y")

    summary = {
        "canonical_rows": int(len(frame)),
        "valid_pitch_symbol_rows": int(len(valid)),
        "unique_symbols_scanned": len(symbols),
        "rows_with_fundamentals_file": count_true("has_fundamentals_file"),
        "rows_with_point_in_time_filing_on_or_before_publication": point_in_time,
        "rows_with_filing_publication_to_3y": future_3y,
        "rows_with_eod_covering_publication_date": eod_at_pub,
        "rows_with_eod_covering_publication_to_3y": eod_to_3y,
        "pct_valid_with_fundamentals": pct(count_true("has_fundamentals_file"), len(valid)),
        "pct_fundamentals_with_point_in_time_filing": pct(point_in_time, len(with_fundamentals)),
        "pct_fundamentals_with_filing_publication_to_3y": pct(future_3y, len(with_fundamentals)),
        "pct_valid_with_eod_covering_publication_date": pct(eod_at_pub, len(valid)),
        "pct_valid_with_eod_covering_publication_to_3y": pct(eod_to_3y, len(valid)),
        "audit_csv": str(OUT_CSV),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
