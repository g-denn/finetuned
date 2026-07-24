from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context_hf_stage" / "data"
OUT_DIR = ROOT / "eodhd_output" / "row_level_readiness_audit"
OUT_CSV = OUT_DIR / "vic_row_level_readiness.csv"
OUT_JSON = OUT_DIR / "vic_row_level_readiness_summary.json"

MIN_PITCH_CHARS = 500


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def iter_rows():
    paths = sorted(SOURCE_DIR.glob("vic_pitch_financial_context-*.jsonl"))
    if not paths:
        raise SystemExit(f"No staged JSONL shards found in {SOURCE_DIR}")
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    yield path.name, line_number, json.loads(line)


def row_audit(source_file: str, source_line: int, row: dict[str, Any]) -> dict[str, Any]:
    counts = row.get("financial_context_counts") or {}
    performance = row.get("performance") or {}
    text = str(row.get("full_stock_pitch_text") or "")

    has_pitch_text = len(text.strip()) >= MIN_PITCH_CHARS
    has_publication_date = bool(str(row.get("publication_date") or "").strip())
    has_any_return = any(
        parse_float(performance.get(key)) is not None
        for key in (
            "raw_perf_1y",
            "raw_perf_3y",
            "raw_perf_5y",
            "raw_perf_10y",
            "raw_perf_20y",
        )
    )
    has_3y_return = parse_float(row.get("raw_perf_3y")) is not None
    has_3y_outcome = bool(str(row.get("outcome_3y") or "").strip())

    latest_annual = int(counts.get("latest_annual_asof_pitch_records") or 0)
    latest_quarterly = int(counts.get("latest_quarterly_asof_pitch_records") or 0)
    trailing = int(counts.get("trailing_5y_asof_pitch_records") or 0)
    forward_3y = int(counts.get("forward_3y_after_pitch_records") or 0)
    forward_5y = int(counts.get("forward_5y_after_pitch_records") or 0)
    all_statement_records = int(counts.get("all_statement_records_available") or 0)

    has_point_in_time = bool(counts.get("has_point_in_time_financials"))
    has_complete_latest_annual = latest_annual >= 3
    has_complete_latest_quarterly = latest_quarterly >= 3
    has_trailing_history = trailing > 0
    has_forward_3y = forward_3y > 0
    has_forward_5y = forward_5y > 0
    has_fundamentals_file = bool(row.get("raw_fundamentals_path")) and all_statement_records > 0
    has_model_ready_financials = (
        has_point_in_time
        and has_complete_latest_annual
        and has_complete_latest_quarterly
        and has_trailing_history
    )
    fine_tune_ready_3y = (
        has_pitch_text
        and has_publication_date
        and has_3y_return
        and has_3y_outcome
        and has_model_ready_financials
    )

    missing = []
    checks = {
        "pitch_text": has_pitch_text,
        "publication_date": has_publication_date,
        "any_return": has_any_return,
        "3y_return": has_3y_return,
        "3y_outcome": has_3y_outcome,
        "fundamentals_file_with_statement_records": has_fundamentals_file,
        "point_in_time_financials": has_point_in_time,
        "complete_latest_annual_statements": has_complete_latest_annual,
        "complete_latest_quarterly_statements": has_complete_latest_quarterly,
        "trailing_financial_history": has_trailing_history,
        "forward_3y_financials": has_forward_3y,
    }
    for key, ok in checks.items():
        if not ok:
            missing.append(key)

    return {
        "source_file": source_file,
        "source_line": source_line,
        "idea_id": row.get("idea_id"),
        "raw_symbol": row.get("raw_symbol"),
        "eodhd_symbol": row.get("eodhd_symbol"),
        "company_name": row.get("company_name"),
        "publication_date": row.get("publication_date"),
        "pitch_text_chars": len(text),
        "has_pitch_text": has_pitch_text,
        "has_publication_date": has_publication_date,
        "has_any_return": has_any_return,
        "has_3y_return": has_3y_return,
        "has_3y_outcome": has_3y_outcome,
        "raw_perf_3y": row.get("raw_perf_3y"),
        "outcome_3y": row.get("outcome_3y"),
        "has_fundamentals_file_with_statement_records": has_fundamentals_file,
        "all_statement_records_available": all_statement_records,
        "has_point_in_time_financials": has_point_in_time,
        "latest_annual_asof_pitch_records": latest_annual,
        "latest_quarterly_asof_pitch_records": latest_quarterly,
        "trailing_5y_asof_pitch_records": trailing,
        "forward_3y_after_pitch_records": forward_3y,
        "forward_5y_after_pitch_records": forward_5y,
        "has_model_ready_financials": has_model_ready_financials,
        "fine_tune_ready_3y": fine_tune_ready_3y,
        "missing_requirements": ";".join(missing),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [row_audit(source, line, row) for source, line, row in iter_rows()]
    fieldnames = list(rows[0].keys()) if rows else []
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    bool_fields = [
        "has_pitch_text",
        "has_publication_date",
        "has_any_return",
        "has_3y_return",
        "has_3y_outcome",
        "has_fundamentals_file_with_statement_records",
        "has_point_in_time_financials",
        "has_model_ready_financials",
        "fine_tune_ready_3y",
    ]
    missing_counter: Counter[str] = Counter()
    for row in rows:
        for item in str(row["missing_requirements"]).split(";"):
            if item:
                missing_counter[item] += 1

    summary = {
        "rows": len(rows),
        "audit_csv": str(OUT_CSV),
        "counts": {field: sum(1 for row in rows if row[field]) for field in bool_fields},
        "missing_requirement_counts": dict(sorted(missing_counter.items())),
        "financial_statement_record_counts": {
            "rows_with_any_statement_records": sum(1 for row in rows if row["all_statement_records_available"] > 0),
            "rows_with_complete_latest_annual": sum(
                1 for row in rows if row["latest_annual_asof_pitch_records"] >= 3
            ),
            "rows_with_complete_latest_quarterly": sum(
                1 for row in rows if row["latest_quarterly_asof_pitch_records"] >= 3
            ),
            "rows_with_trailing_5y_asof_pitch": sum(1 for row in rows if row["trailing_5y_asof_pitch_records"] > 0),
            "rows_with_forward_3y": sum(1 for row in rows if row["forward_3y_after_pitch_records"] > 0),
            "rows_with_forward_5y": sum(1 for row in rows if row["forward_5y_after_pitch_records"] > 0),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
