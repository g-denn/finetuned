from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context_repaired_clean_transcripts_hf_stage"
OUT_JSONL = DATASET_DIR / "row_level_coverage_manifest.jsonl"
OUT_CSV = DATASET_DIR / "row_level_coverage_manifest.csv"
OUT_SUMMARY = DATASET_DIR / "row_level_coverage_summary.json"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def make_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    counts = row.get("financial_context_counts") or {}
    transcript_counts = row.get("earnings_transcript_context_counts") or {}
    label = row.get("label") or {}
    text = row.get("full_stock_pitch_text") or ""
    return {
        "idea_id": row.get("idea_id"),
        "split": row.get("split"),
        "raw_symbol": row.get("raw_symbol"),
        "eodhd_symbol": row.get("eodhd_symbol"),
        "company_name": row.get("company_name"),
        "publication_date": row.get("publication_date"),
        "has_vic_pitch_text": bool(text),
        "pitch_text_chars": len(text),
        "has_publication_date": bool(row.get("publication_date")),
        "has_3y_return": label.get("raw_perf_3y") is not None,
        "raw_perf_3y": label.get("raw_perf_3y"),
        "outcome_3y": label.get("outcome_3y"),
        "has_eodhd_financial_statement_context": True,
        "all_statement_records_available": counts.get("all_statement_records_available"),
        "trailing_5y_asof_pitch_records": counts.get("trailing_5y_asof_pitch_records"),
        "latest_annual_asof_pitch_records": counts.get("latest_annual_asof_pitch_records"),
        "latest_quarterly_asof_pitch_records": counts.get("latest_quarterly_asof_pitch_records"),
        "uses_estimated_availability_dates": counts.get("uses_estimated_availability_dates"),
        "estimated_trailing_5y_records": counts.get("estimated_trailing_5y_records"),
        "estimated_latest_annual_records": counts.get("estimated_latest_annual_records"),
        "estimated_latest_quarterly_records": counts.get("estimated_latest_quarterly_records"),
        "has_model_ready_financials": (
            (counts.get("trailing_5y_asof_pitch_records") or 0) >= 1
            and (counts.get("latest_annual_asof_pitch_records") or 0) >= 3
            and (counts.get("latest_quarterly_asof_pitch_records") or 0) >= 3
        ),
        "raw_fundamentals_path": row.get("raw_fundamentals_path"),
        "transcript_provider": transcript_counts.get("provider"),
        "ticker_transcript_count_all_dates": transcript_counts.get("ticker_transcript_count_all_dates"),
        "publication_to_3y_transcript_count": transcript_counts.get("publication_to_3y_transcript_count"),
        "publication_to_3y_transcript_chars": transcript_counts.get("publication_to_3y_transcript_chars"),
        "has_publication_to_3y_transcript": (transcript_counts.get("publication_to_3y_transcript_count") or 0) > 0,
        "available_transcript_call_dates": ";".join(transcript_counts.get("available_call_dates") or []),
    }


def main() -> None:
    rows = []
    for split in ("train", "validation", "test"):
        for row in iter_jsonl(DATASET_DIR / "analysis" / f"{split}.jsonl"):
            rows.append(make_manifest_row(row))

    fieldnames = list(rows[0].keys()) if rows else []
    with OUT_JSONL.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "rows": len(rows),
        "manifest_jsonl": str(OUT_JSONL),
        "manifest_csv": str(OUT_CSV),
        "rows_with_vic_pitch_text": sum(1 for row in rows if row["has_vic_pitch_text"]),
        "rows_with_publication_date": sum(1 for row in rows if row["has_publication_date"]),
        "rows_with_3y_return": sum(1 for row in rows if row["has_3y_return"]),
        "rows_with_model_ready_financials": sum(1 for row in rows if row["has_model_ready_financials"]),
        "rows_using_estimated_availability_dates": sum(
            1 for row in rows if row["uses_estimated_availability_dates"]
        ),
        "rows_with_publication_to_3y_transcripts": sum(
            1 for row in rows if row["has_publication_to_3y_transcript"]
        ),
        "total_publication_to_3y_transcripts": sum(
            int(row["publication_to_3y_transcript_count"] or 0) for row in rows
        ),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
