from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RAW_SHARD_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context_hf_stage" / "data"
TRANSCRIPTS_JSONL = (
    ROOT
    / "eodhd_output"
    / "public_transcript_sources"
    / "Rogersurf_earnings-call-transcripts"
    / "exports"
    / "vic_matched_transcripts.json"
)
OUT_DIR = ROOT / "eodhd_output" / "raw_vic_public_transcript_companion"
OUT_JSONL = OUT_DIR / "public_transcripts_publication_to_3y.jsonl"
OUT_COVERAGE = OUT_DIR / "public_transcripts_coverage.jsonl"
OUT_SUMMARY = OUT_DIR / "public_transcripts_summary.json"


def parse_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def base_ticker(eodhd_symbol: str | None) -> str | None:
    symbol = str(eodhd_symbol or "").strip()
    if not symbol.endswith(".US"):
        return None
    base = symbol[:-3].upper()
    if not base or "/" in base:
        return None
    return base.replace(".", "-")


def iter_raw_rows():
    paths = sorted(RAW_SHARD_DIR.glob("vic_pitch_financial_context-*.jsonl"))
    if not paths:
        raise SystemExit(f"No raw staged shards found in {RAW_SHARD_DIR}")
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def load_transcript_index() -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    with TRANSCRIPTS_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ticker = str(row.get("ticker") or "").upper().strip()
            call_date = parse_date(row.get("call_date"))
            transcript = str(row.get("transcript") or "").strip()
            if not ticker or call_date is None or not transcript:
                continue
            item = {
                "provider": "Rogersurf/earnings-call-transcripts",
                "ticker": ticker,
                "company": row.get("company"),
                "quarter": row.get("quarter"),
                "earnings_year": row.get("earnings_year"),
                "call_date": call_date.isoformat(),
                "title": row.get("title"),
                "source_url": row.get("source_url"),
                "scraped_at": row.get("scraped_at"),
                "transcript_chars": len(transcript),
                "transcript": transcript,
            }
            index.setdefault(ticker, []).append(item)
    for rows in index.values():
        rows.sort(key=lambda item: item["call_date"])
    return index


def match_transcripts(index: dict[str, list[dict[str, Any]]], row: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    ticker = base_ticker(row.get("eodhd_symbol"))
    pub_date = parse_date(row.get("publication_date"))
    if not ticker or pub_date is None:
        return [], 0
    end_date = pub_date + timedelta(days=365 * 3)
    candidates = index.get(ticker, [])
    matched = [
        item
        for item in candidates
        if pub_date <= parse_date(item.get("call_date")) <= end_date  # type: ignore[arg-type]
    ]
    return matched, len(candidates)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = load_transcript_index()
    summary = {
        "raw_rows": 0,
        "rows_with_any_publication_to_3y_transcript": 0,
        "total_publication_to_3y_transcripts": 0,
        "total_publication_to_3y_transcript_chars": 0,
        "matched_transcript_source_rows": sum(len(rows) for rows in index.values()),
        "matched_transcript_source_tickers": len(index),
        "source_dataset": "Rogersurf/earnings-call-transcripts",
        "license_name": "research-and-educational-use-only",
    }

    with OUT_JSONL.open("w", encoding="utf-8", newline="\n") as out, OUT_COVERAGE.open(
        "w", encoding="utf-8", newline="\n"
    ) as coverage:
        for row in iter_raw_rows():
            summary["raw_rows"] += 1
            matched, ticker_total = match_transcripts(index, row)
            transcript_chars = sum(item["transcript_chars"] for item in matched)
            if matched:
                summary["rows_with_any_publication_to_3y_transcript"] += 1
                summary["total_publication_to_3y_transcripts"] += len(matched)
                summary["total_publication_to_3y_transcript_chars"] += transcript_chars
                out.write(
                    json.dumps(
                        {
                            "idea_id": row.get("idea_id"),
                            "eodhd_symbol": row.get("eodhd_symbol"),
                            "company_name": row.get("company_name"),
                            "publication_date": row.get("publication_date"),
                            "transcripts": matched,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            coverage.write(
                json.dumps(
                    {
                        "idea_id": row.get("idea_id"),
                        "eodhd_symbol": row.get("eodhd_symbol"),
                        "company_name": row.get("company_name"),
                        "publication_date": row.get("publication_date"),
                        "provider": "Rogersurf/earnings-call-transcripts",
                        "ticker_transcript_count_all_dates": ticker_total,
                        "publication_to_3y_transcript_count": len(matched),
                        "publication_to_3y_transcript_chars": transcript_chars,
                        "available_call_dates": [item["call_date"] for item in matched],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    summary.update(
        {
            "transcript_jsonl": str(OUT_JSONL),
            "coverage_jsonl": str(OUT_COVERAGE),
            "note": (
                "Transcript bodies are stored only for rows with at least one matching public transcript. "
                "Coverage JSONL contains one row per raw VIC row."
            ),
        }
    )
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
