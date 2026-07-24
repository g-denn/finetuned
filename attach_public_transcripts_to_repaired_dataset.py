from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context_repaired_clean_hf_stage"
TRANSCRIPTS_JSONL = (
    ROOT
    / "eodhd_output"
    / "public_transcript_sources"
    / "Rogersurf_earnings-call-transcripts"
    / "exports"
    / "vic_matched_transcripts.json"
)
OUT_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context_repaired_clean_transcripts_hf_stage"


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


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


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


def matching_transcripts(index: dict[str, list[dict[str, Any]]], row: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
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


def enrich_analysis_row(index: dict[str, list[dict[str, Any]]], row: dict[str, Any]) -> dict[str, Any]:
    matched, ticker_transcript_count = matching_transcripts(index, row)
    enriched = dict(row)
    enriched["earnings_transcripts_publication_to_3y"] = matched
    enriched["earnings_transcript_context_counts"] = {
        "provider": "Rogersurf/earnings-call-transcripts",
        "ticker_transcript_count_all_dates": ticker_transcript_count,
        "publication_to_3y_transcript_count": len(matched),
        "publication_to_3y_transcript_chars": sum(item["transcript_chars"] for item in matched),
        "available_call_dates": [item["call_date"] for item in matched],
    }
    enriched["transcript_leakage_note"] = (
        "These transcripts are from the publication date through three years after publication. "
        "They are post-publication context and should not be used as predictive input unless the task "
        "explicitly allows post-pitch evidence."
    )
    return enriched


def enrich_sft_row(counts_by_idea: dict[str, dict[str, Any]], row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    enriched["earnings_transcript_context_counts"] = counts_by_idea.get(str(row.get("idea_id")), {})
    return enriched


def main() -> None:
    index = load_transcript_index()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "analysis").mkdir(exist_ok=True)
    (OUT_DIR / "sft").mkdir(exist_ok=True)

    source_summary = json.loads((SOURCE_DIR / "dataset_summary.json").read_text(encoding="utf-8"))
    counts_by_idea: dict[str, dict[str, Any]] = {}
    coverage_rows = []
    totals = {
        "rows": 0,
        "rows_with_any_publication_to_3y_transcript": 0,
        "total_publication_to_3y_transcripts": 0,
        "total_publication_to_3y_transcript_chars": 0,
        "matched_transcript_source_rows": sum(len(rows) for rows in index.values()),
        "matched_transcript_source_tickers": len(index),
    }

    for split in ("train", "validation", "test"):
        analysis_rows = []
        for row in iter_jsonl(SOURCE_DIR / "analysis" / f"{split}.jsonl"):
            enriched = enrich_analysis_row(index, row)
            counts = enriched["earnings_transcript_context_counts"]
            counts_by_idea[str(row.get("idea_id"))] = counts
            totals["rows"] += 1
            totals["total_publication_to_3y_transcripts"] += counts["publication_to_3y_transcript_count"]
            totals["total_publication_to_3y_transcript_chars"] += counts["publication_to_3y_transcript_chars"]
            if counts["publication_to_3y_transcript_count"] > 0:
                totals["rows_with_any_publication_to_3y_transcript"] += 1
            coverage_rows.append(
                {
                    "idea_id": row.get("idea_id"),
                    "split": split,
                    "eodhd_symbol": row.get("eodhd_symbol"),
                    "company_name": row.get("company_name"),
                    "publication_date": row.get("publication_date"),
                    **counts,
                }
            )
            analysis_rows.append(enriched)
        write_jsonl(OUT_DIR / "analysis" / f"{split}.jsonl", analysis_rows)

        sft_rows = [
            enrich_sft_row(counts_by_idea, row)
            for row in iter_jsonl(SOURCE_DIR / "sft" / f"{split}.jsonl")
        ]
        write_jsonl(OUT_DIR / "sft" / f"{split}.jsonl", sft_rows)

    summary = dict(source_summary)
    summary["transcripts"] = totals
    summary["transcripts"]["source_dataset"] = "Rogersurf/earnings-call-transcripts"
    summary["transcripts"]["source_local_jsonl"] = str(TRANSCRIPTS_JSONL)
    summary["transcripts"]["license_name"] = "research-and-educational-use-only"
    (OUT_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    with (OUT_DIR / "transcript_coverage.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in coverage_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    readme = (SOURCE_DIR / "README.md").read_text(encoding="utf-8")
    readme += (
        "\n\n## Transcript Enrichment\n\n"
        "This staged dataset attaches public earnings-call transcripts from "
        "`Rogersurf/earnings-call-transcripts` when the transcript ticker matches "
        "the EODHD US ticker and the call date falls from the VIC publication date "
        "through three years after publication. Transcript text is attached only "
        "to the `analysis` files; `sft` files include transcript coverage metadata "
        "but not transcript body text.\n\n"
        "License note: the transcript source dataset is marked "
        "`research-and-educational-use-only`; keep this private and review source "
        "terms before redistribution beyond research use.\n\n"
        "```json\n"
        f"{json.dumps(summary['transcripts'], indent=2, sort_keys=True)}\n"
        "```\n"
    )
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
