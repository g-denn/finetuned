from __future__ import annotations

import json
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CLEAN_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context_clean_hf_stage"
TRANSCRIPT_RAW_DIR = ROOT / "eodhd_output" / "alpha_vantage_transcripts" / "raw"
OUT_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context_transcript_enriched_hf_stage"


def parse_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def quarter_for(value: date) -> str:
    return f"{value.year}Q{((value.month - 1) // 3) + 1}"


def add_quarters(start: date, count: int) -> date:
    month_index = (start.year * 12 + start.month - 1) + count * 3
    return date(month_index // 12, month_index % 12 + 1, 1)


def publication_to_3y_quarters(publication_date: date) -> list[str]:
    first = date(publication_date.year, ((publication_date.month - 1) // 3) * 3 + 1, 1)
    end = date(publication_date.year + 3, publication_date.month, min(publication_date.day, 28))
    quarters = []
    cursor = first
    while cursor <= end:
        quarters.append(quarter_for(cursor))
        cursor = add_quarters(cursor, 1)
    return quarters


def alpha_symbol(eodhd_symbol: str | None) -> str | None:
    symbol = str(eodhd_symbol or "").strip()
    if not symbol.endswith(".US"):
        return None
    base = symbol[:-3]
    if not base or "-" in base or "/" in base:
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


def load_transcripts(eodhd_symbol: str | None, publication_date: str | None) -> tuple[list[dict[str, Any]], list[str]]:
    symbol = alpha_symbol(eodhd_symbol)
    pub_date = parse_date(publication_date)
    if not symbol or not pub_date:
        return [], []
    expected = publication_to_3y_quarters(pub_date)
    transcripts = []
    for quarter in expected:
        path = TRANSCRIPT_RAW_DIR / symbol / f"{quarter}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        turns = payload.get("transcript")
        if not isinstance(turns, list) or not turns:
            continue
        transcripts.append(
            {
                "provider": "alpha_vantage",
                "symbol": symbol,
                "quarter": quarter,
                "raw_cache_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "turn_count": len(turns),
                "transcript": turns,
            }
        )
    return transcripts, expected


def enrich_analysis_row(row: dict[str, Any]) -> dict[str, Any]:
    transcripts, expected = load_transcripts(row.get("eodhd_symbol"), row.get("publication_date"))
    enriched = dict(row)
    enriched["earnings_transcripts_publication_to_3y"] = transcripts
    enriched["earnings_transcript_context_counts"] = {
        "provider": "alpha_vantage",
        "expected_quarters_publication_to_3y": len(expected),
        "quarters_with_transcripts": len(transcripts),
        "quarters_missing_transcripts": max(len(expected) - len(transcripts), 0),
        "available_quarters": [item["quarter"] for item in transcripts],
        "missing_quarters": [
            quarter for quarter in expected if quarter not in {item["quarter"] for item in transcripts}
        ],
    }
    enriched["transcript_leakage_note"] = (
        "These transcripts are from the publication quarter through three years after publication. "
        "They are post-publication context and should not be used as predictive input unless the task "
        "explicitly allows post-pitch evidence."
    )
    return enriched


def enrich_sft_row(row: dict[str, Any], transcript_counts: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    enriched["earnings_transcript_context_counts"] = transcript_counts
    return enriched


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sft").mkdir(parents=True, exist_ok=True)

    summary = json.loads((CLEAN_DIR / "dataset_summary.json").read_text(encoding="utf-8"))
    enriched_summary = dict(summary)
    transcript_totals = {
        "rows_with_any_transcript": 0,
        "total_transcript_quarters_attached": 0,
        "total_expected_row_quarters": 0,
    }

    counts_by_idea: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        analysis_rows = []
        for row in iter_jsonl(CLEAN_DIR / "analysis" / f"{split}.jsonl"):
            enriched = enrich_analysis_row(row)
            counts = enriched["earnings_transcript_context_counts"]
            counts_by_idea[str(row.get("idea_id"))] = counts
            transcript_totals["total_expected_row_quarters"] += counts["expected_quarters_publication_to_3y"]
            transcript_totals["total_transcript_quarters_attached"] += counts["quarters_with_transcripts"]
            if counts["quarters_with_transcripts"] > 0:
                transcript_totals["rows_with_any_transcript"] += 1
            analysis_rows.append(enriched)
        write_jsonl(OUT_DIR / "analysis" / f"{split}.jsonl", analysis_rows)

        sft_rows = []
        for row in iter_jsonl(CLEAN_DIR / "sft" / f"{split}.jsonl"):
            sft_rows.append(enrich_sft_row(row, counts_by_idea.get(str(row.get("idea_id")), {})))
        write_jsonl(OUT_DIR / "sft" / f"{split}.jsonl", sft_rows)

    enriched_summary["transcripts"] = {
        "provider": "alpha_vantage",
        "source_cache_dir": str(TRANSCRIPT_RAW_DIR),
        **transcript_totals,
    }
    (OUT_DIR / "dataset_summary.json").write_text(
        json.dumps(enriched_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    readme = (CLEAN_DIR / "README.md").read_text(encoding="utf-8")
    readme += (
        "\n\n## Transcript Enrichment\n\n"
        "This staged variant attaches cached Alpha Vantage earnings call transcripts "
        "from the publication quarter through three years after each pitch when "
        "available. Transcript fields are post-publication context and should be "
        "kept out of predictive model inputs unless the task explicitly allows "
        "post-pitch evidence.\n\n"
        "```json\n"
        f"{json.dumps(enriched_summary['transcripts'], indent=2, sort_keys=True)}\n"
        "```\n"
    )
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(enriched_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
