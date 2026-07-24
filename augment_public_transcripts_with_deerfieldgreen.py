from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RAW_SHARD_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context_hf_stage" / "data"
RAW_COMPANION_DIR = ROOT / "eodhd_output" / "raw_vic_public_transcript_companion"
RAW_TRANSCRIPTS_JSONL = RAW_COMPANION_DIR / "public_transcripts_publication_to_3y.jsonl"
RAW_COVERAGE_JSONL = RAW_COMPANION_DIR / "public_transcripts_coverage.jsonl"
RAW_SUMMARY_JSON = RAW_COMPANION_DIR / "public_transcripts_summary.json"

DEERFIELDGREEN_JSON = (
    ROOT
    / "eodhd_output"
    / "public_transcript_sources"
    / "deerfieldgreen_stk-earnings-transcripts"
    / "deerfieldgreen_transcripts.json"
)

CLEAN_TRANSCRIPT_STAGE = ROOT / "eodhd_output" / "vic_pitch_financial_context_repaired_clean_transcripts_hf_stage"


def parse_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text[: len(fmt)], fmt).date()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
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


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def iter_raw_rows():
    for path in sorted(RAW_SHARD_DIR.glob("vic_pitch_financial_context-*.jsonl")):
        yield from iter_jsonl(path)


def transcript_key(item: dict[str, Any]) -> str:
    ticker = str(item.get("ticker") or "").upper()
    call_date = str(item.get("call_date") or "")[:10]
    transcript = str(item.get("transcript") or "")
    digest = hashlib.sha1(transcript[:4000].encode("utf-8", errors="ignore")).hexdigest()
    return f"{ticker}|{call_date}|{digest}"


def normalize_existing_raw_matches() -> dict[str, list[dict[str, Any]]]:
    by_idea: dict[str, list[dict[str, Any]]] = {}
    if not RAW_TRANSCRIPTS_JSONL.exists():
        return by_idea
    for row in iter_jsonl(RAW_TRANSCRIPTS_JSONL):
        idea_id = str(row.get("idea_id"))
        by_idea[idea_id] = list(row.get("transcripts") or [])
    return by_idea


def load_deerfieldgreen_index() -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    if not DEERFIELDGREEN_JSON.exists():
        raise SystemExit(f"Missing deerfieldgreen export: {DEERFIELDGREEN_JSON}")
    for row in iter_jsonl(DEERFIELDGREEN_JSON):
        ticker = str(row.get("symbol") or "").upper().strip()
        call_date = parse_date(row.get("date"))
        transcript = str(row.get("content") or "").strip()
        if not ticker or call_date is None or not transcript:
            continue
        item = {
            "provider": "deerfieldgreen/stk-earnings-transcripts",
            "ticker": ticker,
            "company": None,
            "quarter": row.get("quarter"),
            "earnings_year": row.get("year"),
            "call_date": call_date.isoformat(),
            "title": None,
            "source_url": None,
            "scraped_at": None,
            "transcript_chars": len(transcript),
            "transcript": transcript,
        }
        index.setdefault(ticker, []).append(item)
    for rows in index.values():
        rows.sort(key=lambda item: item["call_date"])
    return index


def deerfieldgreen_matches(index: dict[str, list[dict[str, Any]]], row: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
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


def merge_transcripts(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for item in group:
            if not item.get("transcript_chars"):
                item["transcript_chars"] = len(str(item.get("transcript") or ""))
            merged.setdefault(transcript_key(item), item)
    return sorted(merged.values(), key=lambda item: (str(item.get("call_date") or ""), str(item.get("provider") or "")))


def update_raw_companion(index: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    existing = normalize_existing_raw_matches()
    summary = {
        "raw_rows": 0,
        "rows_with_any_publication_to_3y_transcript": 0,
        "total_publication_to_3y_transcripts": 0,
        "total_publication_to_3y_transcript_chars": 0,
        "deerfieldgreen_source_rows": sum(len(rows) for rows in index.values()),
        "deerfieldgreen_source_tickers": len(index),
        "source_datasets": [
            "Rogersurf/earnings-call-transcripts",
            "deerfieldgreen/stk-earnings-transcripts",
        ],
        "license_name": "mixed-public-hf-review-source-terms",
    }
    body_rows = []
    coverage_rows = []
    for row in iter_raw_rows():
        summary["raw_rows"] += 1
        idea_id = str(row.get("idea_id"))
        deer_matches, deer_ticker_total = deerfieldgreen_matches(index, row)
        matches = merge_transcripts(existing.get(idea_id, []), deer_matches)
        transcript_chars = sum(int(item.get("transcript_chars") or 0) for item in matches)
        if matches:
            summary["rows_with_any_publication_to_3y_transcript"] += 1
            summary["total_publication_to_3y_transcripts"] += len(matches)
            summary["total_publication_to_3y_transcript_chars"] += transcript_chars
            body_rows.append(
                {
                    "idea_id": row.get("idea_id"),
                    "eodhd_symbol": row.get("eodhd_symbol"),
                    "company_name": row.get("company_name"),
                    "publication_date": row.get("publication_date"),
                    "transcripts": matches,
                }
            )
        providers = sorted({str(item.get("provider")) for item in matches if item.get("provider")})
        coverage_rows.append(
            {
                "idea_id": row.get("idea_id"),
                "eodhd_symbol": row.get("eodhd_symbol"),
                "company_name": row.get("company_name"),
                "publication_date": row.get("publication_date"),
                "provider": "multiple_public_hf_transcript_sources",
                "providers": providers,
                "deerfieldgreen_ticker_transcript_count_all_dates": deer_ticker_total,
                "publication_to_3y_transcript_count": len(matches),
                "publication_to_3y_transcript_chars": transcript_chars,
                "available_call_dates": [item["call_date"] for item in matches],
            }
        )
    write_jsonl(RAW_TRANSCRIPTS_JSONL, body_rows)
    write_jsonl(RAW_COVERAGE_JSONL, coverage_rows)
    summary.update(
        {
            "transcript_jsonl": str(RAW_TRANSCRIPTS_JSONL),
            "coverage_jsonl": str(RAW_COVERAGE_JSONL),
            "note": "Transcript bodies are stored only for rows with at least one matching public transcript. Coverage JSONL contains one row per raw VIC row.",
        }
    )
    RAW_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def update_clean_stage(index: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary_path = CLEAN_TRANSCRIPT_STAGE / "dataset_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    counts_by_idea: dict[str, dict[str, Any]] = {}
    totals = {
        "rows": 0,
        "rows_with_any_publication_to_3y_transcript": 0,
        "total_publication_to_3y_transcripts": 0,
        "total_publication_to_3y_transcript_chars": 0,
        "deerfieldgreen_source_rows": sum(len(rows) for rows in index.values()),
        "deerfieldgreen_source_tickers": len(index),
        "source_datasets": [
            "Rogersurf/earnings-call-transcripts",
            "deerfieldgreen/stk-earnings-transcripts",
        ],
        "license_name": "mixed-public-hf-review-source-terms",
    }
    coverage_rows = []
    for split in ("train", "validation", "test"):
        analysis_path = CLEAN_TRANSCRIPT_STAGE / "analysis" / f"{split}.jsonl"
        analysis_rows = []
        for row in iter_jsonl(analysis_path):
            deer_matches, deer_ticker_total = deerfieldgreen_matches(index, row)
            matches = merge_transcripts(row.get("earnings_transcripts_publication_to_3y") or [], deer_matches)
            row["earnings_transcripts_publication_to_3y"] = matches
            transcript_chars = sum(int(item.get("transcript_chars") or 0) for item in matches)
            providers = sorted({str(item.get("provider")) for item in matches if item.get("provider")})
            counts = {
                "provider": "multiple_public_hf_transcript_sources",
                "providers": providers,
                "deerfieldgreen_ticker_transcript_count_all_dates": deer_ticker_total,
                "publication_to_3y_transcript_count": len(matches),
                "publication_to_3y_transcript_chars": transcript_chars,
                "available_call_dates": [item["call_date"] for item in matches],
            }
            row["earnings_transcript_context_counts"] = counts
            counts_by_idea[str(row.get("idea_id"))] = counts
            totals["rows"] += 1
            totals["total_publication_to_3y_transcripts"] += len(matches)
            totals["total_publication_to_3y_transcript_chars"] += transcript_chars
            if matches:
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
            analysis_rows.append(row)
        write_jsonl(analysis_path, analysis_rows)

        sft_path = CLEAN_TRANSCRIPT_STAGE / "sft" / f"{split}.jsonl"
        sft_rows = []
        for row in iter_jsonl(sft_path):
            row["earnings_transcript_context_counts"] = counts_by_idea.get(str(row.get("idea_id")), {})
            sft_rows.append(row)
        write_jsonl(sft_path, sft_rows)

    write_jsonl(CLEAN_TRANSCRIPT_STAGE / "transcript_coverage.jsonl", coverage_rows)
    summary["transcripts"] = totals
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return totals


def main() -> None:
    index = load_deerfieldgreen_index()
    raw_summary = update_raw_companion(index)
    clean_summary = update_clean_stage(index)
    print(json.dumps({"raw": raw_summary, "clean": clean_summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
