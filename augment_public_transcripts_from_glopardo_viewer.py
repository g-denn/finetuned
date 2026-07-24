from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import timedelta
from pathlib import Path
from typing import Any

from augment_public_transcripts_with_deerfieldgreen import (
    CLEAN_TRANSCRIPT_STAGE,
    RAW_COMPANION_DIR,
    RAW_COVERAGE_JSONL,
    RAW_SUMMARY_JSON,
    RAW_TRANSCRIPTS_JSONL,
    base_ticker,
    iter_jsonl,
    iter_raw_rows,
    merge_transcripts,
    parse_date,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parent
PROVIDER = "glopardo/sp500-earnings-transcripts"
DATASET = "glopardo/sp500-earnings-transcripts"
CONFIG = "default"
SPLIT = "train"
PAGE_SIZE = 100


def fetch_json(url: str, retries: int = 4) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"Failed after {retries} attempts: {url}") from last_error


def raw_tickers() -> set[str]:
    tickers = set()
    for row in iter_raw_rows():
        ticker = base_ticker(row.get("eodhd_symbol"))
        if ticker:
            tickers.add(ticker)
    return tickers


def stream_glopardo_index(wanted_tickers: set[str]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    total = None
    offset = 0
    while total is None or offset < total:
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "offset": offset,
                "length": PAGE_SIZE,
            }
        )
        payload = fetch_json(f"https://datasets-server.huggingface.co/rows?{query}")
        total = int(payload.get("num_rows_total") or 0)
        for wrapped in payload.get("rows") or []:
            row = wrapped.get("row") or {}
            ticker = str(row.get("ticker") or "").upper().strip()
            if ticker not in wanted_tickers:
                continue
            call_date = parse_date(row.get("earnings_date"))
            transcript = str(row.get("transcript") or "").strip()
            if call_date is None or not transcript:
                continue
            item = {
                "provider": PROVIDER,
                "ticker": ticker,
                "company": row.get("company"),
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
        offset += PAGE_SIZE
        print(json.dumps({"offset": offset, "total": total, "matched_source_rows": sum(len(v) for v in index.values())}))
    for rows in index.values():
        rows.sort(key=lambda item: item["call_date"])
    return index


def source_matches(index: dict[str, list[dict[str, Any]]], row: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
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


def load_existing_raw_matches() -> dict[str, list[dict[str, Any]]]:
    by_idea: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(RAW_TRANSCRIPTS_JSONL):
        by_idea[str(row.get("idea_id"))] = list(row.get("transcripts") or [])
    return by_idea


def update_raw(index: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    previous_summary = json.loads(RAW_SUMMARY_JSON.read_text(encoding="utf-8"))
    existing = load_existing_raw_matches()
    body_rows = []
    coverage_rows = []
    summary = {
        "raw_rows": 0,
        "rows_with_any_publication_to_3y_transcript": 0,
        "total_publication_to_3y_transcripts": 0,
        "total_publication_to_3y_transcript_chars": 0,
        "source_datasets": sorted(set((previous_summary.get("source_datasets") or []) + [PROVIDER])),
        "license_name": "mixed-public-hf-review-source-terms",
        "glopardo_source_rows_for_vic_tickers": sum(len(v) for v in index.values()),
        "glopardo_source_tickers_for_vic": len(index),
    }
    for row in iter_raw_rows():
        summary["raw_rows"] += 1
        idea_id = str(row.get("idea_id"))
        matches, source_ticker_total = source_matches(index, row)
        merged = merge_transcripts(existing.get(idea_id, []), matches)
        chars = sum(int(item.get("transcript_chars") or 0) for item in merged)
        if merged:
            summary["rows_with_any_publication_to_3y_transcript"] += 1
            summary["total_publication_to_3y_transcripts"] += len(merged)
            summary["total_publication_to_3y_transcript_chars"] += chars
            body_rows.append(
                {
                    "idea_id": row.get("idea_id"),
                    "eodhd_symbol": row.get("eodhd_symbol"),
                    "company_name": row.get("company_name"),
                    "publication_date": row.get("publication_date"),
                    "transcripts": merged,
                }
            )
        providers = sorted({str(item.get("provider")) for item in merged if item.get("provider")})
        coverage_rows.append(
            {
                "idea_id": row.get("idea_id"),
                "eodhd_symbol": row.get("eodhd_symbol"),
                "company_name": row.get("company_name"),
                "publication_date": row.get("publication_date"),
                "provider": "multiple_public_hf_transcript_sources",
                "providers": providers,
                "glopardo_ticker_transcript_count_all_dates": source_ticker_total,
                "publication_to_3y_transcript_count": len(merged),
                "publication_to_3y_transcript_chars": chars,
                "available_call_dates": [item["call_date"] for item in merged],
            }
        )
    write_jsonl(RAW_TRANSCRIPTS_JSONL, body_rows)
    write_jsonl(RAW_COVERAGE_JSONL, coverage_rows)
    summary["transcript_jsonl"] = str(RAW_TRANSCRIPTS_JSONL)
    summary["coverage_jsonl"] = str(RAW_COVERAGE_JSONL)
    summary["note"] = "Transcript bodies are stored only for rows with at least one matching public transcript. Coverage JSONL contains one row per raw VIC row."
    RAW_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def update_clean(index: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary_path = CLEAN_TRANSCRIPT_STAGE / "dataset_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    previous_sources = summary.get("transcripts", {}).get("source_datasets") or []
    totals = {
        "rows": 0,
        "rows_with_any_publication_to_3y_transcript": 0,
        "total_publication_to_3y_transcripts": 0,
        "total_publication_to_3y_transcript_chars": 0,
        "source_datasets": sorted(set(previous_sources + [PROVIDER])),
        "license_name": "mixed-public-hf-review-source-terms",
        "glopardo_source_rows_for_vic_tickers": sum(len(v) for v in index.values()),
        "glopardo_source_tickers_for_vic": len(index),
    }
    counts_by_idea: dict[str, dict[str, Any]] = {}
    coverage_rows = []
    for split in ("train", "validation", "test"):
        analysis_path = CLEAN_TRANSCRIPT_STAGE / "analysis" / f"{split}.jsonl"
        analysis_rows = []
        for row in iter_jsonl(analysis_path):
            matches, source_ticker_total = source_matches(index, row)
            merged = merge_transcripts(row.get("earnings_transcripts_publication_to_3y") or [], matches)
            row["earnings_transcripts_publication_to_3y"] = merged
            chars = sum(int(item.get("transcript_chars") or 0) for item in merged)
            providers = sorted({str(item.get("provider")) for item in merged if item.get("provider")})
            counts = {
                "provider": "multiple_public_hf_transcript_sources",
                "providers": providers,
                "glopardo_ticker_transcript_count_all_dates": source_ticker_total,
                "publication_to_3y_transcript_count": len(merged),
                "publication_to_3y_transcript_chars": chars,
                "available_call_dates": [item["call_date"] for item in merged],
            }
            row["earnings_transcript_context_counts"] = counts
            counts_by_idea[str(row.get("idea_id"))] = counts
            totals["rows"] += 1
            totals["total_publication_to_3y_transcripts"] += len(merged)
            totals["total_publication_to_3y_transcript_chars"] += chars
            if merged:
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
    index = stream_glopardo_index(raw_tickers())
    raw_summary = update_raw(index)
    clean_summary = update_clean(index)
    print(json.dumps({"raw": raw_summary, "clean": clean_summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
