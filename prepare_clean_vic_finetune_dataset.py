from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context_hf_stage"
SOURCE_DATA_DIR = SOURCE_DIR / "data"
OUT_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context_clean_hf_stage"
ANALYSIS_DIR = OUT_DIR / "analysis"
SFT_DIR = OUT_DIR / "sft"

MIN_PITCH_CHARS = 500
EXPECTED_STATEMENT_COUNT = 3


def parse_date(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d")
    except ValueError:
        return None


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


def clean_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def iter_rows() -> Any:
    paths = sorted(SOURCE_DATA_DIR.glob("vic_pitch_financial_context-*.jsonl"))
    if not paths:
        raise SystemExit(f"No source shards found in {SOURCE_DATA_DIR}")
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    yield path.name, line_number, json.loads(line)


def rejection_reason(row: dict[str, Any]) -> str | None:
    pub_date = parse_date(row.get("publication_date"))
    if pub_date is None:
        return "missing_publication_date"
    if not str(row.get("eodhd_symbol") or "").strip():
        return "missing_eodhd_symbol"
    if len(str(row.get("full_stock_pitch_text") or "")) < MIN_PITCH_CHARS:
        return "pitch_text_too_short"
    if parse_float(row.get("raw_perf_3y")) is None:
        return "missing_3y_return_label"
    if not str(row.get("outcome_3y") or "").strip():
        return "missing_3y_outcome_label"

    counts = row.get("financial_context_counts") or {}
    if not counts.get("has_point_in_time_financials"):
        return "missing_point_in_time_financials"
    if int(counts.get("latest_annual_asof_pitch_records") or 0) < EXPECTED_STATEMENT_COUNT:
        return "missing_complete_latest_annual_statements"
    if int(counts.get("latest_quarterly_asof_pitch_records") or 0) < EXPECTED_STATEMENT_COUNT:
        return "missing_complete_latest_quarterly_statements"
    if int(counts.get("trailing_5y_asof_pitch_records") or 0) == 0:
        return "missing_trailing_financial_history"
    return None


def split_for_date(pub_date: datetime) -> str:
    if pub_date < datetime(2018, 1, 1):
        return "train"
    if pub_date < datetime(2020, 1, 1):
        return "validation"
    return "test"


def compact_statement(record: dict[str, Any]) -> dict[str, Any]:
    data = record.get("data") or {}
    keys = [
        "date",
        "filing_date",
        "currency_symbol",
        "totalAssets",
        "totalLiab",
        "totalStockholderEquity",
        "totalRevenue",
        "grossProfit",
        "operatingIncome",
        "netIncome",
        "totalCashFromOperatingActivities",
        "capitalExpenditures",
        "freeCashFlow",
    ]
    return {
        "statement": record.get("statement"),
        "period": record.get("period"),
        "fiscal_date": record.get("fiscal_date"),
        "filing_date": record.get("filing_date"),
        "data": {key: data.get(key) for key in keys if data.get(key) not in (None, "")},
    }


def make_analysis_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    performance = row.get("performance") or {}
    return {
        "idea_id": row.get("idea_id"),
        "split": split,
        "raw_symbol": row.get("raw_symbol"),
        "eodhd_symbol": row.get("eodhd_symbol"),
        "company_name": row.get("company_name"),
        "publication_date": row.get("publication_date"),
        "is_short": clean_bool(row.get("is_short")),
        "link": row.get("link"),
        "full_stock_pitch_text": row.get("full_stock_pitch_text"),
        "financials_trailing_5y_asof_pitch": row.get("financials_trailing_5y_asof_pitch") or [],
        "financials_latest_annual_asof_pitch": row.get("financials_latest_annual_asof_pitch") or [],
        "financials_latest_quarterly_asof_pitch": row.get("financials_latest_quarterly_asof_pitch") or [],
        "financial_context_counts": row.get("financial_context_counts") or {},
        "fundamentals_snapshot_from_original_dataset": row.get("fundamentals_snapshot_from_original_dataset") or {},
        "eodhd_general": row.get("eodhd_general") or {},
        "label": {
            "raw_perf_3y": parse_float(row.get("raw_perf_3y")),
            "directional_perf_3y": parse_float(performance.get("directional_perf_3y")),
            "outcome_3y": row.get("outcome_3y"),
            "primary_horizon": performance.get("primary_horizon"),
            "primary_outcome": performance.get("primary_outcome"),
        },
        "leakage_note": (
            "This clean row contains only point-in-time financial statement fields for model input. "
            "Forward financial statements are intentionally excluded from the clean fine-tuning split."
        ),
    }


def make_sft_row(analysis_row: dict[str, Any]) -> dict[str, Any]:
    financial_context = {
        "latest_annual": [compact_statement(item) for item in analysis_row["financials_latest_annual_asof_pitch"]],
        "latest_quarterly": [compact_statement(item) for item in analysis_row["financials_latest_quarterly_asof_pitch"]],
        "trailing_5y_record_count": analysis_row["financial_context_counts"].get("trailing_5y_asof_pitch_records"),
    }
    user_payload = {
        "company_name": analysis_row["company_name"],
        "symbol": analysis_row["eodhd_symbol"],
        "publication_date": analysis_row["publication_date"],
        "is_short": analysis_row["is_short"],
        "pitch": analysis_row["full_stock_pitch_text"],
        "point_in_time_financials": financial_context,
    }
    assistant_payload = analysis_row["label"]
    return {
        "idea_id": analysis_row["idea_id"],
        "split": analysis_row["split"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an investment research evaluation assistant. "
                    "Use only information known at the publication date and return strict JSON."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")),
            },
            {
                "role": "assistant",
                "content": json.dumps(assistant_payload, ensure_ascii=False, separators=(",", ":")),
            },
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    SFT_DIR.mkdir(parents=True, exist_ok=True)

    accepted: list[dict[str, Any]] = []
    rejected_counts: dict[str, int] = {}
    total_rows = 0
    symbols: set[str] = set()

    for _source, _line_number, row in iter_rows():
        total_rows += 1
        reason = rejection_reason(row)
        if reason:
            rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
            continue
        pub_date = parse_date(row["publication_date"])
        assert pub_date is not None
        split = split_for_date(pub_date)
        analysis_row = make_analysis_row(row, split)
        accepted.append(analysis_row)
        symbols.add(str(analysis_row["eodhd_symbol"]))

    accepted.sort(key=lambda item: (item["publication_date"], item["idea_id"]))
    by_split = {
        "train": [row for row in accepted if row["split"] == "train"],
        "validation": [row for row in accepted if row["split"] == "validation"],
        "test": [row for row in accepted if row["split"] == "test"],
    }

    sft_by_split = {split: [make_sft_row(row) for row in rows] for split, rows in by_split.items()}
    for split, rows in by_split.items():
        write_jsonl(ANALYSIS_DIR / f"{split}.jsonl", rows)
        write_jsonl(SFT_DIR / f"{split}.jsonl", sft_by_split[split])

    summary = {
        "source_dataset_dir": str(SOURCE_DIR),
        "total_source_rows": total_rows,
        "accepted_rows": len(accepted),
        "rejected_rows": total_rows - len(accepted),
        "rejected_counts": dict(sorted(rejected_counts.items())),
        "unique_symbols": len(symbols),
        "criteria": {
            "min_pitch_chars": MIN_PITCH_CHARS,
            "requires_3y_return_label": True,
            "requires_3y_outcome_label": True,
            "requires_complete_latest_annual_statements": True,
            "requires_complete_latest_quarterly_statements": True,
            "requires_trailing_financial_history": True,
            "excludes_forward_financials_from_clean_model_input": True,
        },
        "splits": {
            split: {
                "rows": len(rows),
                "analysis_file": f"analysis/{split}.jsonl",
                "sft_file": f"sft/{split}.jsonl",
            }
            for split, rows in by_split.items()
        },
    }
    (OUT_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (OUT_DIR / "README.md").write_text(
        f"""---
license: other
pretty_name: VIC Pitch Financial Context Clean SFT
task_categories:
- text-generation
- text-classification
language:
- en
tags:
- finance
- value-investing
- supervised-fine-tuning
- financial-statements
private-dataset: true
configs:
- config_name: sft
  data_files:
  - split: train
    path: sft/train.jsonl
  - split: validation
    path: sft/validation.jsonl
  - split: test
    path: sft/test.jsonl
- config_name: analysis
  data_files:
  - split: train
    path: analysis/train.jsonl
  - split: validation
    path: analysis/validation.jsonl
  - split: test
    path: analysis/test.jsonl
---

# VIC Pitch Financial Context Clean SFT

Private clean fine-tuning dataset derived from `Gden/vic-pitch-financial-context-eodhd`.

Rows are retained only when the pitch has a publication date, usable pitch text,
a 3-year return/outcome label, complete latest annual statements, complete latest
quarterly statements, and trailing point-in-time financial history as of the
publication date.

Forward financial statements are intentionally excluded from the clean model
input to avoid leakage. The `analysis` config keeps richer point-in-time
statement records; the `sft` config is chat-format JSONL ready for supervised
fine-tuning.

## Summary

```json
{json.dumps(summary, indent=2, sort_keys=True)}
```
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
