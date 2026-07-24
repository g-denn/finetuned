from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CANONICAL_CSV = ROOT / "data" / "processed" / "investment_canonical.csv"
RAW_DIR = ROOT / "eodhd_output" / "dataset_financial_pull" / "raw"
OUT_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context_repaired_clean_hf_stage"
ANALYSIS_DIR = OUT_DIR / "analysis"
SFT_DIR = OUT_DIR / "sft"

MIN_PITCH_CHARS = 500
STATEMENTS = {
    "Balance_Sheet": "balance_sheet",
    "Cash_Flow": "cash_flow",
    "Income_Statement": "income_statement",
}
PERIODS = ("yearly", "quarterly")
FALLBACK_LAGS = {"quarterly": 60, "yearly": 120}


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
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    data = record["data"]
    return {
        "statement": record["statement"],
        "period": record["period"],
        "fiscal_date": record["fiscal_date"].isoformat() if record["fiscal_date"] else None,
        "filing_date": record["filing_date"].isoformat() if record["filing_date"] else None,
        "effective_availability_date": record["effective_date"].isoformat() if record["effective_date"] else None,
        "availability_date_source": record["date_source"],
        "currency_symbol": data.get("currency_symbol"),
        "data": json_ready(data),
    }


def extract_records(symbol: str) -> dict[str, Any]:
    path = RAW_DIR / symbol / "fundamentals.json"
    if not path.exists():
        return {"records": [], "general": {}, "fundamentals_path": f"raw/{symbol}/fundamentals.json"}
    try:
        fundamentals = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"records": [], "general": {}, "fundamentals_path": f"raw/{symbol}/fundamentals.json"}

    records: list[dict[str, Any]] = []
    financials = fundamentals.get("Financials", {})
    if isinstance(financials, dict):
        for source_key, statement_name in STATEMENTS.items():
            statement = financials.get(source_key, {})
            if not isinstance(statement, dict):
                continue
            for period in PERIODS:
                period_records = statement.get(period, {})
                if not isinstance(period_records, dict):
                    continue
                for record_key, payload in period_records.items():
                    if not isinstance(payload, dict):
                        continue
                    fiscal_date = parse_date(payload.get("date") or record_key)
                    filing_date = parse_date(payload.get("filing_date"))
                    fallback_date = fiscal_date + timedelta(days=FALLBACK_LAGS[period]) if fiscal_date else None
                    effective_date = filing_date or fallback_date
                    records.append(
                        {
                            "statement": statement_name,
                            "period": period,
                            "fiscal_date": fiscal_date,
                            "filing_date": filing_date,
                            "effective_date": effective_date,
                            "date_source": "filing_date" if filing_date else "estimated_from_fiscal_date",
                            "data": payload,
                        }
                    )
    general = fundamentals.get("General", {})
    return {
        "records": records,
        "general": json_ready(general) if isinstance(general, dict) else {},
        "fundamentals_path": f"raw/{symbol}/fundamentals.json",
    }


def latest_by_statement(records: list[dict[str, Any]], period: str, pub_date: date) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        effective_date = record["effective_date"]
        if record["period"] != period or effective_date is None or effective_date > pub_date:
            continue
        current = latest.get(record["statement"])
        candidate_key = (effective_date, record["fiscal_date"] or date.min)
        current_key = (current["effective_date"], current["fiscal_date"] or date.min) if current else None
        if current_key is None or candidate_key > current_key:
            latest[record["statement"]] = record
    return [compact_record(latest[key]) for key in sorted(latest)]


def window_records(records: list[dict[str, Any]], start: date, end: date) -> list[dict[str, Any]]:
    return [
        compact_record(record)
        for record in records
        if record["effective_date"] is not None and start <= record["effective_date"] <= end
    ]


def full_pitch_text(row: dict[str, Any]) -> str:
    parts = []
    if row.get("description"):
        parts.append("Description:\n" + row["description"])
    if row.get("catalyst"):
        parts.append("Catalyst:\n" + row["catalyst"])
    return "\n\n".join(parts)


def has_estimated(records: list[dict[str, Any]]) -> bool:
    return any(record.get("availability_date_source") == "estimated_from_fiscal_date" for record in records)


def split_for(pub_date: date) -> str:
    if pub_date < date(2018, 1, 1):
        return "train"
    if pub_date < date(2020, 1, 1):
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
        "effective_availability_date": record.get("effective_availability_date"),
        "availability_date_source": record.get("availability_date_source"),
        "data": {key: data.get(key) for key in keys if data.get(key) not in (None, "")},
    }


def make_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    user_payload = {
        "company_name": row["company_name"],
        "symbol": row["eodhd_symbol"],
        "publication_date": row["publication_date"],
        "is_short": row["is_short"],
        "pitch": row["full_stock_pitch_text"],
        "point_in_time_financials": {
            "latest_annual": [compact_statement(item) for item in row["financials_latest_annual_asof_pitch"]],
            "latest_quarterly": [compact_statement(item) for item in row["financials_latest_quarterly_asof_pitch"]],
            "trailing_5y_record_count": row["financial_context_counts"]["trailing_5y_asof_pitch_records"],
            "uses_estimated_availability_dates": row["financial_context_counts"]["uses_estimated_availability_dates"],
        },
    }
    assistant_payload = row["label"]
    return {
        "idea_id": row["idea_id"],
        "split": row["split"],
        "messages": [
            {
                "role": "system",
                "content": "You are an investment research evaluation assistant. Use only information available at publication and return strict JSON.",
            },
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False, separators=(",", ":"))},
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
    symbol_cache: dict[str, dict[str, Any]] = {}
    accepted: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}

    with CANONICAL_CSV.open("r", encoding="utf-8", newline="") as handle:
        for pitch in csv.DictReader(handle):
            pub_date = parse_date(pitch.get("publication_date"))
            text = full_pitch_text(pitch)
            raw_perf_3y = parse_float(pitch.get("raw_perf_3y"))
            if pub_date is None:
                rejected["missing_publication_date"] = rejected.get("missing_publication_date", 0) + 1
                continue
            if len(text) < MIN_PITCH_CHARS:
                rejected["pitch_text_too_short"] = rejected.get("pitch_text_too_short", 0) + 1
                continue
            if raw_perf_3y is None:
                rejected["missing_3y_return_label"] = rejected.get("missing_3y_return_label", 0) + 1
                continue
            if not str(pitch.get("outcome_3y") or "").strip():
                rejected["missing_3y_outcome_label"] = rejected.get("missing_3y_outcome_label", 0) + 1
                continue

            symbol = str(pitch.get("eodhd_symbol") or "").strip()
            symbol_data = symbol_cache.setdefault(symbol, extract_records(symbol)) if symbol else {
                "records": [],
                "general": {},
                "fundamentals_path": None,
            }
            records = symbol_data["records"]
            trailing = window_records(records, pub_date - timedelta(days=365 * 5), pub_date)
            latest_annual = latest_by_statement(records, "yearly", pub_date)
            latest_quarterly = latest_by_statement(records, "quarterly", pub_date)
            if not trailing or len(latest_annual) < 3 or len(latest_quarterly) < 3:
                rejected["missing_repaired_model_ready_financials"] = rejected.get(
                    "missing_repaired_model_ready_financials", 0
                ) + 1
                continue

            split = split_for(pub_date)
            counts = {
                "all_statement_records_available": len(records),
                "trailing_5y_asof_pitch_records": len(trailing),
                "latest_annual_asof_pitch_records": len(latest_annual),
                "latest_quarterly_asof_pitch_records": len(latest_quarterly),
                "uses_estimated_availability_dates": has_estimated(trailing + latest_annual + latest_quarterly),
                "estimated_trailing_5y_records": sum(
                    1 for item in trailing if item["availability_date_source"] == "estimated_from_fiscal_date"
                ),
                "estimated_latest_annual_records": sum(
                    1 for item in latest_annual if item["availability_date_source"] == "estimated_from_fiscal_date"
                ),
                "estimated_latest_quarterly_records": sum(
                    1 for item in latest_quarterly if item["availability_date_source"] == "estimated_from_fiscal_date"
                ),
            }
            row = {
                "idea_id": pitch.get("idea_id"),
                "split": split,
                "raw_symbol": pitch.get("raw_symbol"),
                "eodhd_symbol": symbol,
                "company_name": pitch.get("company_name"),
                "publication_date": pitch.get("publication_date"),
                "is_short": clean_bool(pitch.get("is_short")),
                "link": pitch.get("link"),
                "full_stock_pitch_text": text,
                "financials_trailing_5y_asof_pitch": trailing,
                "financials_latest_annual_asof_pitch": latest_annual,
                "financials_latest_quarterly_asof_pitch": latest_quarterly,
                "financial_context_counts": counts,
                "eodhd_general": symbol_data["general"],
                "raw_fundamentals_path": symbol_data["fundamentals_path"],
                "label": {
                    "raw_perf_3y": raw_perf_3y,
                    "directional_perf_3y": parse_float(pitch.get("directional_perf_3y")),
                    "outcome_3y": pitch.get("outcome_3y"),
                    "primary_horizon": pitch.get("primary_horizon"),
                    "primary_outcome": pitch.get("primary_outcome"),
                },
                "financial_date_repair_note": (
                    "Rows may use estimated effective availability dates when EODHD statement records have fiscal dates "
                    "but no filing_date. Quarterly records use fiscal_date+60 days; annual records use fiscal_date+120 days. "
                    "Each statement record carries availability_date_source."
                ),
            }
            accepted.append(row)

    accepted.sort(key=lambda item: (item["publication_date"], item["idea_id"]))
    by_split = {split: [row for row in accepted if row["split"] == split] for split in ("train", "validation", "test")}
    for split, rows in by_split.items():
        write_jsonl(ANALYSIS_DIR / f"{split}.jsonl", rows)
        write_jsonl(SFT_DIR / f"{split}.jsonl", [make_sft_row(row) for row in rows])

    summary = {
        "source": str(CANONICAL_CSV),
        "accepted_rows": len(accepted),
        "rejected_counts": dict(sorted(rejected.items())),
        "unique_symbols": len({row["eodhd_symbol"] for row in accepted}),
        "rows_using_estimated_availability_dates": sum(
            1 for row in accepted if row["financial_context_counts"]["uses_estimated_availability_dates"]
        ),
        "fallback_lags_days": FALLBACK_LAGS,
        "splits": {
            split: {
                "rows": len(rows),
                "analysis_file": f"analysis/{split}.jsonl",
                "sft_file": f"sft/{split}.jsonl",
            }
            for split, rows in by_split.items()
        },
    }
    (OUT_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        f"""---
license: other
pretty_name: VIC Pitch Financial Context Repaired Clean SFT
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

# VIC Pitch Financial Context Repaired Clean SFT

Clean fine-tuning dataset with conservative financial statement date repair.
When EODHD records lack `filing_date`, this build estimates an effective
availability date from fiscal date: quarterly +60 days, annual +120 days. Every
statement record carries `availability_date_source`.

Forward financial statements and earnings transcripts are not included in model
input fields.

```json
{json.dumps(summary, indent=2, sort_keys=True)}
```
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
