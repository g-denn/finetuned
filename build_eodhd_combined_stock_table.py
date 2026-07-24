from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "eodhd_output" / "dataset_financial_pull"
RAW_DIR = DATASET_DIR / "raw"
MANIFEST_CSV = DATASET_DIR / "stock_pull_manifest.csv"
OUT_CSV = DATASET_DIR / "eodhd_combined_stock_table.csv"
OUT_JSON = DATASET_DIR / "eodhd_combined_stock_table_summary.json"

SCALAR_SECTIONS = [
    "General",
    "Highlights",
    "Valuation",
    "SharesStats",
    "Technicals",
    "SplitsDividends",
]

STATEMENT_SECTIONS = {
    "Balance_Sheet": "balance_sheet",
    "Cash_Flow": "cash_flow",
    "Income_Statement": "income_statement",
}


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def clean_column(value: str) -> str:
    out = []
    prev_underscore = False
    for char in value:
        if char.isalnum():
            out.append(char.lower())
            prev_underscore = False
        else:
            if not prev_underscore:
                out.append("_")
            prev_underscore = True
    return "".join(out).strip("_")


def add_scalar_section(row: dict[str, Any], prefix: str, payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    for key, value in payload.items():
        if is_scalar(value):
            row[f"{prefix}_{clean_column(str(key))}"] = value


def latest_record(records: Any) -> dict[str, Any] | None:
    if not isinstance(records, dict) or not records:
        return None
    keys = sorted(records.keys(), reverse=True)
    for key in keys:
        value = records.get(key)
        if isinstance(value, dict):
            return value
    return None


def add_latest_statement_fields(
    row: dict[str, Any],
    fundamentals: dict[str, Any],
    statement_key: str,
    prefix: str,
    period: str,
) -> None:
    records = (
        fundamentals.get("Financials", {})
        .get(statement_key, {})
        .get(period, {})
    )
    latest = latest_record(records)
    if not latest:
        return
    for key, value in latest.items():
        if is_scalar(value):
            row[f"{prefix}_{period}_{clean_column(str(key))}"] = value


def adjusted_return(first: Any, last: Any) -> float | None:
    try:
        first_num = float(first)
        last_num = float(last)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(first_num) or not math.isfinite(last_num) or first_num == 0:
        return None
    return (last_num / first_num) - 1.0


def add_eod_summary(row: dict[str, Any], eod_rows: Any) -> None:
    if not isinstance(eod_rows, list) or not eod_rows:
        row["eod_observation_count"] = 0
        return

    ordered = sorted(
        (item for item in eod_rows if isinstance(item, dict) and item.get("date")),
        key=lambda item: str(item.get("date")),
    )
    if not ordered:
        row["eod_observation_count"] = 0
        return

    first = ordered[0]
    latest = ordered[-1]
    row["eod_observation_count"] = len(ordered)
    row["eod_first_date"] = first.get("date")
    row["eod_latest_date"] = latest.get("date")

    for label, payload in [("first", first), ("latest", latest)]:
        for key in ["open", "high", "low", "close", "adjusted_close", "volume"]:
            row[f"eod_{label}_{key}"] = payload.get(key)

    row["eod_adjusted_return_total"] = adjusted_return(
        first.get("adjusted_close"),
        latest.get("adjusted_close"),
    )

    volumes = []
    adjusted = []
    for item in ordered[-252:]:
        try:
            if item.get("volume") is not None:
                volumes.append(float(item.get("volume")))
            if item.get("adjusted_close") is not None:
                adjusted.append(float(item.get("adjusted_close")))
        except (TypeError, ValueError):
            continue
    if volumes:
        row["eod_avg_volume_last_252"] = sum(volumes) / len(volumes)
    if adjusted:
        row["eod_min_adjusted_close_last_252"] = min(adjusted)
        row["eod_max_adjusted_close_last_252"] = max(adjusted)


def build_row(manifest_row: dict[str, Any]) -> dict[str, Any]:
    symbol = manifest_row["symbol"]
    symbol_dir = RAW_DIR / symbol
    fundamentals = read_json(symbol_dir / "fundamentals.json") or {}
    eod_rows = read_json(symbol_dir / "eod_daily.json") or []

    row: dict[str, Any] = {}
    for key, value in manifest_row.items():
        row[f"manifest_{clean_column(key)}"] = value

    row["symbol"] = symbol
    row["raw_eod_daily_path"] = f"raw/{symbol}/eod_daily.json"
    row["raw_fundamentals_path"] = f"raw/{symbol}/fundamentals.json"

    if isinstance(fundamentals, dict):
        for section in SCALAR_SECTIONS:
            add_scalar_section(row, clean_column(section), fundamentals.get(section))

        for statement_key, prefix in STATEMENT_SECTIONS.items():
            add_latest_statement_fields(row, fundamentals, statement_key, prefix, "yearly")
            add_latest_statement_fields(row, fundamentals, statement_key, prefix, "quarterly")

        earnings = fundamentals.get("Earnings", {})
        if isinstance(earnings, dict):
            add_scalar_section(row, "earnings_annual", latest_record(earnings.get("Annual")))
            add_scalar_section(row, "earnings_trend", latest_record(earnings.get("Trend")))

    add_eod_summary(row, eod_rows)
    return row


def main() -> None:
    if not MANIFEST_CSV.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST_CSV}")

    rows = []
    with MANIFEST_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for manifest_row in reader:
            rows.append(build_row(manifest_row))

    frame = pd.DataFrame(rows)
    ordered_columns = ["symbol"] + [col for col in frame.columns if col != "symbol"]
    frame = frame[ordered_columns]

    tmp_csv = OUT_CSV.with_suffix(".csv.tmp")
    frame.to_csv(tmp_csv, index=False)
    tmp_csv.replace(OUT_CSV)

    summary = {
        "table_path": str(OUT_CSV),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "source_symbols": len(rows),
        "raw_dir": str(RAW_DIR),
        "manifest": str(MANIFEST_CSV),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
