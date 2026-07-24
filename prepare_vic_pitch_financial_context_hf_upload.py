from __future__ import annotations

import json
import shutil
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context"
FULL_JSONL = OUT_DIR / "vic_pitch_financial_context.jsonl"
UPLOAD_DIR = Path(os.environ.get("VIC_PITCH_CONTEXT_UPLOAD_DIR", str(OUT_DIR))).resolve()
DATA_DIR = UPLOAD_DIR / "data"
SOURCE_SUMMARY_PATH = OUT_DIR / "dataset_summary.json"
SOURCE_PREVIEW_CSV = OUT_DIR / "vic_pitch_financial_context_preview.csv"
SUMMARY_PATH = UPLOAD_DIR / "dataset_summary.json"
README_PATH = UPLOAD_DIR / "README.md"
APPLE_EXAMPLES = UPLOAD_DIR / "apple_examples.jsonl"
PREVIEW_CSV = UPLOAD_DIR / "vic_pitch_financial_context_preview.csv"

MAX_SHARD_BYTES = 400 * 1024 * 1024


def shard_jsonl() -> list[dict]:
    if not FULL_JSONL.exists():
        raise SystemExit(f"Missing full JSONL: {FULL_JSONL}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for old in DATA_DIR.glob("vic_pitch_financial_context-*.jsonl"):
        old.unlink()

    shard_paths: list[Path] = []
    shard_index = 0
    current_size = 0
    current_rows = 0
    current = None
    shard_rows: list[int] = []

    def open_next():
        nonlocal shard_index, current_size, current_rows, current
        if current is not None:
            current.close()
            shard_rows.append(current_rows)
        path = DATA_DIR / f"vic_pitch_financial_context-{shard_index:05d}.jsonl"
        shard_paths.append(path)
        shard_index += 1
        current_size = 0
        current_rows = 0
        current = path.open("wb")

    open_next()
    with FULL_JSONL.open("rb") as source:
        for line in source:
            if current_size and current_size + len(line) > MAX_SHARD_BYTES:
                open_next()
            current.write(line)
            current_size += len(line)
            current_rows += 1
    if current is not None:
        current.close()
        shard_rows.append(current_rows)

    return [
        {"path": str(path.relative_to(UPLOAD_DIR)).replace("\\", "/"), "bytes": path.stat().st_size, "rows": rows}
        for path, rows in zip(shard_paths, shard_rows)
    ]


def write_apple_examples() -> int:
    count = 0
    with FULL_JSONL.open("r", encoding="utf-8") as source, APPLE_EXAMPLES.open("w", encoding="utf-8", newline="\n") as out:
        for line in source:
            row = json.loads(line)
            if str(row.get("eodhd_symbol", "")).startswith("AAPL"):
                out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
    return count


def write_readme(summary: dict) -> None:
    README_PATH.write_text(
        f"""---
license: other
pretty_name: VIC Pitch Financial Context EODHD
task_categories:
- text-classification
- tabular-regression
language:
- en
tags:
- finance
- eodhd
- value-investing
- financial-statements
- stock-pitches
private-dataset: true
configs:
- config_name: pitch_financial_context
  data_files:
  - split: train
    path: data/*.jsonl
---

# VIC Pitch Financial Context EODHD

Private pitch-level dataset joining full VIC stock pitches to EODHD financial
statement records.

## Files

- `data/*.jsonl`: sharded full dataset, one JSON object per pitch.
- `vic_pitch_financial_context_preview.csv`: compact preview with performance
  fields and financial-context counts.
- `apple_examples.jsonl`: Apple-only example rows.
- `dataset_summary.json`: row counts, shard list, and coverage summary.

## Main Columns

- `full_stock_pitch_text`: full pitch text from the source dataset, preserving
  the full description and catalyst text.
- `performance`: 1y, 3y, 5y, 10y, and 20y raw/directional performance and
  outcome labels where available.
- `financials_trailing_5y_asof_pitch`: full EODHD statement records with
  `filing_date` from five years before the pitch through the publication date.
- `financials_latest_annual_asof_pitch`: latest annual balance sheet, cash flow,
  and income statement filed on or before the pitch.
- `financials_latest_quarterly_asof_pitch`: latest quarterly balance sheet, cash
  flow, and income statement filed on or before the pitch.
- `financials_forward_3y_after_pitch`: statement records filed after the pitch
  through three years after the pitch.
- `financials_forward_5y_after_pitch`: statement records filed after the pitch
  through five years after the pitch.

## Leakage Note

For prediction features, use only the `asof_pitch` columns. The forward columns
are included for outcome analysis and should not be used as model input features.

## Summary

```json
{json.dumps(summary, indent=2, sort_keys=True)}
```
""",
        encoding="utf-8",
    )


def main() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if UPLOAD_DIR != OUT_DIR:
        shutil.copy2(SOURCE_PREVIEW_CSV, PREVIEW_CSV)
    summary = json.loads(SOURCE_SUMMARY_PATH.read_text(encoding="utf-8"))
    shards = shard_jsonl()
    apple_rows = write_apple_examples()
    summary.update(
        {
            "shards": shards,
            "shard_count": len(shards),
            "apple_examples_path": str(APPLE_EXAMPLES),
            "apple_example_rows": apple_rows,
        }
    )
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_readme(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
