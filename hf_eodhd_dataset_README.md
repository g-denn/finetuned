---
license: other
pretty_name: VIC EODHD Financial Data Pull
task_categories:
- tabular-regression
- time-series-forecasting
language:
- en
tags:
- finance
- eodhd
- equities
- fundamentals
- daily-prices
private-dataset: true
configs:
- config_name: combined_stock_table
  data_files:
  - split: train
    path: eodhd_combined_stock_table.csv
---

# VIC EODHD Financial Data Pull

Private archive of EODHD financial data fetched for all EODHD symbols present in
the local VIC investment outcome dataset. The dataset is uploaded in an
inspectable folder layout, so files can be opened directly in the Hugging Face
web UI without downloading a zip archive.

## Contents

- `eodhd_combined_stock_table.csv`: one-row-per-symbol combined table for
  direct inspection in the Hugging Face UI.
- `raw/<symbol>/eod_daily.json`: EODHD daily price history for the symbol.
- `raw/<symbol>/fundamentals.json`: EODHD fundamentals payload for the symbol.
- `raw/<symbol>/news.json`: optional news payload for symbols used in the test
  pull.
- `raw/<symbol>/earnings.json`: optional earnings payload for symbols used in
  the test pull.
- `stock_pull_manifest.csv`: one-row-per-symbol manifest.
- `stock_pull_manifest.json`: JSON version of the manifest.
- `progress_summary.json`: completion summary.
- `progress_checklist.csv`: one-row-per-symbol completion checklist.
- `progress_checklist.json`: JSON version of the checklist.

Example paths:

```text
raw/AAPL.US/eod_daily.json
raw/AAPL.US/fundamentals.json
raw/000660.KQ/eod_daily.json
raw/000660.KQ/fundamentals.json
```

## Combined Table

The combined table has one row per symbol. It joins the local manifest, selected
scalar fundamentals fields, latest annual and quarterly financial statement
fields, and daily-price summary fields such as first/latest adjusted close,
latest volume, observation count, and total adjusted return.

## Completion

The local completion audit before upload verified:

- total symbols: 4,541
- EOD daily files: 4,541
- fundamentals files: 4,541
- symbols with errors: 0

## Source

Data was pulled from EODHD using the user's API access. This dataset should
remain private unless the data owner explicitly decides otherwise.
