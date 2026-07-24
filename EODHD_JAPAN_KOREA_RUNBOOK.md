# EODHD Japan/Korea Fundamentals Runbook

This runbook collects all available EODHD common-stock fundamentals for Japan
and South Korea, writes raw and normalized outputs, and uploads the dataset to a
private Hugging Face dataset repo.

## Required Secrets

Set secrets only in the shell environment. Do not hardcode them in files.

```powershell
$env:EODHD_API_TOKEN = "..."
$env:HF_TOKEN = "..."
$env:HF_REPO_ID = "username/eodhd-japan-korea-fundamentals"
```

Because a Hugging Face write token was pasted into chat, rotate that token after
the upload and use a new token for future long-term automation.

## Smoke Test

Fetch one discovered symbol and build normalized outputs:

```powershell
python eodhd_japan_korea_fundamentals.py --limit 1 --sleep 0.25
```

## Full Collection

Run the full resumable pull:

```powershell
python eodhd_japan_korea_fundamentals.py --sleep 0.25 --retries 3
```

If the run is interrupted, rerun the same command. Existing raw fundamentals
JSON files are reused unless `--force` is passed.

For a full collect, normalize, screen, and upload run:

```powershell
.\run_japan_korea_eodhd_pipeline.ps1 -Sleep 0.25 -Retries 3 -MaxCallsPerMinute 120 -Top 100
```

For collection and screening without upload:

```powershell
.\run_japan_korea_eodhd_pipeline.ps1 -SkipUpload
```

Useful variants:

```powershell
python eodhd_japan_korea_fundamentals.py --status-only
python eodhd_japan_korea_fundamentals.py --normalize-only
python eodhd_japan_korea_fundamentals.py --offset 1000 --limit 500
python eodhd_japan_korea_fundamentals.py --include-delisted
python eodhd_japan_korea_fundamentals.py --max-calls-per-minute 120
```

## Outputs

The collector writes to `eodhd_output/japan_korea_fundamentals`.
Set `EODHD_JK_OUT_DIR` to override this path for tests or alternate runs.

- `raw/<symbol>/fundamentals.json`: original EODHD Fundamentals API v1.1 JSON.
- `normalized/companies.csv`: company-level scalar fundamentals.
- `normalized/income_statement.csv`: annual and quarterly income statement rows.
- `normalized/balance_sheet.csv`: annual and quarterly balance sheet rows.
- `normalized/cash_flow.csv`: annual and quarterly cash flow rows.
- `normalized/income_statement_latest_5y.csv`: latest five annual income
  statement rows per symbol.
- `normalized/balance_sheet_latest_5y.csv`: latest five annual balance sheet
  rows per symbol.
- `normalized/cash_flow_latest_5y.csv`: latest five annual cash flow rows per
  symbol.
- `normalized/earnings.csv`: available earnings rows.
- `normalized/fundamentals_raw_payloads.jsonl`: raw payload JSONL.
- `stock_pull_manifest.csv`: discovered Japan/Korea common-stock universe.
- `progress_summary.json`: coverage by country and completion totals.
- `progress_checklist.csv`: per-symbol completion/errors.
- `earnings_transcript_availability.json`: transcript availability audit.

## Hugging Face Upload

Upload to the private dataset repo named by `HF_REPO_ID`:

```powershell
python upload_japan_korea_fundamentals_to_huggingface.py
```

The upload script creates the dataset repo if needed, makes it private, uploads
the inspectable folder layout, and writes
`eodhd_output/hf_japan_korea_fundamentals_upload_status.json`.

## Shareholder Yield Screening

After the dataset has raw fundamentals files, run:

```powershell
python screen_japan_korea_shareholder_yield.py --top 100
```

Outputs are written to `eodhd_output/japan_korea_fundamentals/screening`.

- `shareholder_yield_screen_all.csv`: all ranked companies.
- `shareholder_yield_screen_top.csv`: top ranked companies.
- `screening_skipped.csv`: missing-data and filter skips.
- `screening_summary.json`: metric definitions and output paths.

The primary rank is shareholder yield:

```text
(latest buyback/share-repurchase cash outflow + latest dividends paid cash outflow) / market cap
```

The screener also calculates average ROE, revenue CAGR, net income CAGR, and
free cash flow CAGR using the latest usable five annual records. Every row
includes source-field columns and data-quality warnings, because EODHD cash-flow
field names and sign conventions can vary by market and reporting standard.

## Completion Audit

After collection and screening, run:

```powershell
python audit_japan_korea_eodhd_dataset.py --require-screening
```

After upload, run:

```powershell
python audit_japan_korea_eodhd_dataset.py --require-screening --require-upload-status
```

The audit writes `dataset_audit.json` and exits non-zero if required artifacts,
raw payloads, normalized statement tables, transcript status, screening outputs,
or upload verification are missing.

## Transcript Status

The public EODHD Fundamentals and Calendar docs document earnings history,
earnings trends, annual/quarterly earnings, and financial statements. They do
not document earnings transcript text. The collector records this in
`earnings_transcript_availability.json` and scans returned payload keys for any
transcript-like fields in case the account receives undocumented fields.
