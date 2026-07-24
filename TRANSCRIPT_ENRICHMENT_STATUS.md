# Transcript Enrichment Status

## Current Published Dataset

Private Hugging Face dataset:

`Gden/vic-pitch-financial-context-repaired-clean-transcripts`

This dataset includes:

- 5,639 repaired clean VIC rows.
- Row-level coverage manifests proving pitch text, publication date, 3-year return label, and model-ready EODHD financial statement context for every included row.
- 9,036 attached earnings-call transcript bodies across 843 rows, joined from public Hugging Face transcript datasets by ticker and call date within publication date through publication date + 3 years.

Transcript body text is included only in `analysis/*.jsonl`. The `sft/*.jsonl` files include transcript coverage metadata only, to avoid post-publication leakage in predictive inputs.

## Full Raw Dataset Audit

Private Hugging Face dataset:

`Gden/vic-pitch-financial-context-eodhd`

Uploaded audit files:

- `audits/vic_row_level_readiness.csv`
- `audits/vic_row_level_readiness_summary.json`
- `audits/financial_statement_date_repair_rows.csv`
- `audits/financial_statement_date_repair_summary.json`

These cover all 8,341 raw rows.

Uploaded transcript companion files:

- `transcripts/public_transcripts_coverage.jsonl`
- `transcripts/public_transcripts_publication_to_3y.jsonl`
- `transcripts/public_transcripts_summary.json`

These include one coverage record per raw VIC row. Transcript bodies are present for 202 raw rows, with 579 transcript bodies from publication date through publication date + 3 years.
After adding `deerfieldgreen/stk-earnings-transcripts` and `glopardo/sp500-earnings-transcripts`, transcript bodies are present for 866 raw rows, with 9,270 transcript bodies from publication date through publication date + 3 years.

## Transcript Sources Tested

### Yahoo Finance

Result: metadata visible, body unavailable.

- Earnings pages expose transcript metadata and internal S3 paths.
- Same-session embedded transcript URLs returned `404`.
- Embedded S3 transcript JSON URLs returned `403`.

### Financial Modeling Prep

Result: public/demo calls unavailable.

- Tested v3/stable/v4 transcript endpoint patterns.
- Demo/public calls returned `401`.

### Alpha Vantage

Result: viable with a real API key.

- Official `EARNINGS_CALL_TRANSCRIPT` demo call for IBM 2024Q1 returned 37 transcript turns.
- Demo key is limited and does not work for arbitrary symbols.
- No `ALPHAVANTAGE_API_KEY` or `ALPHA_VANTAGE_API_KEY` is currently configured in the environment.

### EODHD

Result: no usable transcript route verified with the currently supplied credential.

- Known EODHD endpoints such as fundamentals and earnings calendar returned `401` with the currently supplied credential.
- Transcript-shaped routes tested for `earning-call-transcript`, `earnings-call-transcript`, and `transcripts` returned `404`.
- Existing local fundamentals data and the row-level financial audits remain valid artifacts from the prior EODHD pull, but this credential cannot currently be used for additional transcript enrichment.

### Public Hugging Face Transcript Datasets

Result: joined and published.

- Source: `Rogersurf/earnings-call-transcripts`
- Additional source: `deerfieldgreen/stk-earnings-transcripts`
- Additional source: `glopardo/sp500-earnings-transcripts`
- License note: source terms are mixed public-HF dataset terms; keep private and review source licenses before redistribution.
- Matched 5,265 transcript source rows across 1,518 VIC-compatible tickers.
- Added 1,788 `deerfieldgreen` source rows across 26 tickers.
- Streamed `glopardo/sp500-earnings-transcripts` through the Dataset Viewer and retained 15,596 source rows across 372 VIC-compatible tickers.
- Attached 9,036 transcript bodies to 843 repaired clean VIC rows.
- For the full raw repo companion files, attached 9,270 transcript bodies to 866 raw VIC rows.

## Remaining Gap

The dataset now has the best verified API-free transcript coverage found so far, but not full transcript coverage for every row. Broader coverage requires either:

1. A real Alpha Vantage API key, then run `enrich_vic_with_alpha_vantage_transcripts.py`.
2. Another licensed transcript corpus with ticker and call-date metadata.
3. A paid transcript provider API key such as Financial Modeling Prep, if available.
4. A working EODHD credential plus documented transcript endpoint, if EODHD provides this product in the active plan.
