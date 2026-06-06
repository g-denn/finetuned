# Performance Validation Runbook

This layer makes `public.performance_yahoo` raw evidence only. Fine-tuning must
read from `public.performance_training_labels_v1`, which exposes only rows that
passed identity, corporate-action, and adversarial validation gates.

## 1. Apply schema

```powershell
supabase db query --linked -f supabase\migrations\20260522071842_add_performance_validation_layer.sql
```

## 2. Seed validation rows

This copies high-priority raw Yahoo rows into `public.performance_validation` as
`unreviewed`. It does not mark anything as verified.

```powershell
$env:SUPABASE_SERVICE_ROLE_KEY = "<service-role-key>"
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  performance_validation.py --seed --limit 50
```

Use `--dry-run` first to inspect the rows:

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  performance_validation.py --seed --limit 10 --dry-run
```

## 3. Claim rows for agent review

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  performance_validation.py --claim --agent-id agent-a-001 --limit 5
```

Agent A researches the claimed row and writes a structured proposal into
`agent_a_result`. Agent B independently verifies or rejects and writes
`agent_b_result`. A row may enter training only if the database gate allows
`include_in_training = true`.

## 4. Agent A required output

```json
{
  "identity_resolution": {
    "resolved_issuer_name": null,
    "resolved_security_name": null,
    "cik": null,
    "figi": null,
    "cusip": null,
    "isin": null,
    "ticker_start": null,
    "ticker_end": null,
    "exchange_start": null,
    "exchange_end": null,
    "identity_confidence": 0.0
  },
  "corporate_action_timeline": [],
  "return_resolution": {
    "method": "adjusted_close",
    "corrected_start_value": null,
    "corrected_end_value": null,
    "final_return": null,
    "return_confidence": 0.0
  },
  "decision": {
    "status": "needs_manual_review",
    "include_in_training": false,
    "reason": ""
  },
  "sources": []
}
```

Agent A cannot mark a row verified from Yahoo alone when there is a ticker
change, acquisition, delisting, bankruptcy, spin-off, extreme return, suspicious
raw/adjusted close jump, or weak identity evidence.

## 5. Agent B required output

```json
{
  "reviewer_status": "pass",
  "reason_code": "NONE",
  "calculation_reproduced": true,
  "identity_passed": true,
  "source_quality_passed": true,
  "matching_label": true,
  "objections": [],
  "required_changes": []
}
```

Agent B must reject or mark unresolved if ticker identity, trading dates,
adjusted price treatment, split handling, dividend handling, merger endpoint,
delisting endpoint, or source quality cannot be reproduced independently.

## 6. Training query

```sql
select *
from public.performance_training_labels_v1;
```

Do not train from `public.performance_yahoo` directly.

## 7. EODHD delisted archive

When running the EODHD backfill, include the delisted archive lookup. The script
uses EODHD `exchange-symbol-list/{EXCHANGE}?delisted=1`, caches one file per
exchange, and matches rows by `CODE.EXCHANGE` before calculating local returns.
Matched rows are still manual-review candidates until acquisition, bankruptcy,
cash consideration, OTC transfer, or equity cancellation is explicitly modeled.

```powershell
$env:EODHD_API_TOKEN = "<token>"
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  eodhd_backfill.py `
  --ideas-file eodhd_output\all_ideas.json `
  --output-dir eodhd_output\full_run `
  --include-master-delisted
```

Use `--delisted-exchanges US HK LSE XETRA` to limit the archive pull during a
test run. Without that flag, the script uses every exchange suffix present in
the input ideas.

## 8. EODHD fundamentals archive

For business-reality review, pull full EODHD Fundamentals v1.1 payloads once per
unique `SYMBOL.EXCHANGE`. This stores raw JSON in `fundamentals_cache`, plus
`fundamentals_summary.csv/json` with review-ready fields like instrument type,
delisted status, sector, revenue, profit, market cap, and financial report
coverage.

EODHD's skill docs recommend `bulk-fundamentals` for broad exchange-level pulls.
Use bulk first when the account has the Extended Fundamentals plan:

```powershell
$env:EODHD_API_TOKEN = "<token>"
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  eodhd_backfill.py `
  --ideas-file eodhd_output\all_ideas.json `
  --output-dir eodhd_output\full_run `
  --include-master-delisted `
  --include-fundamentals `
  --fundamentals-mode bulk
```

If EODHD returns `403 Forbidden` for bulk fundamentals, fall back to the
single-symbol endpoint without rerunning price validation:

```powershell
$env:EODHD_API_TOKEN = "<token>"
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  eodhd_backfill.py `
  --ideas-file eodhd_output\all_ideas.json `
  --output-dir eodhd_output\full_run `
  --include-fundamentals `
  --fundamentals-mode single `
  --retry-fundamentals-errors `
  --fundamentals-only `
  --concurrency 16
```

The manual-review verifier reads those cached fundamentals automatically:

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  manual_review_validation.py `
  --results-csv eodhd_output\full_run\validation_results.csv `
  --cache-dir eodhd_output\full_run\symbol_cache `
  --fundamentals-cache-dir eodhd_output\full_run\fundamentals_cache `
  --limit 50
```

Fundamentals evidence does not by itself promote an extreme return to training.
It supplies the revenue/profit/market-cap/instrument evidence Agent C needs to
write a sourced pass/reject/manual-review decision.

## 9. Review stages and queue

`validation_results.csv` deliberately separates price math from training
readiness:

- `math_validation_status`: `math_reproduced`, `math_incomplete`, or
  `provider_error`.
- `review_stage`: `math_reproduced_low_risk`, `provider_warning`,
  `math_incomplete`, or `provider_error` before manual review.
- `training_readiness`: `candidate_low_risk`, `manual_review_required`, or
  `not_training_ready`.

Rows with delisted archive matches, fundamentals delisted flags, early-ended
price histories, reverse splits, non-common instruments, missing common-stock
financials, or extreme returns are not training-ready just because adjusted-close
math reproduced. They are exported to:

```text
eodhd_output\full_run\manual_review_queue.csv
```

The queue is sorted by:

1. Extreme winners above `15x`
2. Severe losers below `0.05x`
3. Delisted / early-ended histories / missing primary endpoint prices
4. Reverse splits
5. Ticker-lineage overrides
6. Non-common instruments
7. Missing common-stock financials
8. Provider errors

Use this queue as the row-by-row worklist for Agent A/B/C. Only rows that later
receive sourced identity, corporate-action, cross-provider, and business-reality
approval should be promoted to `training_ready`.

## 10. Automation cadence

- Every 30 minutes: process 5-10 highest-risk unreviewed rows.
- Daily: retry provider errors and stale `in_progress` rows.
- Weekly: re-audit high-risk verified rows.

Never overwrite verified rows unless the run is explicitly configured to refresh
verified labels.

## 11. Local tests

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  -m unittest test_eodhd_backfill.py test_manual_review_validation.py test_performance_validation.py
```
