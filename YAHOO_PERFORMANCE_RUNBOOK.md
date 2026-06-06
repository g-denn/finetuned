# Yahoo Performance Repair Runbook

This repair intentionally leaves `public.performance` untouched. The raw,
split-adjusted Yahoo labels go into `public.performance_yahoo` first so they can
be audited.

Important: `public.performance_yahoo` is raw evidence only. It is not
fine-tuning-ready because ticker changes, delistings, acquisitions, spin-offs,
bankruptcies, and ticker reuse need separate validation. Use
`PERFORMANCE_VALIDATION_RUNBOOK.md` and train only from
`public.performance_training_labels_v1`.

## 1. Apply schema

Run this SQL in Supabase SQL Editor, or with an authenticated CLI:

```powershell
supabase db query --linked -f schema_yahoo_performance.sql
```

The CLI path needs `supabase login` or `SUPABASE_ACCESS_TOKEN`.

## 2. Dry run

```powershell
$env:SUPABASE_SERVICE_ROLE_KEY = "<service-role-key>"
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  fetch_yahoo_performance_adjusted.py --limit 20 --dry-run --sleep 0
```

## 3. Backfill

```powershell
$env:SUPABASE_SERVICE_ROLE_KEY = "<service-role-key>"
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  fetch_yahoo_performance_adjusted.py --apply --only-missing --batch-size 100
```

The script is resumable. Re-run the same command to fill rows that were not
written yet.

## 4. Retry Yahoo failures

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  fetch_yahoo_performance_adjusted.py --apply --refresh-errors --batch-size 100
```

## 5. Audit

```sql
select source_status, count(*)
from public.performance_yahoo
group by source_status
order by count(*) desc;

select *
from public.performance_yahoo_quality
where one_year_relative_diff > 0.5
order by one_year_relative_diff desc
limit 100;
```

## 6. Validate before fine-tuning

```powershell
supabase db query --linked -f supabase\migrations\20260522071842_add_performance_validation_layer.sql

C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  performance_validation.py --seed --limit 50 --dry-run
```

See `PERFORMANCE_VALIDATION_RUNBOOK.md` for the two-agent validation gate.

## Label meaning

`perf_1y = adj_price_1y / base_adj_close`

So:

- `1.0` means unchanged.
- `0.9` means down 10%.
- `3.0` means up 3x.

For short ideas, `short_perf_*` is also stored as the inverse multiplier.
