# Investment Fine-Tuning Dataset Audit

- Raw validated training rows: 8353
- Canonical rows with memo text: 8341
- Missing idea metadata skipped: 0
- Missing descriptions before filtering: 12
- Long ideas: 3655
- Short ideas: 4686

## Time-Based Splits

- train: 6672 rows (2000-02-10 to 2020-05-11)
- val: 834 rows (2020-05-11 to 2021-08-06)
- test: 835 rows (2021-08-07 to 2022-11-03)

## Horizon Coverage

- 1y: 6793 rows
- 3y: 5973 rows
- 5y: 4763 rows
- 10y: 2315 rows
- 20y: 310 rows

## Primary Outcome Counts

- excellent: 845
- failed: 1121
- good: 1651
- neutral: 2475
- poor: 2249

## Outputs

- `C:\Users\Dell\finetuned\data\processed\investment_canonical.jsonl`
- `C:\Users\Dell\finetuned\data\processed\investment_canonical.csv`
- `C:\Users\Dell\finetuned\data\processed\investment_train.jsonl`
- `C:\Users\Dell\finetuned\data\processed\investment_val.jsonl`
- `C:\Users\Dell\finetuned\data\processed\investment_test.jsonl`
