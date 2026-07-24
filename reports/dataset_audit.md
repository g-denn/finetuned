# Investment Fine-Tuning Dataset Audit

- Raw validated training rows: 8353
- Canonical rows with memo text: 8341
- SFT rows with 3y return target: 5973
- Missing idea metadata skipped: 0
- Missing descriptions before filtering: 12
- Long ideas: 6187
- Short ideas: 2154

## Time-Based Splits

- train: 4778 rows (2000-02-25 to 2020-09-01)
- val: 597 rows (2020-09-01 to 2021-10-14)
- test: 598 rows (2021-10-14 to 2022-11-03)

## Horizon Coverage

- 1y: 6793 rows
- 3y: 5973 rows
- 5y: 4763 rows
- 10y: 2315 rows
- 20y: 310 rows

## Primary Outcome Counts

- excellent: 823
- failed: 1097
- good: 2044
- neutral: 2630
- poor: 1747

## Outputs

- `C:\Users\Dell\finetuned\data\processed\investment_canonical.jsonl`
- `C:\Users\Dell\finetuned\data\processed\investment_canonical.csv`
- `C:\Users\Dell\finetuned\data\processed\investment_train.jsonl`
- `C:\Users\Dell\finetuned\data\processed\investment_val.jsonl`
- `C:\Users\Dell\finetuned\data\processed\investment_test.jsonl`
