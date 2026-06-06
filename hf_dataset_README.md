---
license: other
pretty_name: VIC Investment Outcome SFT Dataset
task_categories:
- text-generation
- text-classification
language:
- en
tags:
- finance
- investment-research
- supervised-fine-tuning
- private-dataset
---

# VIC Investment Outcome SFT Dataset

Private supervised fine-tuning dataset built from validated investment research
memos and verified future performance outcomes.

## Contents

- `investment_train.jsonl`
- `investment_val.jsonl`
- `investment_test.jsonl`
- `investment_canonical.jsonl`
- `reports/dataset_audit.md`
- `reports/dataset_audit.json`

## Dataset Construction

The dataset joins:

- validated performance rows from `training_ready_after_sec_yahoo_salvage.csv`
- original VIC investment memo text from `VIC_IDEAS.sql`
- catalyst notes from `VIC_IDEAS.sql`
- idea direction from `VIC_IDEAS.sql`

Rows are included only when they were marked training-ready in the local
validation pipeline and had usable memo text.

## Splits

The split is time-based:

- train: older ideas
- validation: middle period
- test: newest held-out ideas

This avoids random leakage across repeated authors, market regimes, and similar
company cases.

## Labels

Labels include:

- raw stock performance multipliers
- direction-adjusted performance multipliers
- outcome buckets: `excellent`, `good`, `neutral`, `poor`, `failed`

Direction adjustment is essential:

- for long ideas, higher stock performance is better
- for short ideas, lower stock performance is better

## Intended Use

This dataset is intended to fine-tune an investment research assistant that can:

- summarize historical memo outcomes
- learn the user's investment research format
- separate information known at publication from future outcome evidence
- support downstream evaluation of investment-process signals

It is not intended to produce live investment recommendations by itself.

## Privacy

This dataset should remain private unless the data owner explicitly decides
otherwise.
