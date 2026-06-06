---
license: other
base_model: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
library_name: peft
tags:
- finance
- investment-research
- lora
- qlora
- unsloth
- private-model
---

# VIC Investment Qwen3 LoRA Adapter

Private LoRA adapter trained on validated investment research memos and
direction-adjusted future outcome labels.

## Base Model

`unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`

This is the recommended first-pass model because it is small enough for free
Colab/Kaggle GPU training while still having useful instruction-following and
long-context behavior.

## Training Data

Private dataset repo:

`YOUR_HF_USERNAME/vic-investment-outcomes-sft`

Dataset rows contain:

- historical investment memo
- catalyst notes
- long/short direction
- company/security metadata
- validated raw performance multipliers
- direction-adjusted outcome labels

## Intended Use

The adapter is intended for research workflow experiments:

- structured historical outcome summaries
- investment memo analysis
- thesis/catalyst/risk extraction
- supervised learning from validated long/short outcomes

The adapter is not a standalone investment adviser and should not be used as
the sole basis for live investment decisions.

## Evaluation Gate

Before considering the adapter useful, run it on the held-out test set and
compare against the majority-class baseline:

- majority baseline test accuracy: 29.70%
- long baseline accuracy: 30.57%
- short baseline accuracy: 28.75%

The adapter should beat this baseline and should be checked separately on long
and short ideas.

## Privacy

Keep this model private unless the data owner explicitly approves publication.
