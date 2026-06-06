# Investment Model Fine-Tuning Runbook

## Current Status

The local validated dataset has been joined to the VIC investment memo text.

Generated files:

- `data/processed/investment_canonical.jsonl`
- `data/processed/investment_canonical.csv`
- `data/processed/investment_train.jsonl`
- `data/processed/investment_val.jsonl`
- `data/processed/investment_test.jsonl`
- `reports/dataset_audit.md`
- `reports/dataset_audit.json`
- `reports/majority_baseline_predictions.jsonl`
- `reports/majority_baseline_metrics.json`
- `reports/text_baseline_metrics.json`
- `hf_dataset_README.md`
- `hf_model_README.md`
- `investment_finetune_all_in_one_colab.py`
- `investment_finetune_all_in_one_colab.ipynb`
- `upload_dataset_to_huggingface_colab.py`
- `upload_dataset_to_huggingface_colab.ipynb`
- `package_hf_dataset_upload_bundle.ps1`
- `run_base_model_test_inference_colab.py`
- `run_base_model_test_inference_colab.ipynb`
- `run_lora_test_inference_colab.py`
- `train_unsloth_qwen3_lora_colab.ipynb`
- `run_lora_test_inference_colab.ipynb`
- `make_colab_notebooks.py`
- `prepare_finetune_local.ps1`
- `fetch_hf_adapter_results.py`
- `check_hf_finetune_status.py`
- `check_finetune_gate.py`
- `train_text_baseline.py`
- `dist/investment_finetune_handoff.zip`
- `dist/hf_dataset_upload_bundle.zip`

Dataset audit:

- Raw validated rows: 8,353
- Canonical rows with memo text: 8,341
- Train: 6,672 rows, dated 2000-02-10 to 2020-05-11
- Validation: 834 rows, dated 2020-05-11 to 2021-08-06
- Test: 835 rows, dated 2021-08-07 to 2022-11-03
- Long ideas: 3,655
- Short ideas: 4,686

The split is time-based so the test set behaves like future unseen data.

Current baseline:

- Majority-class baseline: predicts `neutral` for every test row.
- Held-out test accuracy: 29.70%.
- Long accuracy: 30.57%.
- Short accuracy: 28.75%.
- TF-IDF + logistic regression text baseline test accuracy: 30.90%.
- Text baseline long accuracy: 31.72%.
- Text baseline short accuracy: 30.00%.

The fine-tuned model should beat both baselines on the held-out test set before
it is considered useful.

## What The Fine-Tune Can And Cannot Prove

Fine-tuning teaches the model your investment-process format and historical
memo/outcome patterns. It does not, by itself, prove the strategy works.

Proof requires held-out evaluation:

- Train only on older ideas.
- Tune/check against validation ideas.
- Report final results only on the newest test ideas.
- Compare base model vs fine-tuned model.
- Check long and short ideas separately.
- Check whether higher model confidence/ranking maps to better
  direction-adjusted returns.

## Why Outcomes Are Included

Each training row includes:

- Information known at publication time:
  - memo text
  - catalyst text
  - long/short direction
  - company/security metadata
- Validated future outcome labels:
  - raw stock multipliers
  - direction-adjusted multipliers
  - outcome buckets

Direction adjustment is essential:

- Long: stock up is good.
- Short: stock down is good.

For shorts, direction-adjusted multiplier is `1 / raw_stock_multiplier`.

## Recommended Model

First pass:

`unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`

Reason:

- Small enough for a free Colab GPU.
- Strong instruction-following base.
- Long-context family, useful for investment memos.
- LoRA adapter training is realistic on free hardware.

Do not start with a 14B/27B model. Prove the dataset/task works first.

Upgrade path if the first pass works:

1. Qwen3 4B LoRA: cheapest/free proof of signal.
2. Qwen3 8B LoRA: better quality if Colab/Kaggle GPU allows it.
3. 14B+ LoRA: only if paid GPU or stronger free allocation is available.

The Colab notebooks expose:

```python
TRAINING_PROFILE = "free_t4_qwen3_4b"
```

Keep that default for the first proof run. If the 4B result beats the gate and
you get an L4/A100-class runtime, switch all training/evaluation notebooks to:

```python
TRAINING_PROFILE = "strong_l4_or_a100_qwen3_8b"
```

The base-model and LoRA evaluation notebooks must use the same profile as the
training notebook so they read/write the matching private adapter repo.

Do not do full fine-tuning unless you have paid GPU infrastructure. LoRA/QLoRA
is the right format for this dataset and budget.

## Free Training Path

Hugging Face AutoTrain is convenient but not the free route, because AutoTrain
charges by hardware minute and requires a payment method.

The free route is:

1. Build dataset locally.
2. Upload dataset to a private Hugging Face dataset repo.
3. Open Google Colab free GPU.
4. Run `train_unsloth_qwen3_lora_colab.ipynb`.
5. Push a private LoRA adapter to Hugging Face.

The lowest-friction version is the all-in-one notebook:

`investment_finetune_all_in_one_colab.ipynb`

It can upload the dataset bundle, train the selected profile, run base-model
and fine-tuned evaluation, upload metrics, and print the final gate result.

Relevant docs:

- Hugging Face AutoTrain cost: https://huggingface.co/docs/autotrain/v0.6.48/cost
- Hugging Face dataset upload: https://huggingface.co/docs/datasets/v2.7.0/en/upload_dataset
- Hugging Face private repos: https://huggingface.co/docs/huggingface_hub/en/guides/repository
- Unsloth docs: https://unsloth.ai/docs
- Qwen3-4B Unsloth model: https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
- Qwen3-8B Unsloth model: https://huggingface.co/unsloth/Qwen3-8B-unsloth-bnb-4bit

## Step 1: Rebuild Dataset

Run:

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe build_investment_finetune_dataset.py
```

Check:

```powershell
Get-Content reports\dataset_audit.md
```

Or run the full local preflight:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\Users\Dell\finetuned\prepare_finetune_local.ps1
```

This rebuilds the dataset, validates JSONL splits, regenerates the baseline,
trains/scores the TF-IDF text baseline, and checks whether Hugging Face auth is
available.

## Step 2: Create A Hugging Face Token

Create a token in Hugging Face with write access.

Do not paste permanent tokens into source files. Prefer setting it only for the
current terminal session:

```powershell
$env:HF_TOKEN = "hf_your_write_token"
```

If a token has already been pasted into chat or logs, revoke it and create a new
one before uploading private data.

Alternative: use a secure prompt instead of setting an environment variable:

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe upload_dataset_to_huggingface.py --prompt-token
```

## Step 3: Install Upload Dependency

In a normal Python environment:

```powershell
pip install huggingface_hub
```

Or with the bundled Python:

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m pip install huggingface_hub
```

## Step 4: Upload Dataset Privately

There are two working upload paths:

- Local upload: fastest if the terminal can use `HF_TOKEN`.
- Colab upload: best if the Codex/Hugging Face OAuth app launch is broken.

### Option A: Local Upload

First verify the upload manifest without contacting Hugging Face:

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe upload_dataset_to_huggingface.py --dry-run
```

Current dry-run manifest:

- files: 13
- total size: about 210 MB
- includes train/val/test JSONL, canonical JSONL, dataset card, runbook, and
  baseline reports

The script can infer your Hugging Face username from `HF_TOKEN`:

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe upload_dataset_to_huggingface.py
```

Or pass an explicit repo:

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe upload_dataset_to_huggingface.py --repo-id YOUR_USERNAME/vic-investment-outcomes-sft
```

The script creates a private dataset repo and uploads:

- train split
- validation split
- test split
- canonical dataset
- audit report
- dataset card
- majority-class baseline metrics
- TF-IDF text baseline metrics

### Option B: Colab Upload

If local Hugging Face login or OAuth is broken, package the upload bundle:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\Users\Dell\finetuned\package_hf_dataset_upload_bundle.ps1
```

Output:

`dist/hf_dataset_upload_bundle.zip`

Then open Google Colab, upload these two files into the Colab file browser:

- `upload_dataset_to_huggingface_colab.ipynb`
- `dist/hf_dataset_upload_bundle.zip`

Run all cells in `upload_dataset_to_huggingface_colab.ipynb`.

That notebook asks for the Hugging Face write token with `getpass`, creates a
private dataset repo, and uploads the same 13 files as the local uploader.

## Step 4B: All-In-One Colab Option

Instead of running the upload, train, and evaluation notebooks separately, you
can run:

`investment_finetune_all_in_one_colab.ipynb`

Upload these two files into the Colab file browser:

- `investment_finetune_all_in_one_colab.ipynb`
- `dist/hf_dataset_upload_bundle.zip`

Then run all cells. Leave:

```python
DO_UPLOAD_DATASET = True
DO_TRAIN = True
DO_EVALUATE_BASE = True
DO_EVALUATE_LORA = True
EVAL_LIMIT = None
```

For a quick smoke test only, set `EVAL_LIMIT = 20`; return it to `None` for the
real proof run. The all-in-one notebook writes the same private dataset/model
repos as the separate notebook path.

## Step 5: Train In Colab

Open Google Colab:

1. Runtime -> Change runtime type -> GPU.
2. Upload `train_unsloth_qwen3_lora_colab.ipynb`.
3. Leave `HF_USERNAME = "YOUR_HF_USERNAME"` to infer it from the token, or set
   it manually.
4. Confirm `DATASET_REPO`.
5. Run all cells.

The script trains a private LoRA adapter and pushes it to:

`YOUR_USERNAME/vic-investment-qwen3-4b-lora-private`

## Step 6: Run Held-Out Inference In Colab

After training, run:

`run_lora_test_inference_colab.ipynb`

Leave the placeholder alone to infer the username, or set it manually:

```python
HF_USERNAME = "YOUR_HF_USERNAME"
```

The script loads the private adapter, predicts held-out test outcomes, computes
accuracy, and uploads:

- `finetuned_test_predictions.jsonl`
- `finetuned_test_metrics.json`

It evaluates the full 835-row held-out test set by default. For a smoke test
only, temporarily set:

```python
LIMIT = 20
```

Then return `LIMIT` to `None` before recording real metrics.

For a direct base-model comparison, also run:

`run_base_model_test_inference_colab.ipynb`

That uploads:

- `base_model_test_predictions.jsonl`
- `base_model_test_metrics.json`

To fetch those artifacts back locally after Colab uploads them:

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe fetch_hf_adapter_results.py
```

If you trained the 8B upgrade profile:

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe fetch_hf_adapter_results.py --profile strong_l4_or_a100_qwen3_8b
```

Or use a secure token prompt:

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe fetch_hf_adapter_results.py --prompt-token
```

That downloads the model repo predictions/metrics and re-scores the predictions
locally with `evaluate_outcome_predictions.py`.

Then run the local pass/fail gate:

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe check_finetune_gate.py
```

The gate requires the fine-tuned model to score all 835 test rows and beat the
best available comparison: majority baseline, TF-IDF text baseline, and base
model if its metrics were fetched.

To check Hugging Face status without downloading the full prediction files:

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe check_hf_finetune_status.py --prompt-token
```

For the 8B upgrade profile:

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe check_hf_finetune_status.py --profile strong_l4_or_a100_qwen3_8b --prompt-token
```

## Regenerate Colab Notebooks

The notebooks are generated from the `.py` scripts. If you edit either Colab
script, regenerate notebooks with:

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe make_colab_notebooks.py
```

## Create A Handoff Zip

To package the runbook, HF cards, upload script, notebooks, and baseline reports:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\Users\Dell\finetuned\package_finetune_handoff.ps1
```

Output:

`dist/investment_finetune_handoff.zip`

The zip intentionally excludes the large local dataset files; the uploader reads
those directly from `data/processed`.

To package the large dataset files for Colab upload:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\Users\Dell\finetuned\package_hf_dataset_upload_bundle.ps1
```

## Step 7: Evaluate Before Trusting It

Required evaluation:

- Base model output on test set.
- Fine-tuned model output on test set.
- Compare exact label formatting and outcome accuracy.
- Separately evaluate:
  - longs
  - shorts
  - 1Y/3Y/5Y horizons
  - excellent/good vs poor/failed buckets

The model should not be considered useful until it improves held-out test
performance or produces materially better structured investment analysis.

Evaluator:

```powershell
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe evaluate_outcome_predictions.py predictions.jsonl --output reports\finetuned_test_metrics.json
C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe check_finetune_gate.py
```

Prediction file format:

```json
{"idea_id": "example-id", "predicted_outcome": "good"}
```

Accepted outcomes:

```text
excellent, good, neutral, poor, failed
```

## What Is Still Needed

To actually train/upload:

- A valid Hugging Face write token set as `HF_TOKEN`.
- A private Hugging Face dataset repo; the upload script can create it.
- Google Colab access for free GPU training.
- If using the Colab upload path, upload `dist/hf_dataset_upload_bundle.zip`
  into Colab before running `upload_dataset_to_huggingface_colab.ipynb`.
- In Colab, either leave `HF_USERNAME = "YOUR_HF_USERNAME"` so the script
  infers it from the token, or set it manually.

Current local status:

- `huggingface_hub` is installed in the bundled Python.
- `HF_TOKEN` is not currently set in this local shell.
- No upload has been attempted from this machine without a token.

Optional but recommended later:

- A baseline LightGBM/XGBoost ranking model.
- A deployment target for the private LoRA adapter.
