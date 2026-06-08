# %% [markdown]
# # VIC Investment Fine-Tune: Upload, Train, Evaluate
#
# Run this in Google Colab with a GPU runtime.
#
# Upload `dist/hf_dataset_upload_bundle.zip` into the Colab file browser first
# if you want this notebook to create/update the private Hugging Face dataset
# repo. If the dataset repo is already uploaded, set `DO_UPLOAD_DATASET = False`.

# %%
!pip install -q --upgrade pip
!pip install -q "unsloth[colab-new]" "trl" "datasets" "huggingface_hub" "accelerate" "bitsandbytes"

# %%
import gc
import json
import os
import re
import shutil
import zipfile
from getpass import getpass
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo, hf_hub_download, login, upload_file, whoami
from trl import SFTConfig, SFTTrainer

if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = getpass("Paste a Hugging Face write token: ")

login(token=os.environ["HF_TOKEN"])

# %%
HF_USERNAME = "Gden"
TRAINING_PROFILE = "free_t4_qwen3_4b"

DO_UPLOAD_DATASET = False
DO_TRAIN = True
DO_EVALUATE_BASE = True
DO_EVALUATE_LORA = True

# Use 20 for the first safety smoke test. Set this to None for the real proof
# run over all 835 held-out test rows.
EVAL_LIMIT = 20

if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected. In Colab, choose Runtime -> Change runtime type -> GPU.")

print(f"GPU: {torch.cuda.get_device_name(0)}")

BUNDLE_ZIP = Path("/content/hf_dataset_upload_bundle.zip")
EXTRACT_DIR = Path("/content/hf_dataset_upload_bundle")
DATASET_REPO_NAME = "vic-investment-outcomes-sft"

MODEL_PROFILES = {
    "free_t4_qwen3_4b": {
        "base_model": "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
        "adapter_repo_name": "vic-investment-qwen3-4b-lora-private",
        "max_seq_length": 4096,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "num_train_epochs": 1,
        "notes": "Default zero-cost Colab T4/L4 proof run.",
    },
    "strong_l4_or_a100_qwen3_8b": {
        "base_model": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
        "adapter_repo_name": "vic-investment-qwen3-8b-lora-private",
        "max_seq_length": 3072,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1.5e-4,
        "num_train_epochs": 1,
        "notes": "Upgrade profile for stronger free allocations after the 4B run passes.",
    },
}

PROFILE = MODEL_PROFILES[TRAINING_PROFILE]
BASE_MODEL = PROFILE["base_model"]
ADAPTER_REPO_NAME = PROFILE["adapter_repo_name"]
MAX_SEQ_LENGTH = PROFILE["max_seq_length"]
GRADIENT_ACCUMULATION_STEPS = PROFILE["gradient_accumulation_steps"]
LEARNING_RATE = PROFILE["learning_rate"]
NUM_TRAIN_EPOCHS = PROFILE["num_train_epochs"]

if HF_USERNAME == "YOUR_HF_USERNAME":
    HF_USERNAME = whoami(token=os.environ["HF_TOKEN"])["name"]

DATASET_REPO = f"{HF_USERNAME}/{DATASET_REPO_NAME}"
ADAPTER_REPO = f"{HF_USERNAME}/{ADAPTER_REPO_NAME}"

print(json.dumps({
    "training_profile": TRAINING_PROFILE,
    "profile_notes": PROFILE["notes"],
    "dataset_repo": DATASET_REPO,
    "base_model": BASE_MODEL,
    "adapter_repo": ADAPTER_REPO,
    "eval_limit": EVAL_LIMIT,
}, indent=2))

# %% [markdown]
# ## Upload Private Dataset Repo

# %%
def extract_bundle():
    if not BUNDLE_ZIP.exists():
        raise FileNotFoundError(
            "Upload dist/hf_dataset_upload_bundle.zip into Colab first, "
            "or set DO_UPLOAD_DATASET = False if the dataset is already on Hugging Face."
        )
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BUNDLE_ZIP) as archive:
        archive.extractall(EXTRACT_DIR)
    return sorted(str(path.relative_to(EXTRACT_DIR)) for path in EXTRACT_DIR.rglob("*") if path.is_file())


def upload_dataset_bundle():
    files = extract_bundle()
    print(f"extracted {len(files)} files")
    create_repo(DATASET_REPO, repo_type="dataset", private=True, token=os.environ["HF_TOKEN"], exist_ok=True)
    uploads = [
        ("data/processed/investment_train.jsonl", "investment_train.jsonl"),
        ("data/processed/investment_val.jsonl", "investment_val.jsonl"),
        ("data/processed/investment_test.jsonl", "investment_test.jsonl"),
        ("data/processed/investment_canonical.jsonl", "investment_canonical.jsonl"),
        ("hf_dataset_README.md", "README.md"),
        ("FINETUNING_RUNBOOK.md", "FINETUNING_RUNBOOK.md"),
        ("reports/dataset_audit.md", "reports/dataset_audit.md"),
        ("reports/dataset_audit.json", "reports/dataset_audit.json"),
        ("reports/majority_baseline_metrics.json", "reports/majority_baseline_metrics.json"),
        ("reports/majority_baseline_predictions.jsonl", "reports/majority_baseline_predictions.jsonl"),
        ("reports/text_baseline_metrics.json", "reports/text_baseline_metrics.json"),
        ("reports/text_baseline_test_predictions.jsonl", "reports/text_baseline_test_predictions.jsonl"),
        ("reports/text_baseline_val_predictions.jsonl", "reports/text_baseline_val_predictions.jsonl"),
    ]
    for local_name, repo_name in uploads:
        local_path = EXTRACT_DIR / local_name
        if not local_path.exists():
            raise FileNotFoundError(local_path)
        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_name,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            token=os.environ["HF_TOKEN"],
        )
        print(f"uploaded dataset file: {repo_name}")
    info = HfApi(token=os.environ["HF_TOKEN"]).dataset_info(DATASET_REPO)
    print(f"private dataset ready: https://huggingface.co/datasets/{info.id}")


if DO_UPLOAD_DATASET:
    upload_dataset_bundle()
else:
    print("skipping dataset upload")

# %% [markdown]
# ## Load Dataset And Baselines

# %%
data_files = {
    "train": hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename="investment_train.jsonl",
        token=os.environ["HF_TOKEN"],
    ),
    "validation": hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename="investment_val.jsonl",
        token=os.environ["HF_TOKEN"],
    ),
    "test": hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename="investment_test.jsonl",
        token=os.environ["HF_TOKEN"],
    ),
}

dataset = load_dataset("json", data_files=data_files)
test_rows = [row for row in dataset["test"]]
if EVAL_LIMIT:
    test_rows = test_rows[:EVAL_LIMIT]

majority_baseline_path = hf_hub_download(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    filename="reports/majority_baseline_metrics.json",
    token=os.environ["HF_TOKEN"],
)
text_baseline_path = hf_hub_download(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    filename="reports/text_baseline_metrics.json",
    token=os.environ["HF_TOKEN"],
)

majority_baseline = json.loads(Path(majority_baseline_path).read_text())
text_baseline = json.loads(Path(text_baseline_path).read_text())["test"]
create_repo(ADAPTER_REPO, repo_type="model", private=True, token=os.environ["HF_TOKEN"], exist_ok=True)
print({
    "train_rows": len(dataset["train"]),
    "validation_rows": len(dataset["validation"]),
    "test_rows_for_eval": len(test_rows),
    "majority_accuracy": majority_baseline["accuracy"],
    "text_baseline_accuracy": text_baseline["accuracy"],
})

# %% [markdown]
# ## Shared Evaluation Helpers

# %%
OUTCOME_RE = re.compile(r"primary training label:\s*\S+\s+outcome\s+is\s+(\w+)", re.I)
VALID_OUTCOMES = {"excellent", "good", "neutral", "poor", "failed"}


def strip_gold_answer(messages):
    return [messages[0], messages[1]]


def gold_outcome(messages):
    text = messages[-1]["content"]
    match = OUTCOME_RE.search(text)
    value = match.group(1).lower() if match else None
    return value if value in VALID_OUTCOMES else None


def apply_chat_template_safe(tokenizer, messages, **kwargs):
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def predict_one(model, tokenizer, messages):
    prompt_messages = strip_gold_answer(messages)
    inputs = apply_chat_template_safe(
        tokenizer,
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=220,
        temperature=0.1,
        top_p=0.9,
        do_sample=False,
        use_cache=True,
    )
    text = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    match = OUTCOME_RE.search(text)
    predicted = match.group(1).lower() if match else None
    if predicted not in VALID_OUTCOMES:
        predicted = None
    return predicted, text


def evaluate_loaded_model(model, tokenizer, rows, artifact_prefix):
    FastLanguageModel.for_inference(model)
    predictions = []
    correct = 0
    scored = 0
    missing = 0
    by_direction = {
        "long": {"correct": 0, "scored": 0},
        "short": {"correct": 0, "scored": 0},
    }
    confusion = {}
    for index, row in enumerate(rows, start=1):
        predicted, text = predict_one(model, tokenizer, row["messages"])
        truth = gold_outcome(row["messages"])
        user_text = row["messages"][1]["content"]
        direction = "short" if "\n\nDirection: SHORT\n\n" in user_text else "long"
        if predicted is None or truth is None:
            missing += 1
        else:
            scored += 1
            correct += int(predicted == truth)
            by_direction[direction]["scored"] += 1
            by_direction[direction]["correct"] += int(predicted == truth)
            confusion.setdefault(truth, {})
            confusion[truth][predicted] = confusion[truth].get(predicted, 0) + 1
        predictions.append({
            "idea_id": row["metadata"]["idea_id"],
            "predicted_outcome": predicted,
            "gold_outcome": truth,
            "assistant": text,
        })
        if index % 10 == 0:
            print(f"{artifact_prefix}: predicted {index}/{len(rows)}")
    metrics = {
        "limit": EVAL_LIMIT,
        "rows": len(rows),
        "scored": scored,
        "correct": correct,
        "accuracy": correct / scored if scored else 0.0,
        "missing": missing,
        "by_direction": {
            key: {
                "scored": value["scored"],
                "correct": value["correct"],
                "accuracy": value["correct"] / value["scored"] if value["scored"] else 0.0,
            }
            for key, value in by_direction.items()
        },
        "confusion": confusion,
    }
    pred_path = f"{artifact_prefix}_test_predictions.jsonl"
    metrics_path = f"{artifact_prefix}_test_metrics.json"
    with open(pred_path, "w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    for path in (pred_path, metrics_path):
        upload_file(
            path_or_fileobj=path,
            path_in_repo=path,
            repo_id=ADAPTER_REPO,
            repo_type="model",
            token=os.environ["HF_TOKEN"],
        )
        print(f"uploaded model artifact: {path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %% [markdown]
# ## Optional Base Model Evaluation

# %%
base_metrics = None
if DO_EVALUATE_BASE:
    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    base_metrics = evaluate_loaded_model(base_model, base_tokenizer, test_rows, "base_model")
    del base_model
    del base_tokenizer
    clear_gpu()
else:
    print("skipping base model evaluation")

# %% [markdown]
# ## Train Private LoRA Adapter

# %%
model = None
tokenizer = None
trainer_metrics = None

if DO_TRAIN:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    def format_example(example):
        return {
            "text": apply_chat_template_safe(
                tokenizer,
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    formatted = dataset.map(format_example, remove_columns=dataset["train"].column_names)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted["train"],
        eval_dataset=formatted["validation"],
        args=SFTConfig(
            output_dir="vic-investment-qwen3-lora",
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            packing=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=20,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=250,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
        ),
    )
    trainer.train()
    trainer_metrics = trainer.evaluate()
    print(trainer_metrics)
    model.push_to_hub(ADAPTER_REPO, token=os.environ["HF_TOKEN"], private=True)
    tokenizer.push_to_hub(ADAPTER_REPO, token=os.environ["HF_TOKEN"], private=True)
    print(f"private adapter uploaded: https://huggingface.co/{ADAPTER_REPO}")
else:
    print("skipping training; loading existing adapter for LoRA evaluation")

# %% [markdown]
# ## Evaluate Fine-Tuned LoRA

# %%
lora_metrics = None
if DO_EVALUATE_LORA:
    if model is None or tokenizer is None:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=ADAPTER_REPO,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
            token=os.environ["HF_TOKEN"],
        )
    lora_metrics = evaluate_loaded_model(model, tokenizer, test_rows, "finetuned")
else:
    print("skipping LoRA evaluation")

# %% [markdown]
# ## Gate And Model Card

# %%
gate = None
if lora_metrics:
    comparisons = {
        "majority": majority_baseline["accuracy"],
        "tfidf_text": text_baseline["accuracy"],
    }
    if base_metrics:
        comparisons["base_model"] = base_metrics["accuracy"]
    best_name, best_accuracy = max(comparisons.items(), key=lambda item: item[1])
    gate = {
        "training_profile": TRAINING_PROFILE,
        "expected_test_rows": 835,
        "finetuned_accuracy": lora_metrics["accuracy"],
        "finetuned_scored": lora_metrics["scored"],
        "finetuned_long_accuracy": lora_metrics["by_direction"]["long"]["accuracy"],
        "finetuned_short_accuracy": lora_metrics["by_direction"]["short"]["accuracy"],
        "best_comparison": best_name,
        "best_comparison_accuracy": best_accuracy,
        "beats_best_comparison": lora_metrics["accuracy"] > best_accuracy,
        "full_test_set": lora_metrics["scored"] == 835,
        "pass": lora_metrics["scored"] == 835 and lora_metrics["accuracy"] > best_accuracy,
    }
    with open("finetune_gate.json", "w", encoding="utf-8") as handle:
        json.dump(gate, handle, indent=2, sort_keys=True)
    upload_file(
        path_or_fileobj="finetune_gate.json",
        path_in_repo="finetune_gate.json",
        repo_id=ADAPTER_REPO,
        repo_type="model",
        token=os.environ["HF_TOKEN"],
    )

model_card = f"""---
license: other
base_model: {BASE_MODEL}
library_name: peft
tags:
- finance
- investment-research
- lora
- qlora
- unsloth
- private-model
---

# VIC Investment LoRA Adapter

Private LoRA adapter trained on validated investment research memos and
direction-adjusted future outcome labels.

## Training Profile

`{TRAINING_PROFILE}`

## Base Model

`{BASE_MODEL}`

## Training Data

Private dataset repo:

`{DATASET_REPO}`

## Evaluation Gate

The model is useful only if it scores the full held-out test set and beats the
best available comparison baseline.

```json
{json.dumps(gate, indent=2, sort_keys=True) if gate else "null"}
```

## Privacy

Keep this model private unless the data owner explicitly approves publication.
"""

with open("README.md", "w", encoding="utf-8") as handle:
    handle.write(model_card)

upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=ADAPTER_REPO,
    repo_type="model",
    token=os.environ["HF_TOKEN"],
)

print(json.dumps({
    "dataset_repo": f"https://huggingface.co/datasets/{DATASET_REPO}",
    "adapter_repo": f"https://huggingface.co/{ADAPTER_REPO}",
    "gate": gate,
}, indent=2, sort_keys=True))
