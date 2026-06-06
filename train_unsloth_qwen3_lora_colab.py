# %% [markdown]
# # VIC Investment Outcome LoRA Fine-Tune
#
# Run this in Google Colab with a free GPU runtime.
#
# Recommended runtime:
# - T4/L4 GPU
# - high-RAM if available
#
# This trains a private LoRA adapter on the processed investment memo/outcome
# dataset. It does not train a full foundation model.

# %%
!pip install -q --upgrade pip
!pip install -q "unsloth[colab-new]" "trl" "datasets" "huggingface_hub" "accelerate" "bitsandbytes"

# %%
import os
from getpass import getpass

if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = getpass("Paste a Hugging Face write token: ")

HF_USERNAME = "YOUR_HF_USERNAME"

# Good free-GPU first pass. Upgrade only after this proves useful.
TRAINING_PROFILE = "free_t4_qwen3_4b"

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

# %%
from huggingface_hub import login, whoami

login(token=os.environ["HF_TOKEN"])
if HF_USERNAME == "YOUR_HF_USERNAME":
    HF_USERNAME = whoami(token=os.environ["HF_TOKEN"])["name"]

DATASET_REPO = f"{HF_USERNAME}/vic-investment-outcomes-sft"
ADAPTER_REPO = f"{HF_USERNAME}/{ADAPTER_REPO_NAME}"
print({
    "training_profile": TRAINING_PROFILE,
    "profile_notes": PROFILE["notes"],
    "dataset_repo": DATASET_REPO,
    "base_model": BASE_MODEL,
    "adapter_repo": ADAPTER_REPO,
})

# %%
from datasets import load_dataset
from huggingface_hub import hf_hub_download

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
dataset

# %%
from unsloth import FastLanguageModel
import torch

if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected. In Colab, choose Runtime -> Change runtime type -> GPU.")

print(f"GPU: {torch.cuda.get_device_name(0)}")

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

# %%
def format_example(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

formatted = dataset.map(format_example, remove_columns=dataset["train"].column_names)
formatted

# %%
from trl import SFTTrainer, SFTConfig

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

# %%
metrics = trainer.evaluate()
metrics

# %%
model.push_to_hub(ADAPTER_REPO, token=os.environ["HF_TOKEN"], private=True)
tokenizer.push_to_hub(ADAPTER_REPO, token=os.environ["HF_TOKEN"], private=True)

from huggingface_hub import upload_file

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

# VIC Investment Qwen3 LoRA Adapter

Private LoRA adapter trained on validated investment research memos and
direction-adjusted future outcome labels.

## Base Model

`{BASE_MODEL}`

## Training Data

Private dataset repo:

`{DATASET_REPO}`

## Evaluation Gate

Before considering the adapter useful, run it on the held-out test set and
compare against the majority-class baseline:

- majority baseline test accuracy: 29.70%
- long baseline accuracy: 30.57%
- short baseline accuracy: 28.75%
- TF-IDF text baseline test accuracy: 30.90%

The adapter should beat these baselines and should be checked separately on long
and short ideas.

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
print(f"Private adapter uploaded to https://huggingface.co/{ADAPTER_REPO}")

# %% [markdown]
# ## After Training
#
# Keep the LoRA adapter private until evaluation is done.
# Next steps:
# 1. Run inference on `investment_test.jsonl`.
# 2. Compare base model vs LoRA adapter on held-out future ideas.
# 3. Decide whether to train a larger model or change the task format.
