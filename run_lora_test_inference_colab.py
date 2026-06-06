# %% [markdown]
# # Evaluate VIC Investment LoRA Adapter On Held-Out Test Set
#
# Run this in Google Colab after `train_unsloth_qwen3_lora_colab.py`.
# It loads the private adapter, generates predictions for the held-out test
# split, and uploads `finetuned_test_predictions.jsonl` back to the model repo.

# %%
!pip install -q --upgrade pip
!pip install -q "unsloth[colab-new]" "datasets" "huggingface_hub" "accelerate" "bitsandbytes"

# %%
import json
import os
import re
from getpass import getpass

if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = getpass("Paste a Hugging Face write token: ")

HF_USERNAME = "YOUR_HF_USERNAME"
TRAINING_PROFILE = "free_t4_qwen3_4b"

MODEL_PROFILES = {
    "free_t4_qwen3_4b": {
        "base_model": "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
        "adapter_repo_name": "vic-investment-qwen3-4b-lora-private",
        "max_seq_length": 4096,
    },
    "strong_l4_or_a100_qwen3_8b": {
        "base_model": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
        "adapter_repo_name": "vic-investment-qwen3-8b-lora-private",
        "max_seq_length": 3072,
    },
}

PROFILE = MODEL_PROFILES[TRAINING_PROFILE]
BASE_MODEL = PROFILE["base_model"]
ADAPTER_REPO_NAME = PROFILE["adapter_repo_name"]
MAX_SEQ_LENGTH = PROFILE["max_seq_length"]

# Keep this as None for the real proof run over the full 835-row held-out test set.
# Use a small integer only for a quick smoke test.
LIMIT = None

# %%
from huggingface_hub import hf_hub_download, login, upload_file, whoami

login(token=os.environ["HF_TOKEN"])
if HF_USERNAME == "YOUR_HF_USERNAME":
    HF_USERNAME = whoami(token=os.environ["HF_TOKEN"])["name"]

DATASET_REPO = f"{HF_USERNAME}/vic-investment-outcomes-sft"
ADAPTER_REPO = f"{HF_USERNAME}/{ADAPTER_REPO_NAME}"
print({"training_profile": TRAINING_PROFILE, "dataset_repo": DATASET_REPO, "adapter_repo": ADAPTER_REPO})

test_path = hf_hub_download(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    filename="investment_test.jsonl",
    token=os.environ["HF_TOKEN"],
)

test_rows = []
with open(test_path, encoding="utf-8") as handle:
    for line in handle:
        test_rows.append(json.loads(line))
if LIMIT:
    test_rows = test_rows[:LIMIT]
len(test_rows)

# %%
from unsloth import FastLanguageModel
import torch

if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected. In Colab, choose Runtime -> Change runtime type -> GPU.")

print(f"GPU: {torch.cuda.get_device_name(0)}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_REPO,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
    token=os.environ["HF_TOKEN"],
)
FastLanguageModel.for_inference(model)

# %%
OUTCOME_RE = re.compile(r"primary training label:\s*\S+\s+outcome\s+is\s+(\w+)", re.I)
VALID = {"excellent", "good", "neutral", "poor", "failed"}


def strip_gold_answer(messages):
    return [messages[0], messages[1]]


def gold_outcome(messages):
    text = messages[-1]["content"]
    match = OUTCOME_RE.search(text)
    value = match.group(1).lower() if match else None
    return value if value in VALID else None


def predict(messages):
    prompt_messages = strip_gold_answer(messages)
    inputs = tokenizer.apply_chat_template(
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
    if predicted not in VALID:
        predicted = None
    return predicted, text


predictions = []
correct = 0
scored = 0
missing = 0
by_direction = {
    "long": {"correct": 0, "scored": 0},
    "short": {"correct": 0, "scored": 0},
}
confusion = {}
for index, row in enumerate(test_rows, start=1):
    predicted, text = predict(row["messages"])
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
    predictions.append(
        {
            "idea_id": row["metadata"]["idea_id"],
            "predicted_outcome": predicted,
            "gold_outcome": truth,
            "assistant": text,
        }
    )
    if index % 10 == 0:
        print(f"predicted {index}/{len(test_rows)}")

predictions[:2]

# %%
pred_path = "finetuned_test_predictions.jsonl"
with open(pred_path, "w", encoding="utf-8") as handle:
    for item in predictions:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")

upload_file(
    path_or_fileobj=pred_path,
    path_in_repo=pred_path,
    repo_id=ADAPTER_REPO,
    repo_type="model",
    token=os.environ["HF_TOKEN"],
)

metrics = {
    "limit": LIMIT,
    "rows": len(test_rows),
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

metrics_path = "finetuned_test_metrics.json"
with open(metrics_path, "w", encoding="utf-8") as handle:
    json.dump(metrics, handle, indent=2, sort_keys=True)

upload_file(
    path_or_fileobj=metrics_path,
    path_in_repo=metrics_path,
    repo_id=ADAPTER_REPO,
    repo_type="model",
    token=os.environ["HF_TOKEN"],
)

print(json.dumps(metrics, indent=2, sort_keys=True))
print(f"Uploaded predictions to https://huggingface.co/{ADAPTER_REPO}/blob/main/{pred_path}")
print(f"Uploaded metrics to https://huggingface.co/{ADAPTER_REPO}/blob/main/{metrics_path}")

# %% [markdown]
# Download `finetuned_test_predictions.jsonl` locally, then run:
#
# ```powershell
# C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe evaluate_outcome_predictions.py finetuned_test_predictions.jsonl --output reports\finetuned_test_metrics.json
# C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe check_finetune_gate.py
# ```
