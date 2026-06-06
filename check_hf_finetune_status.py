#!/usr/bin/env python3
"""Check Hugging Face upload/training status for the investment fine-tune."""

from __future__ import annotations

import argparse
import json
import os
from getpass import getpass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
PROFILE_REPO_NAMES = {
    "free_t4_qwen3_4b": "vic-investment-qwen3-4b-lora-private",
    "strong_l4_or_a100_qwen3_8b": "vic-investment-qwen3-8b-lora-private",
}
DEFAULT_PROFILE = "free_t4_qwen3_4b"
DEFAULT_DATASET_REPO_NAME = "vic-investment-outcomes-sft"
EXPECTED_DATASET_FILES = [
    "investment_train.jsonl",
    "investment_val.jsonl",
    "investment_test.jsonl",
    "investment_canonical.jsonl",
    "README.md",
    "FINETUNING_RUNBOOK.md",
    "reports/dataset_audit.md",
    "reports/dataset_audit.json",
    "reports/majority_baseline_metrics.json",
    "reports/majority_baseline_predictions.jsonl",
    "reports/text_baseline_metrics.json",
    "reports/text_baseline_test_predictions.jsonl",
    "reports/text_baseline_val_predictions.jsonl",
]
EXPECTED_MODEL_FILES = [
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "base_model_test_predictions.jsonl",
    "base_model_test_metrics.json",
    "finetuned_test_predictions.jsonl",
    "finetuned_test_metrics.json",
    "finetune_gate.json",
]


def load_token(prompt_token: bool) -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token and prompt_token:
        token = getpass("Paste a Hugging Face read token: ")
    if not token:
        raise SystemExit("Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN, or run with --prompt-token.")
    return token


def repo_files(api: Any, repo_id: str, repo_type: str) -> list[str] | None:
    try:
        return sorted(api.list_repo_files(repo_id=repo_id, repo_type=repo_type))
    except Exception:
        return None


def download_json(hf_hub_download: Any, repo_id: str, repo_type: str, filename: str, token: str) -> dict | None:
    try:
        path = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=filename, token=token)
    except Exception:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(PROFILE_REPO_NAMES))
    parser.add_argument("--dataset-repo-id", help="Example: username/vic-investment-outcomes-sft")
    parser.add_argument("--model-repo-id", help="Example: username/vic-investment-qwen3-4b-lora-private")
    parser.add_argument("--dataset-repo-name", default=DEFAULT_DATASET_REPO_NAME)
    parser.add_argument("--model-repo-name", help="Override the model repo name inferred from --profile.")
    parser.add_argument("--prompt-token", action="store_true")
    args = parser.parse_args()

    token = load_token(args.prompt_token)

    try:
        from huggingface_hub import HfApi, hf_hub_download, whoami
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    user = whoami(token=token)
    username = user.get("name")
    if not username and (not args.dataset_repo_id or not args.model_repo_id):
        raise SystemExit("Could not infer Hugging Face username. Pass repo ids explicitly.")

    dataset_repo_id = args.dataset_repo_id or f"{username}/{args.dataset_repo_name}"
    model_repo_name = args.model_repo_name or PROFILE_REPO_NAMES[args.profile]
    model_repo_id = args.model_repo_id or f"{username}/{model_repo_name}"

    api = HfApi(token=token)
    dataset_files = repo_files(api, dataset_repo_id, "dataset")
    model_files = repo_files(api, model_repo_id, "model")

    dataset_missing = (
        EXPECTED_DATASET_FILES
        if dataset_files is None
        else [name for name in EXPECTED_DATASET_FILES if name not in dataset_files]
    )
    model_missing = (
        EXPECTED_MODEL_FILES
        if model_files is None
        else [name for name in EXPECTED_MODEL_FILES if name not in model_files]
    )

    finetuned_metrics = download_json(
        hf_hub_download, model_repo_id, "model", "finetuned_test_metrics.json", token
    )
    base_metrics = download_json(
        hf_hub_download, model_repo_id, "model", "base_model_test_metrics.json", token
    )
    gate = download_json(hf_hub_download, model_repo_id, "model", "finetune_gate.json", token)

    result = {
        "profile": args.profile,
        "dataset_repo_id": dataset_repo_id,
        "dataset_repo_exists": dataset_files is not None,
        "dataset_expected_files": len(EXPECTED_DATASET_FILES),
        "dataset_missing_files": dataset_missing,
        "dataset_ready": dataset_files is not None and not dataset_missing,
        "model_repo_id": model_repo_id,
        "model_repo_exists": model_files is not None,
        "model_expected_files": len(EXPECTED_MODEL_FILES),
        "model_missing_files": model_missing,
        "model_ready": model_files is not None and not model_missing,
        "base_model_accuracy": base_metrics.get("accuracy") if base_metrics else None,
        "finetuned_accuracy": finetuned_metrics.get("accuracy") if finetuned_metrics else None,
        "finetuned_scored": finetuned_metrics.get("scored") if finetuned_metrics else None,
        "gate": gate,
        "complete": dataset_files is not None
        and not dataset_missing
        and model_files is not None
        and not model_missing
        and bool(gate and gate.get("pass")),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "hf_finetune_status.json"
    rendered = json.dumps(result, indent=2, sort_keys=True)
    out_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
