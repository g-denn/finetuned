#!/usr/bin/env python3
"""Upload the processed private investment fine-tuning dataset to Hugging Face.

Usage:
    set HF_TOKEN=hf_...
    python upload_dataset_to_huggingface.py

Or:
    python upload_dataset_to_huggingface.py --repo-id YOUR_USERNAME/vic-investment-outcomes-sft

The token is intentionally read from the environment so it is not written into
source files or command history by this script.
"""

from __future__ import annotations

import argparse
import os
from getpass import getpass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", help="Example: username/vic-investment-outcomes-sft")
    parser.add_argument("--repo-name", default="vic-investment-outcomes-sft")
    parser.add_argument("--dry-run", action="store_true", help="Validate upload manifest without contacting Hugging Face.")
    parser.add_argument("--prompt-token", action="store_true", help="Prompt securely for a token if HF_TOKEN is not set.")
    parser.add_argument("--private", action="store_true", default=True)
    args = parser.parse_args()

    uploads = [
        DATA_DIR / "investment_train.jsonl",
        DATA_DIR / "investment_val.jsonl",
        DATA_DIR / "investment_test.jsonl",
        DATA_DIR / "investment_canonical.jsonl",
        ROOT / "hf_dataset_README.md",
        ROOT / "FINETUNING_RUNBOOK.md",
        REPORTS_DIR / "dataset_audit.md",
        REPORTS_DIR / "dataset_audit.json",
        REPORTS_DIR / "majority_baseline_metrics.json",
        REPORTS_DIR / "majority_baseline_predictions.jsonl",
        REPORTS_DIR / "text_baseline_metrics.json",
        REPORTS_DIR / "text_baseline_test_predictions.jsonl",
        REPORTS_DIR / "text_baseline_val_predictions.jsonl",
    ]
    manifest: list[tuple[Path, str]] = []
    for path in uploads:
        if not path.exists():
            raise SystemExit(f"Missing expected file: {path}")
        path_in_repo = (
            "README.md"
            if path.name == "hf_dataset_README.md"
            else path.name
            if path.parent in {DATA_DIR, ROOT}
            else f"reports/{path.name}"
        )
        manifest.append((path, path_in_repo))

    if args.dry_run:
        total_bytes = sum(path.stat().st_size for path, _ in manifest)
        print("dry_run=true")
        print(f"files={len(manifest)}")
        print(f"total_bytes={total_bytes}")
        for path, path_in_repo in manifest:
            print(f"{path_in_repo}\t{path.stat().st_size}\t{path}")
        return 0

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token and args.prompt_token:
        token = getpass("Paste a Hugging Face write token: ")
    if not token:
        raise SystemExit("Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN, or run with --prompt-token.")

    try:
        from huggingface_hub import HfApi, create_repo, upload_file, whoami
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    api = HfApi(token=token)
    repo_id = args.repo_id
    if not repo_id:
        user = whoami(token=token)
        username = user.get("name")
        if not username:
            raise SystemExit("Could not infer Hugging Face username from token. Pass --repo-id explicitly.")
        repo_id = f"{username}/{args.repo_name}"

    create_repo(repo_id, repo_type="dataset", private=args.private, token=token, exist_ok=True)
    for path, path_in_repo in manifest:
        upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print(f"uploaded {path}")

    info = api.dataset_info(repo_id)
    print(f"dataset_repo=https://huggingface.co/datasets/{info.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
