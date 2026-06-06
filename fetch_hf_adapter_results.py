#!/usr/bin/env python3
"""Download fine-tuned adapter evaluation artifacts from Hugging Face.

Usage:
    set HF_TOKEN=hf_...
    python fetch_hf_adapter_results.py

The script infers the username from the token unless --repo-id is passed.
It downloads:

- finetuned_test_predictions.jsonl
- finetuned_test_metrics.json
- base_model_test_predictions.jsonl, if present
- base_model_test_metrics.json, if present

Then it re-runs the local evaluator against the downloaded predictions and,
when enough artifacts exist, runs the local fine-tune pass/fail gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from getpass import getpass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
PROFILE_REPO_NAMES = {
    "free_t4_qwen3_4b": "vic-investment-qwen3-4b-lora-private",
    "strong_l4_or_a100_qwen3_8b": "vic-investment-qwen3-8b-lora-private",
}
DEFAULT_PROFILE = "free_t4_qwen3_4b"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", help="Example: username/vic-investment-qwen3-4b-lora-private")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(PROFILE_REPO_NAMES))
    parser.add_argument("--repo-name", help="Override the model repo name inferred from --profile.")
    parser.add_argument("--prompt-token", action="store_true", help="Prompt securely for a token if HF_TOKEN is not set.")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token and args.prompt_token:
        token = getpass("Paste a Hugging Face read token: ")
    if not token:
        raise SystemExit("Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN, or run with --prompt-token.")

    try:
        from huggingface_hub import hf_hub_download, whoami
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    repo_id = args.repo_id
    if not repo_id:
        user = whoami(token=token)
        username = user.get("name")
        if not username:
            raise SystemExit("Could not infer Hugging Face username from token. Pass --repo-id explicitly.")
        repo_name = args.repo_name or PROFILE_REPO_NAMES[args.profile]
        repo_id = f"{username}/{repo_name}"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, str] = {}
    required = ("finetuned_test_predictions.jsonl", "finetuned_test_metrics.json")
    optional = ("base_model_test_predictions.jsonl", "base_model_test_metrics.json")

    for filename in required + optional:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                filename=filename,
                token=token,
            )
        except Exception:
            if filename in required:
                raise
            continue
        target = REPORTS_DIR / filename
        target.write_bytes(Path(path).read_bytes())
        downloaded[filename] = str(target)

    local_metrics = REPORTS_DIR / "finetuned_test_metrics_local_rescore.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "evaluate_outcome_predictions.py"),
            downloaded["finetuned_test_predictions.jsonl"],
            "--output",
            str(local_metrics),
        ],
        check=True,
    )

    base_local_metrics = None
    if "base_model_test_predictions.jsonl" in downloaded:
        base_local_metrics = REPORTS_DIR / "base_model_test_metrics_local_rescore.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "evaluate_outcome_predictions.py"),
                downloaded["base_model_test_predictions.jsonl"],
                "--output",
                str(base_local_metrics),
            ],
            check=True,
        )

    gate = subprocess.run(
        [sys.executable, str(ROOT / "check_finetune_gate.py")],
        check=False,
        capture_output=True,
        text=True,
    )

    result = {
        "repo_id": repo_id,
        "profile": args.profile,
        "downloaded": downloaded,
        "local_rescore": str(local_metrics),
        "base_local_rescore": str(base_local_metrics) if base_local_metrics else None,
        "gate_exit_code": gate.returncode,
        "gate_output": gate.stdout,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
