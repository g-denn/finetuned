from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "eodhd_output" / "dataset_financial_pull"
REPO_ID = "Gden/eodhd-vic-financial-data"
STATUS_PATH = ROOT / "eodhd_output" / "hf_extracted_upload_status.json"

ALLOW_PATTERNS = [
    "README.md",
    "raw/**",
    "progress_summary.json",
    "progress_checklist.csv",
    "progress_checklist.json",
    "stock_pull_manifest.csv",
    "stock_pull_manifest.json",
]

IGNORE_PATTERNS = [
    ".cache/**",
    "*.log",
    "*.tmp",
    "checkpoint_*.json",
    "pull_summary_*.json",
    "full_pull_runner_state*.json",
    "latest_checkpoint*",
    "parallel_pending_*",
]


def write_status(payload: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required in the environment.")
    if not DATASET_DIR.exists():
        raise SystemExit(f"Missing dataset directory: {DATASET_DIR}")

    api = HfApi(token=token)
    info = api.dataset_info(REPO_ID, token=token)
    if not info.private:
        api.update_repo_visibility(REPO_ID, private=True, repo_type="dataset", token=token)

    write_status(
        {
            "repo_id": REPO_ID,
            "stage": "uploading_extracted_files",
            "private_before_upload": bool(info.private),
            "allow_patterns": ALLOW_PATTERNS,
            "ignore_patterns": IGNORE_PATTERNS,
        }
    )

    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(DATASET_DIR),
        private=True,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        num_workers=8,
        print_report=True,
        print_report_every=30,
    )

    try:
        api.delete_file(
            path_in_repo="eodhd_dataset_financial_pull.zip",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Remove zip archive after extracted upload",
            token=token,
        )
        zip_delete_error = None
    except Exception as exc:  # noqa: BLE001 - report verification status below.
        zip_delete_error = repr(exc)

    verified = api.dataset_info(REPO_ID, token=token)
    sibling_names = {s.rfilename for s in verified.siblings}
    expected = {
        "README.md",
        "progress_summary.json",
        "progress_checklist.csv",
        "progress_checklist.json",
        "stock_pull_manifest.csv",
        "stock_pull_manifest.json",
        "raw/000660.KQ/eod_daily.json",
        "raw/000660.KQ/fundamentals.json",
    }
    missing = sorted(expected - sibling_names)

    try:
        api.repo_info(REPO_ID, repo_type="dataset", token=False)
        public_accessible = True
    except HfHubHTTPError:
        public_accessible = False

    write_status(
        {
            "repo_id": REPO_ID,
            "stage": "complete",
            "private": bool(verified.private),
            "public_accessible_without_token": public_accessible,
            "sibling_count": len(verified.siblings),
            "missing_expected_files": missing,
            "zip_present_on_hub": "eodhd_dataset_financial_pull.zip" in sibling_names,
            "zip_delete_error": zip_delete_error,
        }
    )

    print(json.dumps(json.loads(STATUS_PATH.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
