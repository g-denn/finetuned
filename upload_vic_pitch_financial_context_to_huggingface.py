from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError


ROOT = Path(__file__).resolve().parent
DATASET_DIR = Path(
    os.environ.get("VIC_PITCH_CONTEXT_UPLOAD_DIR", str(ROOT / "eodhd_output" / "vic_pitch_financial_context"))
).resolve()
REPO_ID = "Gden/vic-pitch-financial-context-eodhd"
STATUS_PATH = ROOT / "eodhd_output" / "hf_vic_pitch_financial_context_upload_status.json"

IGNORE_PATTERNS = [
    "vic_pitch_financial_context.jsonl",
    ".cache/**",
]

EXPECTED_FILES = {
    "README.md",
    "dataset_summary.json",
    "vic_pitch_financial_context_preview.csv",
    "apple_examples.jsonl",
    "data/vic_pitch_financial_context-00000.jsonl",
    "data/vic_pitch_financial_context-00001.jsonl",
    "data/vic_pitch_financial_context-00002.jsonl",
    "data/vic_pitch_financial_context-00003.jsonl",
    "data/vic_pitch_financial_context-00004.jsonl",
    "data/vic_pitch_financial_context-00005.jsonl",
}


def write_status(payload: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required in the environment.")
    if not DATASET_DIR.exists():
        raise SystemExit(f"Missing dataset dir: {DATASET_DIR}")

    api = HfApi(token=token)
    create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True, token=token)
    info = api.dataset_info(REPO_ID, token=token)
    if not info.private:
        api.update_repo_visibility(REPO_ID, private=True, repo_type="dataset", token=token)

    write_status(
        {
            "repo_id": REPO_ID,
            "stage": "uploading",
            "private_before_upload": bool(info.private),
            "ignore_patterns": IGNORE_PATTERNS,
        }
    )

    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(DATASET_DIR),
        private=True,
        ignore_patterns=IGNORE_PATTERNS,
        num_workers=8,
        print_report=True,
        print_report_every=30,
    )

    verified = api.dataset_info(REPO_ID, token=token)
    sibling_names = {s.rfilename for s in verified.siblings}
    missing = sorted(EXPECTED_FILES - sibling_names)
    unexpected_full_jsonl = "vic_pitch_financial_context.jsonl" in sibling_names

    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename="dataset_summary.json",
        token=token,
    )
    summary = json.loads(Path(downloaded).read_text(encoding="utf-8"))

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
            "full_unsharded_jsonl_present_on_hub": unexpected_full_jsonl,
            "summary_rows": summary.get("rows"),
            "summary_shard_count": summary.get("shard_count"),
            "summary_apple_example_rows": summary.get("apple_example_rows"),
            "summary_jsonl_bytes": summary.get("jsonl_bytes"),
        }
    )

    print(json.dumps(json.loads(STATUS_PATH.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
