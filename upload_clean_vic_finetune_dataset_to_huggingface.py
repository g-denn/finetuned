from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT / ".codex_deps" / "python"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

from huggingface_hub import HfApi, create_repo, hf_hub_download  # noqa: E402
from huggingface_hub.errors import HfHubHTTPError  # noqa: E402


DATASET_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context_clean_hf_stage"
REPO_ID = os.environ.get("HF_CLEAN_VIC_REPO_ID", "Gden/vic-pitch-financial-context-clean-sft")
STATUS_PATH = ROOT / "eodhd_output" / "hf_clean_vic_finetune_upload_status.json"

EXPECTED_FILES = {
    "README.md",
    "dataset_summary.json",
    "analysis/train.jsonl",
    "analysis/validation.jsonl",
    "analysis/test.jsonl",
    "sft/train.jsonl",
    "sft/validation.jsonl",
    "sft/test.jsonl",
}


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
    create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True, token=token)
    info = api.dataset_info(REPO_ID, token=token)
    if not info.private:
        api.update_repo_visibility(REPO_ID, private=True, repo_type="dataset", token=token)

    local_summary = json.loads((DATASET_DIR / "dataset_summary.json").read_text(encoding="utf-8"))
    write_status(
        {
            "repo_id": REPO_ID,
            "stage": "uploading",
            "private_before_upload": bool(info.private),
            "local_summary": local_summary,
        }
    )

    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(DATASET_DIR),
        private=True,
        num_workers=8,
        print_report=True,
        print_report_every=30,
    )

    verified = api.dataset_info(REPO_ID, token=token)
    sibling_names = {s.rfilename for s in verified.siblings}
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename="dataset_summary.json",
        token=token,
    )
    remote_summary = json.loads(Path(downloaded).read_text(encoding="utf-8"))

    try:
        api.repo_info(REPO_ID, repo_type="dataset", token=False)
        public_accessible = True
    except HfHubHTTPError:
        public_accessible = False

    status = {
        "repo_id": REPO_ID,
        "stage": "complete",
        "private": bool(verified.private),
        "public_accessible_without_token": public_accessible,
        "sibling_count": len(verified.siblings),
        "missing_expected_files": sorted(EXPECTED_FILES - sibling_names),
        "remote_summary_matches_local": remote_summary == local_summary,
        "accepted_rows": remote_summary.get("accepted_rows"),
        "splits": remote_summary.get("splits"),
    }
    write_status(status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
