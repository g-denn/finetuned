from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


ROOT = Path(__file__).resolve().parent
DATASET_DIR = Path(os.environ.get("EODHD_JK_OUT_DIR", str(ROOT / "eodhd_output" / "japan_korea_fundamentals")))
STATUS_PATH = ROOT / "eodhd_output" / "hf_japan_korea_fundamentals_upload_status.json"

ALLOW_PATTERNS = [
    "README.md",
    "raw/**",
    "normalized/**",
    "screening/**",
    "stock_pull_manifest.csv",
    "stock_pull_manifest.json",
    "progress_summary.json",
    "progress_checklist.csv",
    "progress_checklist.json",
    "exchanges_selected.json",
    "symbol_list_status.json",
    "earnings_transcript_availability.json",
    "normalization_summary.json",
    "dataset_audit.json",
]

IGNORE_PATTERNS = [
    "*.tmp",
    "*.log",
    "checkpoint_*.json",
    "latest_checkpoint.json",
]


def write_status(payload: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID")
    if not token:
        raise SystemExit("HF_TOKEN is required in the environment.")
    if not repo_id:
        raise SystemExit("HF_REPO_ID is required in the environment, for example username/eodhd-japan-korea-fundamentals.")
    if not DATASET_DIR.exists():
        raise SystemExit(f"Missing dataset directory: {DATASET_DIR}")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True, token=token)
    info = api.dataset_info(repo_id, token=token)
    if not info.private:
        api.update_repo_visibility(repo_id, private=True, repo_type="dataset", token=token)

    write_status(
        {
            "repo_id": repo_id,
            "stage": "uploading",
            "private_before_upload": bool(info.private),
            "allow_patterns": ALLOW_PATTERNS,
            "ignore_patterns": IGNORE_PATTERNS,
        }
    )

    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(DATASET_DIR),
        private=True,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        num_workers=8,
        print_report=True,
        print_report_every=30,
    )

    verified = api.dataset_info(repo_id, token=token)
    sibling_names = {s.rfilename for s in verified.siblings}
    expected = {
        "README.md",
        "stock_pull_manifest.csv",
        "progress_summary.json",
        "progress_checklist.csv",
        "earnings_transcript_availability.json",
        "normalized/companies.csv",
        "normalized/income_statement.csv",
        "normalized/income_statement_latest_5y.csv",
        "normalized/balance_sheet.csv",
        "normalized/balance_sheet_latest_5y.csv",
        "normalized/cash_flow.csv",
        "normalized/cash_flow_latest_5y.csv",
        "normalized/earnings.csv",
        "normalized/fundamentals_raw_payloads.jsonl",
        "dataset_audit.json",
        "screening/shareholder_yield_screen_all.csv",
        "screening/shareholder_yield_screen_top.csv",
        "screening/screening_summary.json",
    }

    try:
        api.repo_info(repo_id, repo_type="dataset", token=False)
        public_accessible = True
    except HfHubHTTPError:
        public_accessible = False

    write_status(
        {
            "repo_id": repo_id,
            "stage": "complete",
            "private": bool(verified.private),
            "public_accessible_without_token": public_accessible,
            "sibling_count": len(verified.siblings),
            "missing_expected_files": sorted(expected - sibling_names),
        }
    )
    print(json.dumps(json.loads(STATUS_PATH.read_text(encoding="utf-8")), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
