from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, upload_file


ROOT = Path(__file__).resolve().parent
REPO_ID = "Gden/vic-pitch-financial-context-eodhd"
DATASET_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context"
STATUS_PATH = ROOT / "eodhd_output" / "hf_aapl_sbb_direction_fix_status.json"

FILES = [
    ("apple_examples.jsonl", DATASET_DIR / "apple_examples.jsonl"),
    ("vic_pitch_financial_context_preview.csv", DATASET_DIR / "vic_pitch_financial_context_preview.csv"),
    (
        "data/vic_pitch_financial_context-00003.jsonl",
        DATASET_DIR / "data" / "vic_pitch_financial_context-00003.jsonl",
    ),
    (
        "examples/aapl_sbb_2011_05_02_full_record.json",
        DATASET_DIR / "examples" / "aapl_sbb_2011_05_02_full_record.json",
    ),
    (
        "examples/aapl_sbb_2011_05_02_summary.json",
        DATASET_DIR / "examples" / "aapl_sbb_2011_05_02_summary.json",
    ),
]


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required.")

    commits = []
    for path_in_repo, local_path in FILES:
        commit = upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token,
            commit_message=f"Fix AAPL SBB 2011 direction in {path_in_repo}",
        )
        commits.append({"path": path_in_repo, "commit": commit.oid, "url": commit.commit_url})

    api = HfApi(token=token)
    info = api.dataset_info(REPO_ID, token=token)
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename="apple_examples.jsonl",
        token=token,
        force_download=True,
    )
    found = None
    with open(downloaded, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("idea_id") == "1f23707e-b4c5-46cc-b39c-11fa4e949b87":
                found = {
                    "is_short": row.get("is_short"),
                    "raw_perf_3y": row.get("raw_perf_3y"),
                    "outcome_3y": row.get("outcome_3y"),
                    "raw_perf_5y": row.get("raw_perf_5y"),
                    "outcome_5y": row.get("outcome_5y"),
                    "performance": row.get("performance"),
                }
                break
    status = {
        "repo_id": REPO_ID,
        "private": bool(info.private),
        "commits": commits,
        "verified_aapl_sbb_row": found,
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
