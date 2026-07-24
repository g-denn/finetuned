from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "eodhd_output" / "dataset_financial_pull"
REPO_ID = "Gden/eodhd-vic-financial-data"
STATUS_PATH = ROOT / "eodhd_output" / "hf_combined_table_upload_status.json"

UPLOADS = [
    (DATASET_DIR / "README.md", "README.md"),
    (DATASET_DIR / "eodhd_combined_stock_table.csv", "eodhd_combined_stock_table.csv"),
    (
        DATASET_DIR / "eodhd_combined_stock_table_summary.json",
        "eodhd_combined_stock_table_summary.json",
    ),
]


def write_status(payload: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required in the environment.")

    missing_local = [str(local) for local, _ in UPLOADS if not local.exists()]
    if missing_local:
        raise SystemExit(f"Missing local upload files: {missing_local}")

    api = HfApi(token=token)
    info = api.dataset_info(REPO_ID, token=token)
    if not info.private:
        api.update_repo_visibility(REPO_ID, private=True, repo_type="dataset", token=token)

    write_status(
        {
            "repo_id": REPO_ID,
            "stage": "uploading",
            "private_before_upload": bool(info.private),
            "files": [remote for _, remote in UPLOADS],
        }
    )

    commits = []
    for local, remote in UPLOADS:
        commit = api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Upload {remote}",
            token=token,
        )
        commits.append({"path": remote, "commit_oid": commit.oid, "commit_url": commit.commit_url})

    verified = api.dataset_info(REPO_ID, token=token)
    sibling_names = {s.rfilename for s in verified.siblings}
    expected = {remote for _, remote in UPLOADS}
    missing_remote = sorted(expected - sibling_names)

    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename="eodhd_combined_stock_table_summary.json",
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
            "missing_remote_files": missing_remote,
            "summary_rows": summary.get("rows"),
            "summary_columns": summary.get("columns"),
            "commits": commits,
        }
    )

    print(json.dumps(json.loads(STATUS_PATH.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
