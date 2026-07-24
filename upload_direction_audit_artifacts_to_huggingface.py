from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


ROOT = Path(__file__).resolve().parent
REPO_ID = "Gden/vic-investment-outcomes-sft-duplicate"

STATIC_UPLOADS = [
    (ROOT / "idea_direction_overrides.json", "audits/idea_direction_overrides.json"),
    (ROOT / "idea_direction_confirmations.json", "audits/idea_direction_confirmations.json"),
    (ROOT / "idea_quality_flags.json", "audits/idea_quality_flags.json"),
    (
        ROOT / "reports" / "direction_label_audit" / "direction_label_audit_summary.json",
        "audits/direction_label_audit_summary.json",
    ),
    (
        ROOT / "reports" / "direction_label_audit" / "direction_label_audit.csv",
        "audits/direction_label_audit.csv",
    ),
    (
        ROOT / "reports" / "direction_label_audit" / "direction_review_queue.csv",
        "audits/direction_review_queue.csv",
    ),
]


def review_packet_uploads() -> list[tuple[Path, str]]:
    packet_dir = ROOT / "reports" / "direction_label_audit" / "review_packets"
    uploads: list[tuple[Path, str]] = []
    for pattern in [
        "direction_review_packet*.csv",
        "direction_review_packet*.md",
        "manual_direction_review_overrides*.csv",
    ]:
        for path in sorted(packet_dir.glob(pattern)):
            uploads.append((path, f"audits/review_packets/{path.name}"))
    return uploads


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required in the environment.")

    uploads = STATIC_UPLOADS + review_packet_uploads()
    missing = [str(path) for path, _ in uploads if not path.exists()]
    if missing:
        raise SystemExit("Missing expected files:\n" + "\n".join(missing))

    api = HfApi(token=token)
    create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True, token=token)
    info = api.dataset_info(REPO_ID, token=token)
    if not info.private:
        api.update_repo_visibility(REPO_ID, private=True, repo_type="dataset", token=token)

    uploaded = []
    for path, path_in_repo in uploads:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token,
        )
        uploaded.append(path_in_repo)

    verified = api.dataset_info(REPO_ID, token=token)
    sibling_names = {s.rfilename for s in verified.siblings}
    missing_after = sorted(path_in_repo for _, path_in_repo in uploads if path_in_repo not in sibling_names)
    print(
        json.dumps(
            {
                "repo_id": REPO_ID,
                "private": bool(verified.private),
                "uploaded": uploaded,
                "missing_after_upload": missing_after,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
