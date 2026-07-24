from __future__ import annotations

import csv
import json
import argparse
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OVERRIDES_PATH = ROOT / "idea_direction_overrides.json"
CONFIRMATIONS_PATH = ROOT / "idea_direction_confirmations.json"
QUALITY_FLAGS_PATH = ROOT / "idea_quality_flags.json"
DEFAULT_PACKET_CSV = (
    ROOT
    / "reports"
    / "direction_label_audit"
    / "review_packets"
    / "direction_review_packet.csv"
)
DEFAULT_AUDIT_OUT = (
    ROOT
    / "reports"
    / "direction_label_audit"
    / "review_packets"
    / "manual_direction_review_overrides_2026-06-12.csv"
)

# Reviewed from the 2026-06-12 high-confidence long packet. This row opens by
# discussing a prior long write-up, but the author then says they are short.
SKIP_IDEA_IDS = {
    "9d99cb91-6d30-44e3-b2fc-3c5d29152266",  # WAB
}


def clean_space(value: str) -> str:
    return " ".join(str(value or "").split())


def load_id_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_AUDIT_OUT)
    parser.add_argument(
        "--source",
        default="manual_thesis_direction_review_2026_06_12_packet_001",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Idea id to leave unchanged. Can be repeated.",
    )
    parser.add_argument(
        "--skip-quality-flags",
        action="store_true",
        help=(
            "Also leave quality-flagged rows unchanged. By default quality "
            "flags do not block direction corrections because symbol identity "
            "and thesis direction are separate review findings."
        ),
    )
    args = parser.parse_args()

    now = datetime.now(UTC).isoformat()
    overrides = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    applied_rows: list[dict[str, str]] = []
    skip_ids = SKIP_IDEA_IDS | set(args.skip) | load_id_set(CONFIRMATIONS_PATH)
    if args.skip_quality_flags:
        skip_ids |= load_id_set(QUALITY_FLAGS_PATH)

    with args.packet.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            idea_id = row["idea_id"]
            if idea_id in skip_ids:
                continue
            if row.get("current_is_short") != "True":
                continue
            if row.get("inferred_direction") != "long":
                continue

            evidence = clean_space(row.get("long_evidence_compact", ""))[:700]
            reason = (
                "Manual high-confidence thesis direction review inferred long; "
                f"current direction flag disagreed. Evidence: {evidence}"
            )
            overrides[idea_id] = {
                "is_short": False,
                "reason": reason,
                "source": args.source,
                "applied_at_utc": now,
            }
            applied_rows.append(
                {
                    "idea_id": idea_id,
                    "raw_symbol": row.get("raw_symbol", ""),
                    "company_name": row.get("company_name", ""),
                    "publication_date": row.get("publication_date", ""),
                    "new_is_short": "False",
                    "review_source": args.source,
                    "evidence": evidence,
                }
            )

    OVERRIDES_PATH.write_text(
        json.dumps(overrides, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "idea_id",
            "raw_symbol",
            "company_name",
            "publication_date",
            "new_is_short",
            "review_source",
            "evidence",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(applied_rows)

    print(
        json.dumps(
            {
                "applied": len(applied_rows),
                "audit_csv": str(args.output),
                "overrides": str(OVERRIDES_PATH),
                "skipped": sorted(skip_ids),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
