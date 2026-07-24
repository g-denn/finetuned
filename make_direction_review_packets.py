from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
AUDIT_CSV = ROOT / "reports" / "direction_label_audit" / "direction_label_audit.csv"
CANONICAL_CSV = ROOT / "data" / "processed" / "investment_canonical.csv"
CONFIRMATIONS_JSON = ROOT / "idea_direction_confirmations.json"
QUALITY_FLAGS_JSON = ROOT / "idea_quality_flags.json"
OUT_DIR = ROOT / "reports" / "direction_label_audit" / "review_packets"


def clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def load_evidence(value: str) -> list[dict[str, Any]]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def compact_evidence(value: str) -> str:
    evidence = load_evidence(value)
    return " | ".join(clean_space(item.get("excerpt", "")) for item in evidence[:3])


def load_confirmed_ids() -> set[str]:
    if not CONFIRMATIONS_JSON.exists():
        return set()
    confirmations = json.loads(CONFIRMATIONS_JSON.read_text(encoding="utf-8"))
    return set(confirmations)


def load_quality_flagged_ids() -> set[str]:
    if not QUALITY_FLAGS_JSON.exists():
        return set()
    flags = json.loads(QUALITY_FLAGS_JSON.read_text(encoding="utf-8"))
    return set(flags)


def direction_score(row: dict[str, Any]) -> tuple[int, int, int]:
    confidence_rank = {"high": 3, "medium": 2, "low": 1, "none": 0}
    strict_rank = 1 if str(row.get("strict_mismatch_current")) == "True" else 0
    explicit_rank = 1 if str(row.get("explicit_mismatch_current")) == "True" else 0
    mismatch_rank = 1 if str(row.get("mismatch_current")) == "True" else 0
    margin = int(float(row.get("score_margin") or 0))
    return (
        strict_rank,
        explicit_rank,
        mismatch_rank,
        confidence_rank.get(str(row.get("confidence")), 0),
        margin,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--confidence", default="high,medium")
    parser.add_argument("--inferred", default="long,short")
    parser.add_argument(
        "--prefix",
        default="direction_review_packet",
        help="Output file prefix under reports/direction_label_audit/review_packets.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canonical = pd.read_csv(CANONICAL_CSV, dtype=str, keep_default_na=False)
    audit = pd.read_csv(AUDIT_CSV, dtype=str, keep_default_na=False)
    canonical_by_id = {row["idea_id"]: row for row in canonical.to_dict(orient="records")}
    confirmed_ids = load_confirmed_ids()
    quality_flagged_ids = load_quality_flagged_ids()

    confidences = {item.strip() for item in args.confidence.split(",") if item.strip()}
    inferred = {item.strip() for item in args.inferred.split(",") if item.strip()}
    rows = []
    for row in audit.to_dict(orient="records"):
        if row.get("mismatch_current") != "True":
            continue
        if row.get("idea_id") in confirmed_ids:
            continue
        if row.get("idea_id") in quality_flagged_ids:
            continue
        if row.get("confirmed_current_label") == "True":
            continue
        if row.get("confidence") not in confidences:
            continue
        if row.get("inferred_direction") not in inferred:
            continue
        canonical_row = canonical_by_id.get(row["idea_id"], {})
        text = "\n\n".join(
            part
            for part in [
                canonical_row.get("description", ""),
                canonical_row.get("catalyst", ""),
            ]
            if part
        )
        row["intro_excerpt"] = clean_space(text[:2200])
        row["long_evidence_compact"] = compact_evidence(row.get("long_evidence", ""))
        row["short_evidence_compact"] = compact_evidence(row.get("short_evidence", ""))
        row["explicit_long_evidence_compact"] = compact_evidence(row.get("explicit_long_evidence", ""))
        row["explicit_short_evidence_compact"] = compact_evidence(row.get("explicit_short_evidence", ""))
        rows.append(row)

    rows.sort(key=direction_score, reverse=True)
    rows = rows[: args.limit]

    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.prefix).strip("._")
    if not safe_prefix:
        raise SystemExit("--prefix must contain at least one safe filename character.")
    csv_path = OUT_DIR / f"{safe_prefix}.csv"
    md_path = OUT_DIR / f"{safe_prefix}.md"
    fields = [
        "idea_id",
        "raw_symbol",
        "eodhd_symbol",
        "company_name",
        "publication_date",
        "author_user_id",
        "link",
        "current_is_short",
        "inferred_direction",
        "confidence",
        "long_score",
        "short_score",
        "score_margin",
        "explicit_direction",
        "explicit_long_score",
        "explicit_short_score",
        "strict_direction",
        "strict_support_reason",
        "long_evidence_compact",
        "short_evidence_compact",
        "explicit_long_evidence_compact",
        "explicit_short_evidence_compact",
        "intro_excerpt",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Direction Review Packet",
        "",
        f"Rows: {len(rows)}",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        proposed = "SHORT" if row["inferred_direction"] == "short" else "LONG"
        current = "SHORT" if row["current_is_short"] == "True" else "LONG"
        lines.extend(
            [
                f"## {index}. {row.get('raw_symbol')} {row.get('company_name')}",
                "",
                f"- idea_id: `{row.get('idea_id')}`",
                f"- date: `{row.get('publication_date')}`",
                f"- current: `{current}`",
                f"- proposed: `{proposed}`",
                f"- confidence: `{row.get('confidence')}`",
                f"- scores: long `{row.get('long_score')}`, short `{row.get('short_score')}`",
                f"- explicit: `{row.get('explicit_direction')}` (long `{row.get('explicit_long_score')}`, short `{row.get('explicit_short_score')}`)",
                f"- strict: `{row.get('strict_direction')}`; {row.get('strict_support_reason')}",
                f"- link: {row.get('link')}",
                "",
                "**Evidence**",
                "",
                f"- long: {row.get('long_evidence_compact')}",
                f"- short: {row.get('short_evidence_compact')}",
                f"- explicit long: {row.get('explicit_long_evidence_compact')}",
                f"- explicit short: {row.get('explicit_short_evidence_compact')}",
                "",
                "**Intro**",
                "",
                row.get("intro_excerpt", ""),
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "rows": len(rows),
                "csv": str(csv_path),
                "markdown": str(md_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
