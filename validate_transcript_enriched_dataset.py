from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context_repaired_clean_transcripts_hf_stage"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    summary = json.loads((DATASET_DIR / "dataset_summary.json").read_text(encoding="utf-8"))
    failures = []
    totals = {
        "analysis_rows": 0,
        "sft_rows": 0,
        "analysis_rows_with_transcripts": 0,
        "analysis_transcripts": 0,
        "sft_rows_with_body_text": 0,
    }

    for split, meta in summary["splits"].items():
        for config in ("analysis", "sft"):
            path = DATASET_DIR / config / f"{split}.jsonl"
            rows = 0
            duplicate_ids = 0
            bad_rows = 0
            seen = set()
            for row in iter_jsonl(path):
                rows += 1
                idea_id = row.get("idea_id")
                if idea_id in seen:
                    duplicate_ids += 1
                seen.add(idea_id)

                serialized = json.dumps(row, ensure_ascii=False)
                if config == "analysis":
                    totals["analysis_rows"] += 1
                    transcripts = row.get("earnings_transcripts_publication_to_3y") or []
                    totals["analysis_transcripts"] += len(transcripts)
                    if transcripts:
                        totals["analysis_rows_with_transcripts"] += 1
                    count = (row.get("earnings_transcript_context_counts") or {}).get(
                        "publication_to_3y_transcript_count"
                    )
                    if count != len(transcripts):
                        bad_rows += 1
                else:
                    totals["sft_rows"] += 1
                    if (
                        "earnings_transcripts_publication_to_3y" in row
                        or '"transcript":' in serialized
                        or "Prepared Remarks" in serialized
                    ):
                        totals["sft_rows_with_body_text"] += 1
                        bad_rows += 1

            if rows != meta["rows"] or duplicate_ids or bad_rows:
                failures.append(
                    {
                        "config": config,
                        "split": split,
                        "rows": rows,
                        "expected": meta["rows"],
                        "duplicate_ids": duplicate_ids,
                        "bad_rows": bad_rows,
                    }
                )

    transcript_summary = summary["transcripts"]
    if totals["analysis_rows_with_transcripts"] != transcript_summary["rows_with_any_publication_to_3y_transcript"]:
        failures.append(
            {
                "mismatch": "rows_with_transcripts",
                "computed": totals["analysis_rows_with_transcripts"],
                "summary": transcript_summary["rows_with_any_publication_to_3y_transcript"],
            }
        )
    if totals["analysis_transcripts"] != transcript_summary["total_publication_to_3y_transcripts"]:
        failures.append(
            {
                "mismatch": "transcript_count",
                "computed": totals["analysis_transcripts"],
                "summary": transcript_summary["total_publication_to_3y_transcripts"],
            }
        )

    report = {"totals": totals, "transcripts": transcript_summary, "failures": failures}
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
