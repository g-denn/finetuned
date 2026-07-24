from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context_clean_hf_stage"
SPLITS = ("train", "validation", "test")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def validate_analysis_split(split: str) -> dict:
    path = DATASET_DIR / "analysis" / f"{split}.jsonl"
    stats = {
        "rows": 0,
        "missing_ids": 0,
        "duplicate_ids": 0,
        "missing_labels": 0,
        "missing_financials": 0,
        "has_forward_financial_keys": 0,
    }
    seen = set()
    for _line_number, row in iter_jsonl(path):
        stats["rows"] += 1
        idea_id = row.get("idea_id")
        if not idea_id:
            stats["missing_ids"] += 1
        elif idea_id in seen:
            stats["duplicate_ids"] += 1
        seen.add(idea_id)

        label = row.get("label") or {}
        if label.get("raw_perf_3y") is None or not label.get("outcome_3y"):
            stats["missing_labels"] += 1

        if (
            len(row.get("financials_latest_annual_asof_pitch") or []) < 3
            or len(row.get("financials_latest_quarterly_asof_pitch") or []) < 3
            or not row.get("financials_trailing_5y_asof_pitch")
        ):
            stats["missing_financials"] += 1

        forbidden_keys = {
            key for key in row.keys() if key.startswith("financials_forward_") or key == "financials_forward"
        }
        if forbidden_keys:
            stats["has_forward_financial_keys"] += 1
    return stats


def validate_sft_split(split: str) -> dict:
    path = DATASET_DIR / "sft" / f"{split}.jsonl"
    stats = {
        "rows": 0,
        "missing_ids": 0,
        "duplicate_ids": 0,
        "missing_messages": 0,
        "bad_roles": 0,
        "missing_assistant_target": 0,
        "contains_forward_financial_text": 0,
    }
    seen = set()
    for _line_number, row in iter_jsonl(path):
        stats["rows"] += 1
        idea_id = row.get("idea_id")
        if not idea_id:
            stats["missing_ids"] += 1
        elif idea_id in seen:
            stats["duplicate_ids"] += 1
        seen.add(idea_id)

        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            stats["missing_messages"] += 1
            continue
        if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
            stats["bad_roles"] += 1
        try:
            assistant_payload = json.loads(messages[-1].get("content") or "{}")
        except json.JSONDecodeError:
            assistant_payload = {}
        if assistant_payload.get("raw_perf_3y") is None or not assistant_payload.get("outcome_3y"):
            stats["missing_assistant_target"] += 1
        if "financials_forward_" in json.dumps(messages, ensure_ascii=False):
            stats["contains_forward_financial_text"] += 1
    return stats


def main() -> None:
    summary = json.loads((DATASET_DIR / "dataset_summary.json").read_text(encoding="utf-8"))
    report = {"expected_rows": summary["splits"], "analysis": {}, "sft": {}}
    for split in SPLITS:
        report["analysis"][split] = validate_analysis_split(split)
        report["sft"][split] = validate_sft_split(split)

    failures = []
    for split in SPLITS:
        expected = summary["splits"][split]["rows"]
        for config in ("analysis", "sft"):
            stats = report[config][split]
            if stats["rows"] != expected:
                failures.append(f"{config}/{split}: expected {expected} rows, found {stats['rows']}")
            for key, value in stats.items():
                if key != "rows" and value:
                    failures.append(f"{config}/{split}: {key}={value}")
    report["failures"] = failures
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
