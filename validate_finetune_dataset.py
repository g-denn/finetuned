#!/usr/bin/env python3
"""Validate generated chat JSONL fine-tuning splits."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
SPLITS = ("train", "val", "test")


def validate_split(name: str) -> dict[str, int]:
    path = DATA_DIR / f"investment_{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")

    rows = 0
    max_chars = 0
    missing_messages = 0
    bad_roles = 0
    missing_outcome = 0
    seen_ids: set[str] = set()
    duplicate_ids = 0

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            rows += 1
            messages = payload.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                missing_messages += 1
                continue
            roles = [message.get("role") for message in messages if isinstance(message, dict)]
            if roles != ["system", "user", "assistant"]:
                bad_roles += 1
            assistant = messages[-1].get("content", "")
            if "Validated outcome summary:" not in assistant or "Primary training label:" not in assistant:
                missing_outcome += 1
            max_chars = max(max_chars, sum(len(str(message.get("content", ""))) for message in messages))
            idea_id = (payload.get("metadata") or {}).get("idea_id")
            if idea_id in seen_ids:
                duplicate_ids += 1
            elif idea_id:
                seen_ids.add(idea_id)

    return {
        "rows": rows,
        "unique_idea_ids": len(seen_ids),
        "duplicate_ids": duplicate_ids,
        "max_chars": max_chars,
        "missing_messages": missing_messages,
        "bad_roles": bad_roles,
        "missing_outcome": missing_outcome,
    }


def main() -> int:
    results = {split: validate_split(split) for split in SPLITS}
    print(json.dumps(results, indent=2, sort_keys=True))
    if any(
        stats["missing_messages"] or stats["bad_roles"] or stats["missing_outcome"] or stats["duplicate_ids"]
        for stats in results.values()
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
