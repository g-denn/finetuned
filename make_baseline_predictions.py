#!/usr/bin/env python3
"""Create a simple majority-class baseline for the held-out test set."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CANONICAL_JSONL = ROOT / "data" / "processed" / "investment_canonical.jsonl"
OUT_PATH = ROOT / "reports" / "majority_baseline_predictions.jsonl"


def main() -> int:
    train_counts = Counter()
    test_rows: list[dict] = []
    with CANONICAL_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            date = row.get("publication_date") or ""
            if date <= "2020-05-11":
                train_counts[row["primary_outcome"]] += 1
            elif "2021-08-07" <= date <= "2022-11-03":
                test_rows.append(row)

    majority = train_counts.most_common(1)[0][0]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for row in test_rows:
            handle.write(
                json.dumps(
                    {
                        "idea_id": row["idea_id"],
                        "predicted_outcome": majority,
                        "baseline": "train_majority_class",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    print(json.dumps({"majority_class": majority, "test_rows": len(test_rows), "output": str(OUT_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
