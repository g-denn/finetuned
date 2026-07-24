from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OVERRIDES_PATH = ROOT / "idea_direction_overrides.json"
SUGGESTED_PATH = (
    ROOT
    / "reports"
    / "direction_label_audit"
    / "suggested_direction_overrides_high_confidence.json"
)
CHECKPOINT_PATH = ROOT / "idea_direction_overrides.before_explicit_audit.json"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    current = read_json(OVERRIDES_PATH)
    suggested = read_json(SUGGESTED_PATH)

    if OVERRIDES_PATH.exists() and not CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.write_text(
            json.dumps(current, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    merged = dict(current)
    added = 0
    changed = 0
    unchanged = 0
    for idea_id, override in suggested.items():
        override = dict(override)
        override["source"] = "explicit_thesis_direction_audit"
        override["applied_at_utc"] = datetime.now(timezone.utc).isoformat()
        if idea_id not in merged:
            added += 1
        elif merged[idea_id].get("is_short") != override.get("is_short"):
            changed += 1
        else:
            unchanged += 1
        merged[idea_id] = override

    OVERRIDES_PATH.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "current_before": len(current),
                "suggested": len(suggested),
                "added": added,
                "changed": changed,
                "unchanged": unchanged,
                "merged_total": len(merged),
                "checkpoint": str(CHECKPOINT_PATH),
                "overrides": str(OVERRIDES_PATH),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
