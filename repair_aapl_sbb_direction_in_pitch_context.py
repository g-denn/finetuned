from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
IDEA_ID = "1f23707e-b4c5-46cc-b39c-11fa4e949b87"
OUT_DIR = ROOT / "eodhd_output" / "vic_pitch_financial_context"
CANONICAL = ROOT / "data" / "processed" / "investment_canonical.csv"
FULL_JSONL = OUT_DIR / "vic_pitch_financial_context.jsonl"
APPLE_EXAMPLES = OUT_DIR / "apple_examples.jsonl"
PREVIEW = OUT_DIR / "vic_pitch_financial_context_preview.csv"
DATA_DIR = OUT_DIR / "data"
EXAMPLES_DIR = OUT_DIR / "examples"


def canonical_row() -> dict:
    df = pd.read_csv(CANONICAL, dtype=str, keep_default_na=False)
    rows = df[df["idea_id"] == IDEA_ID]
    if len(rows) != 1:
        raise SystemExit(f"Expected exactly one canonical row for {IDEA_ID}, got {len(rows)}")
    return rows.iloc[0].to_dict()


def repair_payload(row: dict, canonical: dict) -> dict:
    row["is_short"] = canonical["is_short"]
    for horizon in ("1y", "3y", "5y", "10y", "20y"):
        row["performance"][f"raw_perf_{horizon}"] = canonical[f"raw_perf_{horizon}"]
        row["performance"][f"directional_perf_{horizon}"] = canonical[f"directional_perf_{horizon}"]
        row["performance"][f"outcome_{horizon}"] = canonical[f"outcome_{horizon}"]
    row["performance"]["primary_horizon"] = canonical["primary_horizon"]
    row["performance"]["primary_outcome"] = canonical["primary_outcome"]
    row["raw_perf_3y"] = canonical["raw_perf_3y"]
    row["outcome_3y"] = canonical["outcome_3y"]
    row["raw_perf_5y"] = canonical["raw_perf_5y"]
    row["outcome_5y"] = canonical["outcome_5y"]
    return row


def repair_jsonl(path: Path, canonical: dict) -> int:
    tmp = path.with_suffix(path.suffix + ".tmp")
    changed = 0
    with path.open("r", encoding="utf-8") as source, tmp.open("w", encoding="utf-8", newline="\n") as out:
        for line in source:
            row = json.loads(line)
            if row.get("idea_id") == IDEA_ID:
                row = repair_payload(row, canonical)
                changed += 1
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)
    return changed


def repair_preview(canonical: dict) -> int:
    df = pd.read_csv(PREVIEW, dtype=str, keep_default_na=False)
    mask = df["idea_id"] == IDEA_ID
    changed = int(mask.sum())
    if changed:
        df.loc[mask, "raw_perf_3y"] = canonical["raw_perf_3y"]
        df.loc[mask, "outcome_3y"] = canonical["outcome_3y"]
        df.loc[mask, "raw_perf_5y"] = canonical["raw_perf_5y"]
        df.loc[mask, "outcome_5y"] = canonical["outcome_5y"]
        df.to_csv(PREVIEW, index=False)
    return changed


def main() -> None:
    canonical = canonical_row()
    results = {
        "canonical_is_short": canonical["is_short"],
        "canonical_outcome_3y": canonical["outcome_3y"],
        "canonical_outcome_5y": canonical["outcome_5y"],
        "full_jsonl_changed": repair_jsonl(FULL_JSONL, canonical),
        "apple_examples_changed": repair_jsonl(APPLE_EXAMPLES, canonical),
        "preview_changed": repair_preview(canonical),
        "shards_changed": {},
    }
    for path in sorted(DATA_DIR.glob("*.jsonl")):
        changed = repair_jsonl(path, canonical)
        if changed:
            results["shards_changed"][path.name] = changed

    EXAMPLES_DIR.mkdir(exist_ok=True)
    target = None
    for line in APPLE_EXAMPLES.open("r", encoding="utf-8"):
        row = json.loads(line)
        if row.get("idea_id") == IDEA_ID:
            target = row
            break
    if target:
        (EXAMPLES_DIR / "aapl_sbb_2011_05_02_full_record.json").write_text(
            json.dumps(target, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
