#!/usr/bin/env python3
"""Merge safely repaired provider-error symbols into the training output."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from salvage_partial_horizon_labels import scalar


BASE_DIR = Path("eodhd_output/full_run")
REPAIR_DIR = Path("eodhd_output/provider_error_symbol_repair")

VALIDATION_IN = BASE_DIR / "validation_results_with_internal_manual_salvage.csv"
TRAINING_READY_IN = BASE_DIR / "training_ready_after_internal_manual_salvage.csv"
REPAIR_VALIDATION = REPAIR_DIR / "validation_results.csv"
REPAIR_IDEAS = Path("eodhd_output/provider_error_symbol_repair_ideas.json")

VALIDATION_OUT = BASE_DIR / "validation_results_with_provider_symbol_repair.csv"
TRAINING_READY_OUT = BASE_DIR / "training_ready_after_provider_symbol_repair.csv"
SUMMARY_JSON = BASE_DIR / "provider_symbol_repair_summary.json"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("idea_id") or "", row.get("eodhd_symbol") or "", row.get("publication_date") or "")


def safe_repair(row: dict[str, str]) -> bool:
    return (
        row.get("math_validation_status") == "math_reproduced"
        and row.get("training_readiness") == "candidate_low_risk"
        and row.get("fundamentals_type") == "Common Stock"
        and not (row.get("failure_modes") or "").strip()
        and not (row.get("warning_modes") or "").strip()
    )


def write_validation(
    main_rows: list[dict[str, str]],
    safe_repairs_by_idea: dict[str, dict[str, str]],
    repair_ideas_by_idea: dict[str, dict[str, Any]],
) -> None:
    extra = [
        "provider_symbol_repair_status",
        "provider_symbol_repair_original_eodhd_symbol",
        "provider_symbol_repair_reason",
    ]
    fieldnames = list(main_rows[0].keys()) + [field for field in extra if field not in main_rows[0]]
    with VALIDATION_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in main_rows:
            repair = safe_repairs_by_idea.get(row.get("idea_id") or "")
            if repair:
                idea = repair_ideas_by_idea.get(row.get("idea_id") or {}) or {}
                output = {field: repair.get(field, row.get(field, "")) for field in fieldnames}
                output.update(
                    {
                        "provider_symbol_repair_status": "pass",
                        "provider_symbol_repair_original_eodhd_symbol": idea.get("original_eodhd_symbol") or row.get("eodhd_symbol") or "",
                        "provider_symbol_repair_reason": "alternate_eodhd_exchange_symbol_low_risk_math_reproduced",
                    }
                )
                writer.writerow(output)
            else:
                output = dict(row)
                output.setdefault("provider_symbol_repair_status", "")
                output.setdefault("provider_symbol_repair_original_eodhd_symbol", "")
                output.setdefault("provider_symbol_repair_reason", "")
                writer.writerow(output)


def write_training(existing_training: list[dict[str, str]], safe_repairs: list[dict[str, str]]) -> int:
    fieldnames = list(existing_training[0].keys())
    existing_keys = {row_key(row) for row in existing_training}
    now = datetime.now(UTC).isoformat()
    additions: list[dict[str, str]] = []
    for repair in safe_repairs:
        if row_key(repair) in existing_keys:
            continue
        output = {field: "" for field in fieldnames}
        output.update(
            {
                "idea_id": repair.get("idea_id") or "",
                "raw_symbol": repair.get("raw_symbol") or "",
                "eodhd_symbol": repair.get("eodhd_symbol") or "",
                "publication_date": repair.get("publication_date") or "",
                "include_in_training": "true",
                "math_validation_status": "provider_symbol_repaired_math_reproduced",
                "review_stage": "provider_error_symbol_repair_low_risk",
                "training_readiness": "training_ready",
                "review_status": "pass",
                "reviewed_at": now,
                "agent_c_status": "pass",
                "agent_c_reason": "alternate_eodhd_exchange_symbol_low_risk_math_reproduced",
                "agent_c_outcome_type": "provider_symbol_repair",
                "source_count": "4",
                "fundamentals_name": repair.get("fundamentals_name") or "",
                "fundamentals_type": repair.get("fundamentals_type") or "",
                "fundamentals_sector": repair.get("fundamentals_sector") or "",
                "fundamentals_industry": repair.get("fundamentals_industry") or "",
                "fundamentals_market_cap": repair.get("fundamentals_market_cap") or "",
                "fundamentals_revenue_ttm": repair.get("fundamentals_revenue_ttm") or "",
                "fundamentals_profit_margin": repair.get("fundamentals_profit_margin") or "",
                "original_validation_status": "provider_error",
                "original_review_stage": "provider_error_symbol_repair",
                "original_warning_modes": repair.get("warning_modes") or "",
                "original_failure_modes": repair.get("failure_modes") or "",
            }
        )
        for horizon in ("1y", "3y", "5y", "10y", "20y"):
            field = f"validated_perf_{horizon}"
            source = f"perf_{horizon}"
            if field in output:
                output[field] = repair.get(source) or ""
        additions.append(output)

    with TRAINING_READY_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_training)
        writer.writerows(additions)
    return len(additions)


def main() -> int:
    main_rows = load_csv(VALIDATION_IN)
    existing_training = load_csv(TRAINING_READY_IN)
    repair_rows = load_csv(REPAIR_VALIDATION) if REPAIR_VALIDATION.exists() else []
    repair_ideas = json.loads(REPAIR_IDEAS.read_text(encoding="utf-8")) if REPAIR_IDEAS.exists() else []
    repair_ideas_by_idea = {str(row.get("idea_id") or ""): row for row in repair_ideas if isinstance(row, dict)}
    safe_repairs = [row for row in repair_rows if safe_repair(row)]
    safe_repairs_by_idea = {row.get("idea_id") or "": row for row in safe_repairs}

    write_validation(main_rows, safe_repairs_by_idea, repair_ideas_by_idea)
    added = write_training(existing_training, safe_repairs)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_rows": len(main_rows),
        "repair_rows": len(repair_rows),
        "safe_repair_rows": len(safe_repairs),
        "existing_training_ready_rows": len(existing_training),
        "new_training_ready_rows": added,
        "combined_training_ready_rows": len(existing_training) + added,
        "repair_math_status_counts": dict(Counter(row.get("math_validation_status") for row in repair_rows)),
        "outputs": {
            "validation_csv": str(VALIDATION_OUT.resolve()),
            "training_ready_csv": str(TRAINING_READY_OUT.resolve()),
            "summary_json": str(SUMMARY_JSON.resolve()),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
