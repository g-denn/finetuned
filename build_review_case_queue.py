#!/usr/bin/env python3
"""Build a case-level review queue from performance validation outputs.

The row queue is useful for auditability, but it is a painful way to work.
This script groups unresolved rows into issuer/security cases so we can resolve
families of rows at once:

- data-repair cases: provider errors and math-incomplete rows
- corporate-action cases: delisted names, reverse splits, acquisitions
- manual-review cases: provider disagreement, ordinary business-quality review,
  and held extreme winners

No API keys are used or stored here.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path("eodhd_output/full_run")
VALIDATION_CSV = (
    BASE_DIR / "validation_results_with_sec_yahoo_salvage.csv"
    if (BASE_DIR / "validation_results_with_sec_yahoo_salvage.csv").exists()
    else BASE_DIR / "validation_results_with_active_ordinary_salvage.csv"
    if (BASE_DIR / "validation_results_with_active_ordinary_salvage.csv").exists()
    else BASE_DIR / "validation_results_with_delisted_archive_extended_repair.csv"
    if (BASE_DIR / "validation_results_with_delisted_archive_extended_repair.csv").exists()
    else BASE_DIR / "validation_results_with_large_winner_salvage.csv"
    if (BASE_DIR / "validation_results_with_large_winner_salvage.csv").exists()
    else BASE_DIR / "validation_results_with_severe_loser_salvage.csv"
    if (BASE_DIR / "validation_results_with_severe_loser_salvage.csv").exists()
    else BASE_DIR / "validation_results_with_split_event_salvage.csv"
    if (BASE_DIR / "validation_results_with_split_event_salvage.csv").exists()
    else BASE_DIR / "validation_results_with_search_lineage_yahoo_repair.csv"
    if (BASE_DIR / "validation_results_with_search_lineage_yahoo_repair.csv").exists()
    else BASE_DIR / "validation_results_with_reverse_split_salvage.csv"
    if (BASE_DIR / "validation_results_with_reverse_split_salvage.csv").exists()
    else BASE_DIR / "validation_results_with_delisted_archive_symbol_repair.csv"
    if (BASE_DIR / "validation_results_with_delisted_archive_symbol_repair.csv").exists()
    else BASE_DIR / "validation_results_with_provider_error_yahoo_identity.csv"
    if (BASE_DIR / "validation_results_with_provider_error_yahoo_identity.csv").exists()
    else BASE_DIR / "validation_results_with_provider_symbol_repair.csv"
    if (BASE_DIR / "validation_results_with_provider_symbol_repair.csv").exists()
    else BASE_DIR / "validation_results_with_internal_manual_salvage.csv"
    if (BASE_DIR / "validation_results_with_internal_manual_salvage.csv").exists()
    else BASE_DIR / "validation_results_with_partial_horizon_salvage.csv"
    if (BASE_DIR / "validation_results_with_partial_horizon_salvage.csv").exists()
    else BASE_DIR / "validation_results_with_all_manual_review.csv"
    if (BASE_DIR / "validation_results_with_all_manual_review.csv").exists()
    else BASE_DIR / "validation_results_with_extreme_review.csv"
)
TRAINING_READY_CSV = (
    BASE_DIR / "training_ready_after_sec_yahoo_salvage.csv"
    if (BASE_DIR / "training_ready_after_sec_yahoo_salvage.csv").exists()
    else BASE_DIR / "training_ready_after_active_ordinary_salvage.csv"
    if (BASE_DIR / "training_ready_after_active_ordinary_salvage.csv").exists()
    else BASE_DIR / "training_ready_after_delisted_archive_extended_repair.csv"
    if (BASE_DIR / "training_ready_after_delisted_archive_extended_repair.csv").exists()
    else BASE_DIR / "training_ready_after_large_winner_salvage.csv"
    if (BASE_DIR / "training_ready_after_large_winner_salvage.csv").exists()
    else BASE_DIR / "training_ready_after_severe_loser_salvage.csv"
    if (BASE_DIR / "training_ready_after_severe_loser_salvage.csv").exists()
    else BASE_DIR / "training_ready_after_split_event_salvage.csv"
    if (BASE_DIR / "training_ready_after_split_event_salvage.csv").exists()
    else BASE_DIR / "training_ready_after_search_lineage_yahoo_repair.csv"
    if (BASE_DIR / "training_ready_after_search_lineage_yahoo_repair.csv").exists()
    else BASE_DIR / "training_ready_after_reverse_split_salvage.csv"
    if (BASE_DIR / "training_ready_after_reverse_split_salvage.csv").exists()
    else BASE_DIR / "training_ready_after_delisted_archive_symbol_repair.csv"
    if (BASE_DIR / "training_ready_after_delisted_archive_symbol_repair.csv").exists()
    else BASE_DIR / "training_ready_after_provider_error_yahoo_identity.csv"
    if (BASE_DIR / "training_ready_after_provider_error_yahoo_identity.csv").exists()
    else BASE_DIR / "training_ready_after_provider_symbol_repair.csv"
    if (BASE_DIR / "training_ready_after_provider_symbol_repair.csv").exists()
    else BASE_DIR / "training_ready_after_internal_manual_salvage.csv"
    if (BASE_DIR / "training_ready_after_internal_manual_salvage.csv").exists()
    else BASE_DIR / "training_ready_after_partial_horizon_salvage.csv"
    if (BASE_DIR / "training_ready_after_partial_horizon_salvage.csv").exists()
    else BASE_DIR / "training_ready_after_manual_review.csv"
    if (BASE_DIR / "training_ready_after_manual_review.csv").exists()
    else BASE_DIR / "training_ready_with_extreme_15x.csv"
)
CASE_QUEUE_CSV = BASE_DIR / "review_case_queue.csv"
BACKLOG_SUMMARY_JSON = BASE_DIR / "review_backlog_summary.json"


def parse_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def split_flags(row: dict[str, str]) -> set[str]:
    flags: set[str] = set()
    for column in ("failure_modes", "warning_modes", "manual_review_row_failures", "manual_review_row_warnings"):
        flags.update(part.strip() for part in (row.get(column) or "").replace("|", ";").split(";") if part.strip())
    return flags


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("idea_id") or "", row.get("eodhd_symbol") or "", row.get("publication_date") or "")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def final_status(row: dict[str, str], training_keys: set[tuple[str, str, str]]) -> str:
    if row_key(row) in training_keys:
        return "training_ready"
    if (
        row.get("manual_review_status") == "reject"
        or row.get("training_readiness") == "rejected"
        or row.get("extreme_15x_review_status") == "reject"
        or row.get("remaining_manual_review_status") == "reject"
    ):
        return "rejected"
    if row.get("math_validation_status") == "provider_error":
        return "provider_error"
    if row.get("math_validation_status") == "math_incomplete":
        return "math_incomplete"
    return "manual_review_remaining"


def backlog_bucket(row: dict[str, str], status: str) -> str:
    if status in {"training_ready", "rejected", "provider_error", "math_incomplete"}:
        return status
    flags = split_flags(row)
    multiplier = parse_float(row.get("review_target_multiplier"))
    if multiplier is not None and multiplier >= 15:
        return "extreme_15x_held"
    if "symbol_in_delisted_cache" in flags or "fundamentals_is_delisted" in flags:
        return "delisted_or_delisting"
    if "reverse_split_provider_adjusted" in flags:
        return "reverse_split"
    if "provider_adjustment_factor_conflict" in flags or "provider_disagreement" in flags:
        return "provider_disagreement"
    if "yahoo_cross_check_unavailable" in flags or row.get("agent_b_yahoo_rows") in ("", None, "0"):
        return "needs_cross_provider"
    if row.get("fundamentals_has_financials") != "True":
        return "needs_fundamentals_or_identity"
    return "ordinary_business_quality_review"


def priority_for_bucket(bucket: str) -> int:
    return {
        "extreme_15x_held": 10,
        "provider_disagreement": 20,
        "needs_cross_provider": 30,
        "ordinary_business_quality_review": 40,
        "needs_fundamentals_or_identity": 50,
        "delisted_or_delisting": 60,
        "reverse_split": 70,
        "math_incomplete": 80,
        "provider_error": 90,
    }.get(bucket, 99)


def next_action(bucket: str) -> str:
    return {
        "extreme_15x_held": "Resolve corporate-action/identity block first; if clean, add sourced business-quality evidence and promote/reject as a case.",
        "provider_disagreement": "Compare EODHD adjusted prices against Yahoo/secondary provider; reject bad adjusted data unless a corporate-action model explains it.",
        "needs_cross_provider": "Retry Yahoo/secondary provider cross-check, then rerun manual verifier.",
        "ordinary_business_quality_review": "Use fundamentals plus browser/SEC/IR sources; pass only if revenue/profit/market-cap story supports the return.",
        "needs_fundamentals_or_identity": "Pull filtered fundamentals/search data and resolve common-stock identity before price review.",
        "delisted_or_delisting": "Use delisted archive, fundamentals delisted date, SEC/company notices, and acquisition/bankruptcy outcome modeling.",
        "reverse_split": "Pull splits and raw/adjusted EOD around split dates; reject provider-adjustment discontinuities.",
        "math_incomplete": "Repair endpoint/lineage/date-range data first; do not manually promote.",
        "provider_error": "Retry/search/lineage repair first; do not manually review until a valid symbol history is cached.",
    }.get(bucket, "Review manually.")


def doc_endpoints(bucket: str) -> str:
    common = [
        "fundamentals v1.1 with filters",
        "search/{QUERY}",
    ]
    mapping = {
        "extreme_15x_held": common + ["exchange-symbol-list/{EXCHANGE}?delisted=1", "eod", "splits", "dividends"],
        "provider_disagreement": ["eod", "splits", "dividends", "Yahoo/secondary provider"],
        "needs_cross_provider": ["Yahoo/secondary provider", "eod"],
        "ordinary_business_quality_review": common + ["SEC/issuer IR browser sources"],
        "needs_fundamentals_or_identity": common + ["exchange-symbol-list/{EXCHANGE}?delisted=1"],
        "delisted_or_delisting": common + ["exchange-symbol-list/{EXCHANGE}?delisted=1", "symbol-change-history", "SEC/issuer notices"],
        "reverse_split": ["splits", "eod", "dividends", "Yahoo/secondary provider"],
        "math_incomplete": ["eod", "exchange-symbol-list/{EXCHANGE}?delisted=1", "search/{QUERY}", "symbol-change-history"],
        "provider_error": ["search/{QUERY}", "exchange-symbol-list/{EXCHANGE}?delisted=1", "eod retry"],
    }
    return "; ".join(mapping.get(bucket, common))


def case_key(row: dict[str, str], bucket: str) -> tuple[str, str]:
    symbol = row.get("eodhd_symbol") or row.get("raw_symbol") or "UNKNOWN"
    if bucket in {"delisted_or_delisting", "math_incomplete", "provider_error"}:
        isin = row.get("delisted_provider_isin") or ""
        provider_name = row.get("delisted_provider_name") or row.get("fundamentals_name") or ""
        return (bucket, f"{symbol}|{isin}|{provider_name}")
    return (bucket, symbol)


def scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_case_record(bucket: str, key: str, rows: list[dict[str, str]]) -> dict[str, str]:
    multipliers = [parse_float(row.get("review_target_multiplier")) for row in rows]
    multipliers = [value for value in multipliers if value is not None]
    max_multiplier = max(multipliers) if multipliers else None
    max_row = max(rows, key=lambda row: parse_float(row.get("review_target_multiplier")) or -1)
    flags = sorted({flag for row in rows for flag in split_flags(row)})
    publication_dates = sorted(row.get("publication_date") or "" for row in rows if row.get("publication_date"))
    raw_symbols = sorted({row.get("raw_symbol") or "" for row in rows if row.get("raw_symbol")})
    eodhd_symbols = sorted({row.get("eodhd_symbol") or "" for row in rows if row.get("eodhd_symbol")})
    idea_ids = [row.get("idea_id") or "" for row in rows if row.get("idea_id")]
    return {
        "case_id": f"{priority_for_bucket(bucket):02d}-{bucket}-{key}".replace(",", "_"),
        "priority": str(priority_for_bucket(bucket)),
        "bucket": bucket,
        "row_count": str(len(rows)),
        "unique_idea_count": str(len(set(idea_ids))),
        "eodhd_symbols": ";".join(eodhd_symbols),
        "raw_symbols": ";".join(raw_symbols),
        "company_name": max_row.get("fundamentals_name") or max_row.get("delisted_provider_name") or "",
        "fundamentals_type": max_row.get("fundamentals_type") or "",
        "fundamentals_exchange": max_row.get("fundamentals_exchange") or "",
        "fundamentals_is_delisted": max_row.get("fundamentals_is_delisted") or "",
        "is_in_delisted_cache": max_row.get("is_in_delisted_cache") or "",
        "max_multiplier": scalar(max_multiplier),
        "max_horizon": max_row.get("review_target_horizon") or "",
        "publication_date_min": publication_dates[0] if publication_dates else "",
        "publication_date_max": publication_dates[-1] if publication_dates else "",
        "first_price_date_min": min([row.get("first_price_date") or "" for row in rows if row.get("first_price_date")] or [""]),
        "last_price_date_max": max([row.get("last_price_date") or "" for row in rows if row.get("last_price_date")] or [""]),
        "blocking_flags": ";".join(flags),
        "next_action": next_action(bucket),
        "docs_to_use": doc_endpoints(bucket),
        "sample_idea_ids": ";".join(idea_ids[:5]),
    }


def main() -> int:
    rows = load_csv(VALIDATION_CSV)
    training_rows = load_csv(TRAINING_READY_CSV)
    training_keys = {row_key(row) for row in training_rows}

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    final_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    bucket_symbols: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        status = final_status(row, training_keys)
        final_counts[status] += 1
        if status in {"training_ready", "rejected"}:
            continue
        bucket = backlog_bucket(row, status)
        bucket_counts[bucket] += 1
        bucket_symbols[bucket].add(row.get("eodhd_symbol") or row.get("raw_symbol") or "")
        grouped[case_key(row, bucket)].append(row)

    case_records = [
        build_case_record(bucket, key, case_rows)
        for (bucket, key), case_rows in grouped.items()
    ]
    case_records.sort(
        key=lambda record: (
            int(record["priority"]),
            -float(record["max_multiplier"] or 0),
            -int(record["row_count"]),
            record["case_id"],
        )
    )

    fieldnames = list(case_records[0].keys()) if case_records else []
    with CASE_QUEUE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_records)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_rows": len(rows),
        "final_counts": dict(final_counts),
        "unresolved_rows": sum(bucket_counts.values()),
        "unresolved_cases": len(case_records),
        "bucket_rows": dict(bucket_counts),
        "bucket_cases": dict(Counter(record["bucket"] for record in case_records)),
        "bucket_unique_symbols": {bucket: len(symbols) for bucket, symbols in bucket_symbols.items()},
        "outputs": {
            "case_queue_csv": str(CASE_QUEUE_CSV.resolve()),
            "backlog_summary_json": str(BACKLOG_SUMMARY_JSON.resolve()),
        },
    }
    BACKLOG_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
