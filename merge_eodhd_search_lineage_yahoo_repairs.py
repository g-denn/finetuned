#!/usr/bin/env python3
"""Merge EODHD-search lineage repairs validated with Yahoo adjusted prices."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from manual_review_validation import ProviderSeries, add_years
from salvage_partial_horizon_labels import looks_non_common_instrument, scalar
from salvage_provider_error_yahoo_identity import fetch_yahoo_direct, review_horizons


BASE_DIR = Path("eodhd_output/full_run")
REPAIR_DIR = Path("eodhd_output/eodhd_search_lineage_repair")

VALIDATION_IN = BASE_DIR / "validation_results_with_reverse_split_salvage.csv"
TRAINING_READY_IN = BASE_DIR / "training_ready_after_reverse_split_salvage.csv"
REPAIR_VALIDATION = REPAIR_DIR / "validation_results.csv"
REPAIR_IDEAS = Path("eodhd_output/eodhd_search_lineage_repair_ideas.json")

VALIDATION_OUT = BASE_DIR / "validation_results_with_search_lineage_yahoo_repair.csv"
TRAINING_READY_OUT = BASE_DIR / "training_ready_after_search_lineage_yahoo_repair.csv"
ROW_REVIEWS_CSV = BASE_DIR / "search_lineage_yahoo_repair_reviews.csv"
SUMMARY_JSON = BASE_DIR / "search_lineage_yahoo_repair_summary.json"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("idea_id") or "", row.get("eodhd_symbol") or "", row.get("publication_date") or "")


def exchange_hints(raw_symbol: str | None, original_eodhd_symbol: str | None, yahoo_symbol: str | None) -> set[str]:
    combo = f"{raw_symbol or ''} {original_eodhd_symbol or ''} {yahoo_symbol or ''}".upper()
    mapping = {
        ".US": {"US"},
        " US": {"US"},
        ".L": {"LSE"},
        " LN": {"LSE"},
        ".TO": {"TO"},
        " CN": {"TO", "V"},
        ".V": {"V"},
        ".HK": {"HK"},
        " HK": {"HK"},
        ".T": {"TSE", "T"},
        " JP": {"TSE", "T"},
        ".KS": {"KO", "KQ"},
        " KS": {"KO", "KQ"},
        ".DE": {"XETRA", "F"},
        " GR": {"XETRA", "F"},
        " GY": {"XETRA", "F"},
        ".PA": {"PA"},
        " FP": {"PA"},
    }
    hints: set[str] = set()
    for marker, exchanges in mapping.items():
        if marker in combo:
            hints.update(exchanges)
    return hints


def exchange_consistent(repair: dict[str, Any]) -> bool:
    repaired = str(repair.get("eodhd_symbol") or "")
    exchange = repaired.rsplit(".", 1)[-1].upper() if "." in repaired else ""
    original = str(repair.get("original_eodhd_symbol") or "")
    hints = exchange_hints(repair.get("raw_symbol"), original, repair.get("yahoo_symbol"))
    if hints:
        return exchange in hints
    if original.endswith(".US"):
        return exchange == "US"
    return True


def yahoo_symbol_from_eodhd(symbol: str) -> str:
    suffix_map = {
        ".US": "",
        ".LSE": ".L",
        ".TO": ".TO",
        ".V": ".V",
        ".HK": ".HK",
        ".TSE": ".T",
        ".T": ".T",
        ".KO": ".KS",
        ".KQ": ".KQ",
        ".AU": ".AX",
        ".XETRA": ".DE",
        ".F": ".F",
        ".PA": ".PA",
        ".AS": ".AS",
        ".OL": ".OL",
        ".SW": ".SW",
        ".ST": ".ST",
        ".MX": ".MX",
        ".SA": ".SA",
    }
    for eodhd_suffix, yahoo_suffix in suffix_map.items():
        if symbol.endswith(eodhd_suffix):
            code = symbol[: -len(eodhd_suffix)]
            return code + yahoo_suffix
    return symbol


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def search_identity_ok(row: dict[str, str], repair: dict[str, Any]) -> tuple[bool, str, dict[str, str]]:
    """Accept EODHD Search identity when fundamentals were rate-limited.

    This is intentionally narrower than a full fundamentals pass: the Search
    result must be a high-confidence common-stock match on the right exchange,
    and Yahoo still has to pass the price checks before a row is promoted.
    """
    score = repair.get("search_match_score")
    try:
        match_score = float(score)
    except (TypeError, ValueError):
        match_score = 0.0
    search_type = str(repair.get("search_type") or "").strip()
    if search_type.lower() != "common stock":
        return False, "search_identity_not_common_stock", {}
    if match_score < 0.9:
        return False, "search_identity_match_too_weak", {}
    identity_row = {
        **row,
        "fundamentals_type": "Common Stock",
        "fundamentals_name": str(repair.get("search_name") or row.get("fundamentals_name") or ""),
        "company_name": str(repair.get("search_name") or row.get("fundamentals_name") or ""),
        "raw_symbol": str(repair.get("raw_symbol") or row.get("raw_symbol") or ""),
        "eodhd_symbol": str(row.get("eodhd_symbol") or repair.get("eodhd_symbol") or ""),
    }
    if looks_non_common_instrument(identity_row):
        return False, "search_identity_instrument_marker_not_common_stock", {}
    return True, "eodhd_search_common_stock_identity", identity_row


def review_row(row: dict[str, str], repair: dict[str, Any], cache: dict[str, ProviderSeries]) -> dict[str, Any]:
    base = {
        "idea_id": row.get("idea_id"),
        "raw_symbol": repair.get("raw_symbol") or row.get("raw_symbol"),
        "original_eodhd_symbol": repair.get("original_eodhd_symbol"),
        "eodhd_symbol": row.get("eodhd_symbol"),
        "publication_date": row.get("publication_date"),
        "company_name": row.get("fundamentals_name") or repair.get("search_name"),
        "fundamentals_type": row.get("fundamentals_type"),
        "search_match_score": repair.get("search_match_score"),
        "search_name": repair.get("search_name"),
        "identity_source": "",
    }
    if not exchange_consistent(repair):
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": "repaired_exchange_not_consistent_with_original_hint", "passed_horizons": {}}
    if row.get("fundamentals_status") == "fetched":
        if row.get("fundamentals_type") != "Common Stock":
            return {**base, "review_status": "manual_review", "training_action": "hold", "reason": "not_common_stock", "passed_horizons": {}}
        identity_source = "eodhd_fundamentals"
        identity_row = row
    else:
        identity_ok, identity_reason, identity_row = search_identity_ok(row, repair)
        if not identity_ok:
            return {**base, "review_status": "manual_review", "training_action": "hold", "reason": identity_reason, "passed_horizons": {}}
        identity_source = identity_reason
    pub_date = parse_date(row.get("publication_date"))
    if pub_date is None:
        return {**base, "review_status": "manual_review", "training_action": "hold", "reason": "missing_publication_date", "passed_horizons": {}}
    yahoo_symbol = yahoo_symbol_from_eodhd(row.get("eodhd_symbol") or "")
    end = min(add_years(pub_date, 20), date.today())
    cache_key = f"{yahoo_symbol}|{pub_date.isoformat()}|{end.isoformat()}"
    if cache_key not in cache:
        cache[cache_key] = fetch_yahoo_direct(yahoo_symbol, pub_date, end, retries=1)
    series = cache[cache_key]
    if not series.prices:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": "yahoo_price_unavailable_for_repaired_symbol",
            "yahoo_symbol": yahoo_symbol,
            "yahoo_warnings": series.warnings,
            "passed_horizons": {},
        }
    passed, diagnostics, failures = review_horizons(identity_row, series)
    if not passed:
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": ";".join(failures[:5]) or "no_ordinary_horizon_passed",
            "yahoo_symbol": yahoo_symbol,
            "yahoo_rows": len(series.prices),
            "horizon_failures": failures,
            "passed_horizons": {},
        }
    return {
        **base,
        "review_status": "pass",
        "training_action": "add_to_training_ready",
        "reason": "eodhd_search_lineage_identity_yahoo_price_sanity_passed",
        "identity_source": identity_source,
        "yahoo_symbol": yahoo_symbol,
        "yahoo_rows": len(series.prices),
        "passed_horizons": passed,
        "horizon_diagnostics": diagnostics,
        "horizon_failures": failures,
        "confidence": 0.6,
    }


def write_reviews(reviews: list[dict[str, Any]]) -> None:
    fieldnames = [
        "idea_id",
        "raw_symbol",
        "original_eodhd_symbol",
        "eodhd_symbol",
        "publication_date",
        "company_name",
        "review_status",
        "training_action",
        "reason",
        "passed_horizons",
        "yahoo_symbol",
        "yahoo_rows",
        "search_match_score",
        "search_name",
        "identity_source",
        "confidence",
    ]
    with ROW_REVIEWS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for review in reviews:
            writer.writerow({field: scalar(review.get(field)) for field in fieldnames})


def write_validation(
    main_rows: list[dict[str, str]],
    repair_rows_by_idea: dict[str, dict[str, str]],
    reviews_by_idea: dict[str, dict[str, Any]],
) -> None:
    extra = [
        "search_lineage_repair_status",
        "search_lineage_repair_original_eodhd_symbol",
        "search_lineage_repair_reason",
        "search_lineage_repair_passed_horizons",
    ]
    fieldnames = list(main_rows[0].keys()) + [field for field in extra if field not in main_rows[0]]
    with VALIDATION_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in main_rows:
            review = reviews_by_idea.get(row.get("idea_id") or "")
            repair_row = repair_rows_by_idea.get(row.get("idea_id") or "")
            if review and repair_row and review.get("review_status") == "pass":
                output = {field: repair_row.get(field, row.get(field, "")) for field in fieldnames}
                output.update(
                    {
                        "search_lineage_repair_status": "pass",
                        "search_lineage_repair_original_eodhd_symbol": review.get("original_eodhd_symbol") or "",
                        "search_lineage_repair_reason": review.get("reason") or "",
                        "search_lineage_repair_passed_horizons": scalar(review.get("passed_horizons")),
                    }
                )
                writer.writerow(output)
            else:
                output = dict(row)
                output.setdefault("search_lineage_repair_status", "")
                output.setdefault("search_lineage_repair_original_eodhd_symbol", "")
                output.setdefault("search_lineage_repair_reason", "")
                output.setdefault("search_lineage_repair_passed_horizons", "")
                writer.writerow(output)


def write_training(existing_training: list[dict[str, str]], pass_reviews: list[dict[str, Any]], repair_rows_by_idea: dict[str, dict[str, str]]) -> int:
    fieldnames = list(existing_training[0].keys())
    existing_idea_ids = {row.get("idea_id") for row in existing_training}
    now = datetime.now(UTC).isoformat()
    additions: list[dict[str, str]] = []
    for review in pass_reviews:
        idea_id = str(review.get("idea_id") or "")
        if idea_id in existing_idea_ids:
            continue
        repair = repair_rows_by_idea.get(idea_id) or {}
        output = {field: "" for field in fieldnames}
        output.update(
            {
                "idea_id": idea_id,
                "raw_symbol": scalar(review.get("raw_symbol")),
                "eodhd_symbol": scalar(review.get("eodhd_symbol")),
                "publication_date": scalar(review.get("publication_date")),
                "include_in_training": "true",
                "math_validation_status": "eodhd_search_lineage_yahoo_verified",
                "review_stage": "eodhd_search_lineage_yahoo_price",
                "training_readiness": "training_ready",
                "review_status": "pass",
                "reviewed_at": now,
                "agent_b_yahoo_symbol": scalar(review.get("yahoo_symbol")),
                "agent_b_yahoo_rows": scalar(review.get("yahoo_rows")),
                "agent_c_status": "pass",
                "agent_c_reason": scalar(review.get("reason")),
                "agent_c_outcome_type": scalar(review.get("identity_source") or "eodhd_search_lineage_yahoo_price"),
                "source_count": "4" if review.get("identity_source") == "eodhd_fundamentals" else "3",
                "fundamentals_name": repair.get("fundamentals_name") or scalar(review.get("company_name")),
                "fundamentals_type": repair.get("fundamentals_type") or scalar(review.get("fundamentals_type")) or "Common Stock",
                "fundamentals_sector": repair.get("fundamentals_sector") or "",
                "fundamentals_industry": repair.get("fundamentals_industry") or "",
                "fundamentals_market_cap": repair.get("fundamentals_market_cap") or "",
                "fundamentals_revenue_ttm": repair.get("fundamentals_revenue_ttm") or "",
                "fundamentals_profit_margin": repair.get("fundamentals_profit_margin") or "",
                "original_validation_status": "lineage_repair",
                "original_review_stage": "eodhd_search_lineage_yahoo_price",
            }
        )
        for horizon, value in (review.get("passed_horizons") or {}).items():
            field = f"validated_perf_{horizon}"
            if field in output:
                output[field] = scalar(value)
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
    repair_ideas_by_id = {str(row.get("idea_id") or ""): row for row in repair_ideas if isinstance(row, dict)}
    repair_rows_by_idea = {row.get("idea_id") or "": row for row in repair_rows}
    cache: dict[str, ProviderSeries] = {}
    reviews = [
        review_row(row, repair_ideas_by_id.get(row.get("idea_id") or "") or {}, cache)
        for row in repair_rows
    ]
    pass_reviews = [review for review in reviews if review.get("review_status") == "pass"]
    reviews_by_idea = {str(review.get("idea_id") or ""): review for review in reviews}
    write_reviews(reviews)
    write_validation(main_rows, repair_rows_by_idea, reviews_by_idea)
    added = write_training(existing_training, pass_reviews, repair_rows_by_idea)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_rows": len(main_rows),
        "repair_rows": len(repair_rows),
        "safe_repair_rows": len(pass_reviews),
        "existing_training_ready_rows": len(existing_training),
        "new_training_ready_rows": added,
        "combined_training_ready_rows": len(existing_training) + added,
        "review_status_counts": dict(Counter(review.get("review_status") for review in reviews)),
        "training_action_counts": dict(Counter(review.get("training_action") for review in reviews)),
        "top_reasons": dict(Counter(str(review.get("reason")) for review in reviews).most_common(25)),
        "outputs": {
            "row_reviews_csv": str(ROW_REVIEWS_CSV.resolve()),
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
