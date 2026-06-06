#!/usr/bin/env python3
"""Classify zero and severe-loser rows for bankruptcy/delisting outcomes.

This is a diagnostic/enrichment pass. It does not automatically add rows to
training unless the outcome is explicitly modeled elsewhere. The goal is to
separate:

- true possible equity cancellations
- delisted rows that still need shareholder-outcome modeling
- cash/stock/warrant consideration cases that must not be treated as zero
- provider-adjustment or stale-ticker failures

No API keys are used.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from review_extreme_winners_15x import add_years, horizon_years, parse_date, parse_float


BASE_DIR = Path("eodhd_output/full_run")
VALIDATION_CSV = BASE_DIR / "validation_results_with_all_manual_review.csv"
TRAINING_READY_CSV = BASE_DIR / "training_ready_after_manual_review.csv"
OUTPUT_CSV = BASE_DIR / "bankruptcy_delisted_loser_review.csv"
OUTPUT_JSON = BASE_DIR / "bankruptcy_delisted_loser_review.json"

SEVERE_LOSER = 0.05

# These are not broad truth tables. They are source-backed outcome hints for
# rows seen in the current severe-loser set.
KNOWN_OUTCOMES: dict[str, dict[str, Any]] = {
    "SAEX.US": {
        "outcome_type": "bankruptcy_equity_extinguished",
        "effective_date": "2020-12-21",
        "training_interpretation": "zero_after_effective_date_only",
        "notes": "SAExploration emerged from Chapter 11 as private; all prior equity was extinguished and new equity issued to lenders.",
        "sources": [
            {
                "publisher": "SAExploration / GlobeNewswire",
                "url": "https://www.globenewswire.com/news-release/2020/12/21/2148955/34359/en/SAExploration-Successfully-Completes-Financial-Restructuring.html",
                "supports": "all pre-reorganization equity extinguished at emergence",
            },
            {
                "publisher": "SEC Litigation Release",
                "url": "https://www.sec.gov/litigation/litreleases/2020/lr24943.htm",
                "supports": "company had declared bankruptcy in August 2020",
            },
        ],
    },
    "GTT.US": {
        "outcome_type": "bankruptcy_reorganization_with_equityholder_warrants",
        "effective_date": "2022-12-30",
        "training_interpretation": "do_not_treat_as_simple_zero_without_warrant_model",
        "notes": "Existing GTT equity interests were cancelled, but plan documents/coverage describe equityholder warrants, so old common needs consideration modeling.",
        "sources": [
            {
                "publisher": "SEC / restructuring support agreement",
                "url": "https://www.sec.gov/Archives/edgar/data/1315255/000162828021017980/a2ex101-gttxrestructurings.htm",
                "supports": "existing GTT equity interests and equityholder-warrant treatment",
            },
            {
                "publisher": "MarketScreener / S&P Capital IQ",
                "url": "https://www.marketscreener.com/quote/stock/GTT-COMMUNICATIONS-18727033/news/Second-Amended-Third-Modified-Joint-Pre-Packaged-Reorganization-Plan-and-Disclosure-Statement-Approv-42678030/",
                "supports": "plan approval and cancellation language",
            },
        ],
    },
    "WFT.US": {
        "outcome_type": "bankruptcy_reorganization_new_security",
        "effective_date": "2019-12-13",
        "training_interpretation": "do_not_treat_as_simple_zero_without_reorganization_model",
        "notes": "Weatherford emerged from Chapter 11 and later relisted under WFRD; old WFT path requires reorganization/security-modeling.",
        "sources": [
            {
                "publisher": "ICE / NYSE",
                "url": "https://ir.theice.com/press/news-details/2019/NYSE-to-Suspend-Trading-Immediately-in-Weatherford-International-plc-WFT-and-Commence-Delisting-Proceedings/default.aspx",
                "supports": "NYSE suspension/delisting proceedings",
            },
            {
                "publisher": "Weatherford filing",
                "url": "https://weatherford.gcs-web.com/static-files/6b15a314-1abb-45f1-9368-fd16959c7038",
                "supports": "Chapter 11 emergence/reorganization background",
            },
        ],
    },
    "YRCW.US": {
        "outcome_type": "chapter_11_delisting_outcome_unresolved",
        "effective_date": "2023-08-07",
        "training_interpretation": "hold_until_plan_or_shareholder_recovery_known",
        "notes": "Yellow filed Chapter 11 and Nasdaq delisted due to bankruptcy; old-share recovery/cancellation still needs outcome modeling.",
        "sources": [
            {
                "publisher": "Yellow Corporation FAQ",
                "url": "https://investors.myyellow.com/investor-resources/investor-faqs/",
                "supports": "YRCW to YELL ticker change and Nasdaq delisting after Chapter 11 filing",
            }
        ],
    },
    "THQI.US": {
        "outcome_type": "bankruptcy_liquidation",
        "effective_date": "2013-05-01",
        "training_interpretation": "zero_after_liquidation_confirmation_only",
        "notes": "THQ bankruptcy liquidation was approved after asset sales; rows before confirmation still need date-sensitive modeling.",
        "sources": [
            {
                "publisher": "Game Developer",
                "url": "https://www.gamedeveloper.com/business/thq-is-officially-over-as-liquidation-plans-are-approved",
                "supports": "bankruptcy court approved liquidation plan",
            }
        ],
    },
    "ORIG.US": {
        "outcome_type": "acquisition_cash_and_stock",
        "effective_date": "2018-12-05",
        "training_interpretation": "do_not_treat_as_zero_use_merger_consideration",
        "notes": "Ocean Rig was acquired by Transocean for cash plus stock; severe adjusted-price returns after deal need consideration modeling.",
        "sources": [
            {
                "publisher": "Transocean Investor Relations",
                "url": "https://investor.deepwater.com/news-releases/news-release-details/transocean-ltd-announces-agreement-acquire-ocean-rig",
                "supports": "1.6128 Transocean shares plus $12.75 cash per Ocean Rig share",
            }
        ],
    },
    "IEAM.US": {
        "outcome_type": "bankruptcy_common_stock_retained",
        "effective_date": "2013-03-12",
        "training_interpretation": "do_not_treat_as_bankruptcy_zero",
        "notes": "IEAM plan text references retention of common stock; exact-zero price needs separate stale/OTC/provider review.",
        "sources": [
            {
                "publisher": "SEC exhibit",
                "url": "https://www.sec.gov/Archives/edgar/data/1059677/000119312513102430/d500819dex4.htm",
                "supports": "plan references retention of common stock",
            }
        ],
    },
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("idea_id") or "", row.get("eodhd_symbol") or "", row.get("publication_date") or "")


def split_flags(row: dict[str, str]) -> set[str]:
    flags: set[str] = set()
    for column in ("failure_modes", "warning_modes", "manual_review_row_failures", "manual_review_row_warnings"):
        flags.update(part.strip() for part in (row.get(column) or "").replace("|", ";").split(";") if part.strip())
    return flags


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


def target_date(row: dict[str, str]) -> date | None:
    pub_date = parse_date(row.get("publication_date"))
    years = horizon_years(row.get("review_target_horizon"))
    if pub_date is None or years is None:
        return None
    return add_years(pub_date, years)


def classify(row: dict[str, str], status: str) -> dict[str, Any]:
    symbol = row.get("eodhd_symbol") or ""
    multiplier = parse_float(row.get("review_target_multiplier"))
    flags = split_flags(row)
    target = target_date(row)
    outcome = KNOWN_OUTCOMES.get(symbol)
    delisted_date = parse_date(row.get("fundamentals_delisted_date"))
    last_price_date = parse_date(row.get("last_price_date"))

    action = "hold"
    review_status = "manual_review"
    reason = "needs_outcome_evidence"
    modeled_return = None

    if status == "training_ready":
        action = "already_training_ready"
        review_status = "pass"
        reason = "already_passed_prior_review"
    elif status == "rejected":
        action = "already_excluded"
        review_status = "reject"
        reason = "already_rejected_prior_review"
    elif outcome:
        effective = parse_date(outcome.get("effective_date"))
        interpretation = outcome.get("training_interpretation")
        if interpretation in {"zero_after_effective_date_only", "zero_after_liquidation_confirmation_only"}:
            if target and effective and target >= effective:
                action = "candidate_zero_modeled"
                review_status = "pass_candidate"
                reason = f"source_backed_{outcome['outcome_type']}_before_target"
                modeled_return = 0.0
            else:
                action = "hold"
                review_status = "manual_review"
                reason = f"{outcome['outcome_type']}_after_target_or_target_missing"
        elif interpretation == "do_not_treat_as_bankruptcy_zero":
            action = "hold"
            review_status = "manual_review"
            reason = "known_bankruptcy_but_common_stock_retained_or_stale_price"
        elif "do_not_treat_as_zero" in str(interpretation):
            action = "hold"
            review_status = "manual_review"
            reason = f"{outcome['outcome_type']}_requires_consideration_model"
        else:
            action = "hold"
            review_status = "manual_review"
            reason = outcome.get("training_interpretation") or outcome.get("outcome_type")
    elif row.get("fundamentals_is_delisted") == "True" or "symbol_in_delisted_cache" in flags:
        if target and delisted_date and target >= delisted_date:
            action = "hold"
            review_status = "manual_review"
            reason = "delisted_before_target_needs_bankruptcy_acquisition_or_liquidation_source"
        elif target and last_price_date and target >= last_price_date:
            action = "hold"
            review_status = "manual_review"
            reason = "price_history_ended_before_target_needs_shareholder_outcome"
        else:
            action = "hold"
            review_status = "manual_review"
            reason = "delisted_flag_but_target_before_known_delisting"
    elif multiplier == 0:
        action = "exclude_or_hold"
        review_status = "manual_review"
        reason = "exact_zero_without_bankruptcy_or_delisting_outcome_source"
    elif "provider_adjustment_factor_conflict" in flags:
        action = "exclude_or_hold"
        review_status = "manual_review"
        reason = "severe_loser_provider_adjustment_conflict"
    else:
        action = "hold"
        review_status = "manual_review"
        reason = "severe_loser_needs_business_deterioration_or_outcome_source"

    return {
        "idea_id": row.get("idea_id"),
        "raw_symbol": row.get("raw_symbol"),
        "eodhd_symbol": symbol,
        "publication_date": row.get("publication_date"),
        "target_date": target.isoformat() if target else "",
        "horizon": row.get("review_target_horizon"),
        "return_multiplier": multiplier,
        "existing_status": status,
        "review_status": review_status,
        "training_action": action,
        "reason": reason,
        "modeled_return": modeled_return,
        "company_name": row.get("fundamentals_name") or row.get("delisted_provider_name"),
        "fundamentals_is_delisted": row.get("fundamentals_is_delisted"),
        "fundamentals_delisted_date": row.get("fundamentals_delisted_date"),
        "last_price_date": row.get("last_price_date"),
        "warning_modes": row.get("warning_modes"),
        "failure_modes": row.get("failure_modes"),
        "known_outcome_type": (outcome or {}).get("outcome_type"),
        "known_outcome_effective_date": (outcome or {}).get("effective_date"),
        "known_outcome_notes": (outcome or {}).get("notes"),
        "sources": (outcome or {}).get("sources") or [],
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "idea_id",
        "raw_symbol",
        "eodhd_symbol",
        "publication_date",
        "target_date",
        "horizon",
        "return_multiplier",
        "existing_status",
        "review_status",
        "training_action",
        "reason",
        "modeled_return",
        "company_name",
        "fundamentals_is_delisted",
        "fundamentals_delisted_date",
        "last_price_date",
        "known_outcome_type",
        "known_outcome_effective_date",
        "known_outcome_notes",
        "source_count",
        "warning_modes",
        "failure_modes",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: (len(row.get("sources") or []) if field == "source_count" else row.get(field)) for field in fieldnames})


def main() -> int:
    validation_rows = load_csv(VALIDATION_CSV)
    training_rows = load_csv(TRAINING_READY_CSV)
    training_keys = {row_key(row) for row in training_rows}
    severe_rows = [
        row
        for row in validation_rows
        if (parse_float(row.get("review_target_multiplier")) is not None and parse_float(row.get("review_target_multiplier")) <= SEVERE_LOSER)
    ]
    reviews = [classify(row, final_status(row, training_keys)) for row in severe_rows]
    reviews.sort(key=lambda row: (row["return_multiplier"] if row["return_multiplier"] is not None else 1, row["eodhd_symbol"]))
    write_csv(reviews)

    exact_zero = [row for row in reviews if row.get("return_multiplier") == 0]
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "severe_loser_threshold": SEVERE_LOSER,
        "severe_loser_rows": len(reviews),
        "exact_zero_rows": len(exact_zero),
        "existing_status_counts": dict(Counter(row["existing_status"] for row in reviews)),
        "review_status_counts": dict(Counter(row["review_status"] for row in reviews)),
        "training_action_counts": dict(Counter(row["training_action"] for row in reviews)),
        "reason_counts": dict(Counter(row["reason"] for row in reviews).most_common(25)),
        "candidate_zero_modeled_rows": [row for row in reviews if row["training_action"] == "candidate_zero_modeled"],
        "exact_zero_rows_detail": exact_zero,
        "known_outcome_symbols": KNOWN_OUTCOMES,
        "outputs": {
            "csv": str(OUTPUT_CSV.resolve()),
            "json": str(OUTPUT_JSON.resolve()),
        },
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"candidate_zero_modeled_rows", "exact_zero_rows_detail", "known_outcome_symbols"}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
