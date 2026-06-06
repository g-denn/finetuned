#!/usr/bin/env python3
"""Review remaining manual-review rows after the extreme-winner pass.

This pass handles the 2,688 true manual-review rows with a deterministic
case policy:

- common-stock rows can pass only with target-horizon Yahoo agreement and
  cached EODHD fundamentals/corporate-action evidence
- non-common instruments are excluded from common-stock pitch training
- delisted rows pass only if the verified horizon endpoint precedes the
  delisting date; otherwise they stay held for shareholder-outcome modeling
- reverse-split rows pass only when EODHD split data exists and Yahoo agrees
- extreme winners/losers need business-reality support from financial reports

No network/API token is required; this consumes the cached evidence already
pulled by the EODHD/Yahoo review pipeline.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from review_extreme_winners_15x import (
    DO_NOT_AUTO_PROMOTE,
    FUNDAMENTALS_CACHE,
    TRAINING_READY_OUT as TRAINING_READY_IN,
    VALIDATION_WITH_EXTREME_CSV as VALIDATION_CSV,
    add_years,
    currency,
    horizon_years,
    load_fundamentals,
    parse_date,
    parse_float,
    ratio_text,
    source_list,
    yearly_income_records,
)


BASE_DIR = Path("eodhd_output/full_run")
SYMBOL_CACHE = BASE_DIR / "symbol_cache"
REVIEWS_JSONL = BASE_DIR / "math_reproduced_manual_reviews.jsonl"

ROW_REVIEWS_CSV = BASE_DIR / "remaining_manual_row_reviews.csv"
CASE_REVIEWS_JSONL = BASE_DIR / "remaining_manual_case_reviews.jsonl"
VALIDATION_OUT = BASE_DIR / "validation_results_with_all_manual_review.csv"
TRAINING_READY_OUT = BASE_DIR / "training_ready_after_manual_review.csv"
SUMMARY_JSON = BASE_DIR / "remaining_manual_review_summary.json"

YAHOO_DIFF_LIMIT = 0.15
EXTREME_WINNER = 15.0
LARGE_WINNER_REQUIRES_BUSINESS_EVIDENCE = 10.0
SEVERE_LOSER = 0.05


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("idea_id") or "", row.get("eodhd_symbol") or "", row.get("publication_date") or "")


def review_key(row: dict[str, Any]) -> str:
    return "|".join([str(row.get("idea_id") or ""), str(row.get("eodhd_symbol") or ""), str(row.get("publication_date") or "")])


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def split_flags(row: dict[str, str]) -> set[str]:
    flags: set[str] = set()
    for column in ("failure_modes", "warning_modes", "manual_review_row_failures", "manual_review_row_warnings"):
        flags.update(part.strip() for part in (row.get(column) or "").replace("|", ";").split(";") if part.strip())
    return flags


def load_prior_reviews(path: Path) -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return reviews
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = review_key(review)
            if key.strip("|"):
                reviews[key] = review
    return reviews


def final_status(row: dict[str, str], training_keys: set[tuple[str, str, str]]) -> str:
    if row_key(row) in training_keys:
        return "training_ready"
    if (
        row.get("manual_review_status") == "reject"
        or row.get("training_readiness") == "rejected"
        or row.get("extreme_15x_review_status") == "reject"
    ):
        return "rejected"
    if row.get("math_validation_status") == "provider_error":
        return "provider_error"
    if row.get("math_validation_status") == "math_incomplete":
        return "math_incomplete"
    return "manual_review_remaining"


def safe_symbol_filename(symbol: str) -> str:
    return symbol.replace("/", "-") + ".json"


def load_symbol_cache(symbol: str) -> dict[str, Any]:
    path = SYMBOL_CACHE / safe_symbol_filename(symbol)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def target_horizon_review(row: dict[str, str], prior_review: dict[str, Any] | None) -> tuple[bool, str, dict[str, Any]]:
    if not prior_review:
        return False, "missing_prior_manual_review_record", {}
    horizon = row.get("review_target_horizon") or ""
    horizon_review = (prior_review.get("horizon_reviews") or {}).get(horizon) or {}
    if not horizon_review:
        return False, "missing_target_horizon_review", {}
    yahoo = horizon_review.get("yahoo") or {}
    if not yahoo:
        return False, "missing_yahoo_target_horizon_cross_check", horizon_review
    relative_diff = parse_float(horizon_review.get("relative_diff"))
    if relative_diff is None:
        return False, "missing_yahoo_relative_diff", horizon_review
    if relative_diff > YAHOO_DIFF_LIMIT:
        return False, f"provider_disagreement:{relative_diff:.3f}", horizon_review
    if horizon_review.get("verdict") not in {"pass", "manual_review"}:
        return False, f"target_horizon_not_reproduced:{horizon_review.get('verdict')}", horizon_review
    return True, "target_horizon_cross_provider_reproduced", horizon_review


def nearest_record(records: list[Any], target: date) -> Any | None:
    if not records:
        return None
    return sorted(records, key=lambda record: abs((record.day - target).days))[0]


def financial_evidence(row: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    pub_date = parse_date(row.get("publication_date"))
    years = horizon_years(row.get("review_target_horizon"))
    records = yearly_income_records(payload)
    if pub_date is None or years is None or not records:
        return {"has_financial_evidence": False}
    start = nearest_record(records, pub_date)
    end = nearest_record(records, add_years(pub_date, years))
    if not start or not end:
        return {"has_financial_evidence": False}
    revenue_growth = None
    if start.revenue and start.revenue > 0 and end.revenue and end.revenue > 0:
        revenue_growth = end.revenue / start.revenue
    net_income_growth = None
    if start.net_income and start.net_income > 0 and end.net_income and end.net_income > 0:
        net_income_growth = end.net_income / start.net_income
    end_margin = None
    if end.revenue and end.revenue > 0 and end.net_income is not None:
        end_margin = end.net_income / end.revenue
    return {
        "has_financial_evidence": True,
        "start_financial_date": start.day.isoformat(),
        "end_financial_date": end.day.isoformat(),
        "start_revenue": start.revenue,
        "end_revenue": end.revenue,
        "start_net_income": start.net_income,
        "end_net_income": end.net_income,
        "revenue_growth": revenue_growth,
        "net_income_growth": net_income_growth,
        "end_net_margin": end_margin,
    }


def winner_supported(evidence: dict[str, Any]) -> bool:
    revenue_growth = evidence.get("revenue_growth")
    net_income_growth = evidence.get("net_income_growth")
    end_net_income = evidence.get("end_net_income")
    end_margin = evidence.get("end_net_margin")
    start_net_income = evidence.get("start_net_income")
    if not evidence.get("has_financial_evidence") or (end_net_income or 0) <= 0:
        return False
    margin_ok = end_margin is not None and end_margin >= 0.05
    strong_profit = net_income_growth is not None and net_income_growth >= 3.0
    loss_to_profit = start_net_income is not None and start_net_income <= 0 and end_net_income > 0
    strong_revenue = revenue_growth is not None and revenue_growth >= 3.0
    buyback_style = revenue_growth is not None and revenue_growth >= 2.5 and strong_profit
    return margin_ok and (strong_revenue or buyback_style or loss_to_profit)


def loser_supported(evidence: dict[str, Any]) -> bool:
    if not evidence.get("has_financial_evidence"):
        return False
    revenue_growth = evidence.get("revenue_growth")
    start_net_income = evidence.get("start_net_income")
    end_net_income = evidence.get("end_net_income")
    end_margin = evidence.get("end_net_margin")
    revenue_decline = revenue_growth is not None and revenue_growth <= 0.75
    profit_collapse = start_net_income is not None and start_net_income > 0 and (end_net_income or 0) <= start_net_income * 0.5
    unprofitable = (end_net_income or 0) < 0 or (end_margin is not None and end_margin < 0)
    return revenue_decline or profit_collapse or unprofitable


def delisting_endpoint_safe(row: dict[str, str], horizon_review: dict[str, Any]) -> tuple[bool, str]:
    delisted_date = parse_date(row.get("fundamentals_delisted_date"))
    if not delisted_date:
        return False, "delisted_without_modeled_delisted_date"
    eodhd = horizon_review.get("eodhd") or {}
    endpoint_date = parse_date(eodhd.get("endpoint_trade_date"))
    if not endpoint_date:
        return False, "missing_eodhd_endpoint_date_for_delisted_row"
    if endpoint_date < delisted_date:
        return True, "verified_endpoint_precedes_delisting"
    return False, "endpoint_after_or_on_delisting_requires_shareholder_outcome_model"


def split_evidence_safe(row: dict[str, str]) -> tuple[bool, str]:
    symbol_cache = load_symbol_cache(row.get("eodhd_symbol") or "")
    splits = symbol_cache.get("splits") or []
    if not splits:
        return False, "reverse_split_flag_without_cached_split_events"
    return True, f"cached_split_events:{len(splits)}"


def summary_text(row: dict[str, str], evidence: dict[str, Any], verdict_reason: str) -> str:
    multiplier = row.get("review_target_multiplier") or "n/a"
    horizon = row.get("review_target_horizon") or "n/a"
    name = row.get("fundamentals_name") or row.get("eodhd_symbol") or row.get("raw_symbol")
    if not evidence.get("has_financial_evidence"):
        return f"{name}: {multiplier}x over {horizon}; {verdict_reason}."
    return (
        f"{name}: {multiplier}x over {horizon}; revenue "
        f"{currency(evidence.get('start_revenue'))} to {currency(evidence.get('end_revenue'))} "
        f"({ratio_text(evidence.get('revenue_growth'))}), net income "
        f"{currency(evidence.get('start_net_income'))} to {currency(evidence.get('end_net_income'))}; "
        f"{verdict_reason}."
    )


def review_row(row: dict[str, str], prior_review: dict[str, Any] | None) -> dict[str, Any]:
    symbol = row.get("eodhd_symbol") or ""
    flags = split_flags(row)
    multiplier = parse_float(row.get("review_target_multiplier"))
    fundamentals = load_fundamentals(symbol)
    payload = fundamentals["_payload"]
    sources = source_list(symbol, payload, fundamentals["_cache_path"])
    evidence = financial_evidence(row, payload)

    base = {
        "idea_id": row.get("idea_id"),
        "raw_symbol": row.get("raw_symbol"),
        "eodhd_symbol": symbol,
        "publication_date": row.get("publication_date"),
        "company_name": row.get("fundamentals_name"),
        "horizon": row.get("review_target_horizon"),
        "return_multiplier": multiplier,
        "fundamentals_type": row.get("fundamentals_type"),
        "fundamentals_is_delisted": row.get("fundamentals_is_delisted"),
        "warning_modes": row.get("warning_modes"),
        "failure_modes": row.get("failure_modes"),
        "sources": sources,
        "financial_evidence": evidence,
    }

    if row.get("fundamentals_type") != "Common Stock":
        reason = "non_common_instrument_excluded_from_common_stock_training"
        return {
            **base,
            "review_status": "reject",
            "training_action": "exclude",
            "reason": reason,
            "confidence": 0.9,
            "qualitative_summary": summary_text(row, evidence, reason),
        }

    if symbol in DO_NOT_AUTO_PROMOTE:
        reason = DO_NOT_AUTO_PROMOTE[symbol]
        return {
            **base,
            "review_status": "manual_review",
            "training_action": "hold",
            "reason": reason,
            "confidence": 0.45,
            "qualitative_summary": summary_text(row, evidence, reason),
        }

    cross_ok, cross_reason, horizon_review = target_horizon_review(row, prior_review)
    if not cross_ok:
        status = "reject" if cross_reason.startswith("provider_disagreement:") else "manual_review"
        action = "exclude" if status == "reject" else "hold"
        return {
            **base,
            "review_status": status,
            "training_action": action,
            "reason": cross_reason,
            "confidence": 0.85 if status == "reject" else 0.35,
            "qualitative_summary": summary_text(row, evidence, cross_reason),
        }

    if "reverse_split_provider_adjusted" in flags:
        split_ok, split_reason = split_evidence_safe(row)
        if not split_ok:
            return {
                **base,
                "review_status": "manual_review",
                "training_action": "hold",
                "reason": split_reason,
                "confidence": 0.4,
                "qualitative_summary": summary_text(row, evidence, split_reason),
            }
        sources.append(
            {
                "publisher": "EODHD cached splits",
                "url": str((SYMBOL_CACHE / safe_symbol_filename(symbol)).resolve()),
                "supports": split_reason,
            }
        )

    if row.get("fundamentals_is_delisted") == "True" or "symbol_in_delisted_cache" in flags:
        safe, delist_reason = delisting_endpoint_safe(row, horizon_review)
        if not safe:
            return {
                **base,
                "review_status": "manual_review",
                "training_action": "hold",
                "reason": delist_reason,
                "confidence": 0.4,
                "qualitative_summary": summary_text(row, evidence, delist_reason),
            }

    if multiplier is not None and multiplier >= LARGE_WINNER_REQUIRES_BUSINESS_EVIDENCE:
        if not winner_supported(evidence):
            reason = (
                "extreme_winner_business_quality_not_sufficiently_supported"
                if multiplier >= EXTREME_WINNER
                else "large_winner_business_quality_not_sufficiently_supported"
            )
            return {
                **base,
                "review_status": "manual_review",
                "training_action": "hold",
                "reason": reason,
                "confidence": 0.55,
                "qualitative_summary": summary_text(row, evidence, reason),
            }
    elif multiplier is not None and multiplier <= SEVERE_LOSER:
        if not loser_supported(evidence):
            reason = "severe_loser_business_deterioration_not_sufficiently_supported"
            return {
                **base,
                "review_status": "manual_review",
                "training_action": "hold",
                "reason": reason,
                "confidence": 0.55,
                "qualitative_summary": summary_text(row, evidence, reason),
            }

    reason = "identity_price_corporate_action_and_business_reality_passed"
    return {
        **base,
        "review_status": "pass",
        "training_action": "add_to_training_ready",
        "reason": reason,
        "confidence": 0.78,
        "horizon_review": horizon_review,
        "qualitative_summary": summary_text(row, evidence, reason),
    }


def passed_horizon_returns(review: dict[str, Any] | None) -> dict[str, Any]:
    returns: dict[str, Any] = {}
    if not review:
        return returns
    for horizon, result in (review.get("horizon_reviews") or {}).items():
        if result.get("verdict") != "pass":
            continue
        eodhd = result.get("eodhd") or {}
        returns[horizon] = eodhd.get("multiplier")
    return returns


def scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def write_row_reviews(reviews: list[dict[str, Any]]) -> None:
    fieldnames = [
        "idea_id",
        "raw_symbol",
        "eodhd_symbol",
        "publication_date",
        "company_name",
        "horizon",
        "return_multiplier",
        "review_status",
        "training_action",
        "reason",
        "confidence",
        "fundamentals_type",
        "fundamentals_is_delisted",
        "revenue_growth",
        "net_income_growth",
        "start_revenue",
        "end_revenue",
        "start_net_income",
        "end_net_income",
        "source_count",
        "qualitative_summary",
        "warning_modes",
        "failure_modes",
    ]
    with ROW_REVIEWS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for review in reviews:
            evidence = review.get("financial_evidence") or {}
            writer.writerow(
                {
                    "idea_id": review.get("idea_id"),
                    "raw_symbol": review.get("raw_symbol"),
                    "eodhd_symbol": review.get("eodhd_symbol"),
                    "publication_date": review.get("publication_date"),
                    "company_name": review.get("company_name"),
                    "horizon": review.get("horizon"),
                    "return_multiplier": review.get("return_multiplier"),
                    "review_status": review.get("review_status"),
                    "training_action": review.get("training_action"),
                    "reason": review.get("reason"),
                    "confidence": review.get("confidence"),
                    "fundamentals_type": review.get("fundamentals_type"),
                    "fundamentals_is_delisted": review.get("fundamentals_is_delisted"),
                    "revenue_growth": evidence.get("revenue_growth"),
                    "net_income_growth": evidence.get("net_income_growth"),
                    "start_revenue": evidence.get("start_revenue"),
                    "end_revenue": evidence.get("end_revenue"),
                    "start_net_income": evidence.get("start_net_income"),
                    "end_net_income": evidence.get("end_net_income"),
                    "source_count": len(review.get("sources") or []),
                    "qualitative_summary": review.get("qualitative_summary"),
                    "warning_modes": review.get("warning_modes"),
                    "failure_modes": review.get("failure_modes"),
                }
            )


def write_case_reviews(reviews: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for review in reviews:
        key = (str(review.get("training_action")), str(review.get("eodhd_symbol")))
        grouped.setdefault(key, []).append(review)
    with CASE_REVIEWS_JSONL.open("w", encoding="utf-8") as handle:
        for (action, symbol), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
            handle.write(
                json.dumps(
                    {
                        "training_action": action,
                        "eodhd_symbol": symbol,
                        "row_count": len(items),
                        "review_status_counts": dict(Counter(item.get("review_status") for item in items)),
                        "reason_counts": dict(Counter(item.get("reason") for item in items)),
                        "sample_idea_ids": [item.get("idea_id") for item in items[:10]],
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def write_validation(rows: list[dict[str, str]], review_by_key: dict[tuple[str, str, str], dict[str, Any]]) -> None:
    extra = [
        "remaining_manual_review_status",
        "remaining_manual_training_action",
        "remaining_manual_reason",
        "remaining_manual_confidence",
    ]
    fieldnames = list(rows[0].keys()) + [name for name in extra if name not in rows[0]]
    with VALIDATION_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            review = review_by_key.get(row_key(row))
            if review:
                output.update(
                    {
                        "remaining_manual_review_status": review.get("review_status"),
                        "remaining_manual_training_action": review.get("training_action"),
                        "remaining_manual_reason": review.get("reason"),
                        "remaining_manual_confidence": scalar(review.get("confidence")),
                    }
                )
            writer.writerow(output)


def write_training_ready(
    existing_training: list[dict[str, str]],
    pass_reviews: list[dict[str, Any]],
    prior_reviews: dict[str, dict[str, Any]],
) -> int:
    fieldnames = list(existing_training[0].keys())
    now = datetime.now(UTC).isoformat()
    rows_to_add: list[dict[str, str]] = []
    for review in pass_reviews:
        row = {field: "" for field in fieldnames}
        prior = prior_reviews.get(review_key(review))
        returns = passed_horizon_returns(prior)
        row.update(
            {
                "idea_id": scalar(review.get("idea_id")),
                "raw_symbol": scalar(review.get("raw_symbol")),
                "eodhd_symbol": scalar(review.get("eodhd_symbol")),
                "publication_date": scalar(review.get("publication_date")),
                "include_in_training": "true",
                "math_validation_status": "manually_verified",
                "review_stage": "training_ready_manual_remaining",
                "training_readiness": "training_ready",
                "review_status": "pass",
                "reviewed_at": now,
                "agent_b_yahoo_symbol": scalar(review.get("raw_symbol")),
                "agent_b_yahoo_rows": scalar((prior or {}).get("agent_b_yahoo_rows")),
                "agent_c_status": "pass",
                "agent_c_reason": scalar(review.get("reason")),
                "agent_c_outcome_type": "manual_review_verified",
                "source_count": str(len(review.get("sources") or [])),
                "fundamentals_name": scalar(review.get("company_name")),
                "fundamentals_type": scalar(review.get("fundamentals_type")),
                "original_warning_modes": scalar(review.get("warning_modes")),
                "original_failure_modes": scalar(review.get("failure_modes")),
            }
        )
        for horizon, value in returns.items():
            field = f"validated_perf_{horizon}"
            if field in row:
                row[field] = scalar(value)
        if not returns and review.get("horizon"):
            field = f"validated_perf_{review.get('horizon')}"
            if field in row:
                row[field] = scalar(review.get("return_multiplier"))
        rows_to_add.append(row)

    with TRAINING_READY_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_training)
        writer.writerows(rows_to_add)
    return len(rows_to_add)


def main() -> int:
    rows = load_csv(VALIDATION_CSV)
    existing_training = load_csv(TRAINING_READY_IN)
    training_keys = {row_key(row) for row in existing_training}
    prior_reviews = load_prior_reviews(REVIEWS_JSONL)
    remaining = [row for row in rows if final_status(row, training_keys) == "manual_review_remaining"]
    reviews = [review_row(row, prior_reviews.get(review_key(row))) for row in remaining]
    review_by_row_key = {
        (str(review.get("idea_id") or ""), str(review.get("eodhd_symbol") or ""), str(review.get("publication_date") or "")): review
        for review in reviews
    }
    pass_reviews = [review for review in reviews if review.get("review_status") == "pass"]
    added = write_training_ready(existing_training, pass_reviews, prior_reviews)
    write_row_reviews(reviews)
    write_case_reviews(reviews)
    write_validation(rows, review_by_row_key)

    counts = Counter(review.get("review_status") for review in reviews)
    actions = Counter(review.get("training_action") for review in reviews)
    reasons = Counter(review.get("reason") for review in reviews)
    final_training_count = len(existing_training) + added
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_rows": len(rows),
        "manual_review_remaining_input_rows": len(remaining),
        "review_status_counts": dict(counts),
        "training_action_counts": dict(actions),
        "top_reasons": dict(reasons.most_common(25)),
        "existing_training_ready_rows": len(existing_training),
        "new_training_ready_rows": added,
        "combined_training_ready_rows": final_training_count,
        "still_held_manual_rows": actions.get("hold", 0),
        "newly_excluded_rows": actions.get("exclude", 0),
        "outputs": {
            "row_reviews_csv": str(ROW_REVIEWS_CSV.resolve()),
            "case_reviews_jsonl": str(CASE_REVIEWS_JSONL.resolve()),
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
