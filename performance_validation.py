from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


PROJECT_REF = os.environ.get("SUPABASE_PROJECT_REF", "aqfpldvpcoyipkyihuea")
SUPABASE_URL = os.environ.get("SUPABASE_URL", f"https://{PROJECT_REF}.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

FETCH_PAGE = 1000
MAX_BATCH_SIZE = 50
STALE_CLAIM_SECONDS = 2 * 60 * 60


class ValidationStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    IN_PROGRESS = "in_progress"
    VERIFIED_EXACT = "verified_exact"
    VERIFIED_WITH_CORPORATE_ACTION = "verified_with_corporate_action"
    VERIFIED_SUCCESSOR_SECURITY = "verified_successor_security"
    VERIFIED_DELISTED_OTC = "verified_delisted_otc"
    VERIFIED_DELISTED_ZERO_OR_LIQUIDATION = "verified_delisted_zero_or_liquidation"
    IDENTITY_CONFLICT = "identity_conflict"
    TICKER_REUSE_CONFLICT = "ticker_reuse_conflict"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    BAD_YAHOO_ADJUSTMENT = "bad_yahoo_adjustment"
    EXCLUDE_FROM_TRAINING = "exclude_from_training"
    NEEDS_MANUAL_REVIEW = "needs_manual_review"
    PROVIDER_ERROR = "provider_error"


class IdentityStatus(StrEnum):
    SAME_SECURITY = "same_security"
    TICKER_CHANGED = "ticker_changed"
    ACQUIRED_CASH = "acquired_cash"
    ACQUIRED_STOCK = "acquired_stock"
    ACQUIRED_MIXED = "acquired_mixed"
    DELISTED_OTC = "delisted_otc"
    DELISTED_BANKRUPT = "delisted_bankrupt"
    LIQUIDATED = "liquidated"
    TICKER_REUSE_SUSPECTED = "ticker_reuse_suspected"
    UNSUPPORTED_INSTRUMENT = "unsupported_instrument"
    UNKNOWN = "unknown"


class CorporateActionStatus(StrEnum):
    NONE_DETECTED = "none_detected"
    ADJUSTED_BY_PROVIDER = "adjusted_by_provider"
    MANUALLY_MODELED = "manually_modeled"
    PARTIALLY_MODELED = "partially_modeled"
    MISSING_MATERIAL_ACTION = "missing_material_action"
    CONFLICTING_ACTION_DATA = "conflicting_action_data"
    UNKNOWN = "unknown"


class LabelQuality(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNUSABLE = "unusable"


TRAINING_STATUSES = {
    ValidationStatus.VERIFIED_EXACT,
    ValidationStatus.VERIFIED_WITH_CORPORATE_ACTION,
    ValidationStatus.VERIFIED_SUCCESSOR_SECURITY,
    ValidationStatus.VERIFIED_DELISTED_OTC,
    ValidationStatus.VERIFIED_DELISTED_ZERO_OR_LIQUIDATION,
}

TRAINING_CORPORATE_ACTION_STATUSES = {
    CorporateActionStatus.NONE_DETECTED,
    CorporateActionStatus.ADJUSTED_BY_PROVIDER,
    CorporateActionStatus.MANUALLY_MODELED,
}

MATERIAL_ACTION_TYPES = {
    "split",
    "reverse_split",
    "special_dividend",
    "spinoff",
    "ticker_change",
    "merger_cash",
    "merger_stock",
    "merger_mixed",
    "delisting",
    "bankruptcy",
    "liquidation",
    "otc_transfer",
}


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    reasons: list[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_key() -> str:
    if not SUPABASE_KEY:
        raise SystemExit("Set SUPABASE_SERVICE_ROLE_KEY before running validation scripts.")
    return SUPABASE_KEY


def http_json(method: str, path: str, query: dict[str, str] | None = None, body: Any = None, prefer: str | None = None) -> Any:
    key = require_key()
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    if query:
        url += "?" + urllib.parse.urlencode(query, safe="(),.*")
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, method=method, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = resp.read().decode("utf-8")
            return json.loads(payload) if payload else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed HTTP {exc.code}: {detail[:1000]}") from exc


def is_training_status(status: str) -> bool:
    try:
        return ValidationStatus(status) in TRAINING_STATUSES
    except ValueError:
        return False


def evaluate_training_gate(row: dict[str, Any]) -> GateResult:
    reasons: list[str] = []
    try:
        validation_status = ValidationStatus(row.get("validation_status", ""))
    except ValueError:
        validation_status = ValidationStatus.UNREVIEWED
        reasons.append("invalid_validation_status")
    try:
        corporate_action_status = CorporateActionStatus(row.get("corporate_action_status", ""))
    except ValueError:
        corporate_action_status = CorporateActionStatus.UNKNOWN
        reasons.append("invalid_corporate_action_status")
    try:
        label_quality = LabelQuality(row.get("label_quality", ""))
    except ValueError:
        label_quality = LabelQuality.UNUSABLE
        reasons.append("invalid_label_quality")

    if validation_status not in TRAINING_STATUSES:
        reasons.append("validation_status_not_training_eligible")
    if corporate_action_status not in TRAINING_CORPORATE_ACTION_STATUSES:
        reasons.append("corporate_action_status_not_training_eligible")
    if label_quality not in {LabelQuality.HIGH, LabelQuality.MEDIUM}:
        reasons.append("label_quality_too_low")
    if row.get("agent_b_result", {}).get("reviewer_status") != "pass":
        reasons.append("agent_b_not_pass")
    if float(row.get("identity_confidence") or 0) < 0.85:
        reasons.append("identity_confidence_too_low")
    if float(row.get("return_confidence") or 0) < 0.75:
        reasons.append("return_confidence_too_low")
    if has_unresolved_material_action(row):
        reasons.append("unresolved_material_corporate_action")
    if has_weak_sources(row):
        reasons.append("insufficient_sources")
    return GateResult(not reasons, reasons)


def has_unresolved_material_action(row: dict[str, Any]) -> bool:
    action_status = row.get("corporate_action_status")
    if action_status in {
        CorporateActionStatus.MISSING_MATERIAL_ACTION,
        CorporateActionStatus.CONFLICTING_ACTION_DATA,
        CorporateActionStatus.PARTIALLY_MODELED,
        CorporateActionStatus.UNKNOWN,
    }:
        timeline = row.get("corporate_action_timeline") or []
        return any((event or {}).get("type") in MATERIAL_ACTION_TYPES for event in timeline)
    return False


def has_weak_sources(row: dict[str, Any]) -> bool:
    sources = row.get("sources") or []
    if len(sources) < 2:
        return True
    source_types = {source.get("source_type") for source in sources if isinstance(source, dict)}
    primary_types = {"sec", "exchange", "company_ir", "identifier_registry", "data_vendor"}
    return not bool(source_types & primary_types)


def classify_provider_adjustment(row: dict[str, Any]) -> dict[str, Any]:
    split_events = row.get("split_events") or []
    dividend_events = row.get("dividend_events") or []
    has_splits = bool(split_events)
    has_dividends = bool(dividend_events)
    suspicious_split = False
    timeline = []
    for event in split_events:
        numerator = clean_float(event.get("numerator"))
        denominator = clean_float(event.get("denominator"))
        ratio = numerator / denominator if numerator and denominator else None
        event_type = "split"
        if ratio and ratio < 1:
            event_type = "reverse_split"
        if ratio and (ratio > 1.5 or ratio < 0.67):
            suspicious_split = True
        timeline.append(
            {
                "date": event.get("date"),
                "type": event_type,
                "description": f"Provider split event ratio={event.get('splitRatio') or ratio}",
                "materiality": "high" if suspicious_split else "medium",
                "source_ids": ["provider:yahoo"],
            }
        )
    for event in dividend_events:
        timeline.append(
            {
                "date": event.get("date"),
                "type": "dividend",
                "description": f"Provider dividend amount={event.get('amount')}",
                "materiality": "low",
                "source_ids": ["provider:yahoo"],
            }
        )
    return {
        "split_adjusted": has_splits,
        "dividend_adjusted": has_dividends,
        "corporate_action_timeline": timeline,
        "corporate_action_status": CorporateActionStatus.ADJUSTED_BY_PROVIDER
        if (has_splits or has_dividends)
        else CorporateActionStatus.NONE_DETECTED,
        "failure_modes": ["provider_split_requires_identity_review"] if suspicious_split else [],
    }


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_seed_row(raw: dict[str, Any]) -> dict[str, Any]:
    action_info = classify_provider_adjustment(raw)
    return {
        "idea_id": raw["idea_id"],
        "raw_symbol": raw.get("raw_symbol"),
        "yahoo_symbol": raw.get("yahoo_symbol"),
        "company_name": raw.get("company_name"),
        "publication_date": raw.get("publication_date"),
        "position_type": raw.get("position_type") or "unknown",
        "validation_status": ValidationStatus.UNREVIEWED,
        "identity_status": IdentityStatus.UNKNOWN,
        "corporate_action_status": action_info["corporate_action_status"],
        "label_quality": LabelQuality.UNUSABLE,
        "include_in_training": False,
        "split_adjusted": action_info["split_adjusted"],
        "dividend_adjusted": action_info["dividend_adjusted"],
        "spin_off_adjusted": False,
        "merger_adjusted": False,
        "corporate_action_timeline": action_info["corporate_action_timeline"],
        "sources": [
            {
                "source_id": "provider:yahoo",
                "url": "https://query1.finance.yahoo.com/v8/finance/chart",
                "publisher": "Yahoo Finance",
                "source_type": "data_vendor",
                "accessed_date": utc_now()[:10],
                "supports": "price",
                "quote_or_fact": "Raw provider adjusted close, split, and dividend events used as evidence only.",
            }
        ],
        "failure_modes": action_info["failure_modes"],
        "updated_at": utc_now(),
    }


def fetch_queue_rows(limit: int) -> list[dict[str, Any]]:
    rows = http_json(
        "GET",
        "performance_validation_queue_v1",
        {
            "select": "*",
            "order": "risk_priority.asc,publication_date.asc,idea_id.asc",
            "limit": str(min(limit, MAX_BATCH_SIZE)),
        },
    )
    return rows or []


def seed_from_queue(limit: int, dry_run: bool) -> int:
    rows = fetch_queue_rows(limit)
    payload = [build_seed_row(row) for row in rows]
    if dry_run:
        print(json.dumps(payload[:5], indent=2))
        print(f"Prepared {len(payload)} validation seed rows.")
        return len(payload)
    if payload:
        http_json(
            "POST",
            "performance_validation",
            body=payload,
            prefer="resolution=merge-duplicates,return=minimal",
        )
    return len(payload)


def claim_rows(agent_id: str, limit: int) -> list[dict[str, Any]]:
    rows = fetch_queue_rows(limit)
    claimed: list[dict[str, Any]] = []
    now = utc_now()
    for row in rows:
        idea_id = row["idea_id"]
        update = {
            "validation_status": ValidationStatus.IN_PROGRESS,
            "claimed_by": agent_id,
            "claimed_at": now,
            "updated_at": now,
        }
        result = http_json(
            "PATCH",
            "performance_validation",
            {"idea_id": f"eq.{idea_id}", "validation_status": "in.(unreviewed,provider_error,needs_manual_review,insufficient_evidence)"},
            update,
            prefer="return=representation",
        )
        if result:
            claimed.append(result[0])
    return claimed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="store_true", help="Create unreviewed validation rows from the queue.")
    parser.add_argument("--claim", action="store_true", help="Claim queue rows for an external agent run.")
    parser.add_argument("--agent-id", default=f"codex-{int(time.time())}")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.seed:
        count = seed_from_queue(args.limit, args.dry_run)
        print(f"Seeded {count} rows." if not args.dry_run else f"Dry-run seed prepared {count} rows.")
        return 0
    if args.claim:
        rows = claim_rows(args.agent_id, args.limit)
        print(json.dumps(rows, indent=2))
        return 0
    parser.error("Choose --seed or --claim.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
