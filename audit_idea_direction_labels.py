from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from build_investment_finetune_dataset import SQL_DUMP, parse_copy_sections


ROOT = Path(__file__).resolve().parent
CANONICAL = ROOT / "data" / "processed" / "investment_canonical.csv"
CONFIRMATIONS_JSON = ROOT / "idea_direction_confirmations.json"
QUALITY_FLAGS_JSON = ROOT / "idea_quality_flags.json"
OUT_DIR = ROOT / "reports" / "direction_label_audit"
AUDIT_CSV = OUT_DIR / "direction_label_audit.csv"
SUMMARY_JSON = OUT_DIR / "direction_label_audit_summary.json"
SUGGESTED_OVERRIDES_JSON = OUT_DIR / "suggested_direction_overrides_high_confidence.json"
REVIEW_QUEUE_CSV = OUT_DIR / "direction_review_queue.csv"

INTRO_CHARS = 5_000
EXPLICIT_CHARS = 2_500
CONTEXT_CHARS = 120


EXPLICIT_SHORT_PATTERNS: list[tuple[str, int, str]] = [
    (r"(?:^|\s)i\S?m\s+short\s+[A-Za-z0-9&.,'() /-]{1,80}", 10, "explicit short position"),
    (r"\bi['’]m\s+short\s+(?:shares?\s+(?:of|in)\s+|stock\s+(?:of|in)\s+|the\s+stock\s+(?:of|in)\s+)?[A-Za-z0-9&.,'() /-]{1,80}", 10, "explicit short position"),
    (r"\b(?:i|we)\s+(?:are|am|remain)\s+short\s+(?:shares?\s+(?:of|in)\s+|stock\s+(?:of|in)\s+|the\s+stock\s+(?:of|in)\s+)?[A-Za-z0-9&.,'() /-]{1,80}", 10, "explicit short position"),
    (r"\b(?:i|we)\s+(?:am|are)\s+recommending\s+(?:a\s+)?short\s+position\s+(?:in|on)\s+[A-Za-z0-9&.,'() /-]{1,90}", 10, "explicit short position recommendation"),
    (r"\b(?:i|we)\s+(?:recommend|suggest)\s+(?:a\s+)?short\s+position\s+(?:in|on)\s+[A-Za-z0-9&.,'() /-]{1,90}", 10, "explicit short position recommendation"),
    (r"\b(?:i|we)\s+advocate\s+(?:a\s+)?short\s+position\s+(?:in|on)\s+[A-Za-z0-9&.,'() /-]{1,90}", 10, "explicit short position recommendation"),
    (r"\brecommend\s+initiating\s+(?:a\s+)?short\s+position\s+(?:in|on)\s+[A-Za-z0-9&.,'() /-]{1,90}", 10, "explicit short position recommendation"),
    (r"\b(?:i|we)\s+recommend\s+(?:shareholders|investors)\s+short\s+shares?\s+(?:of|in)?\s*[A-Za-z0-9&.,'() /-]{1,90}", 10, "explicit short recommendation"),
    (r"\b(?:i|we)\s+(?:recommend|suggest)\s+(?:shorting|a\s+short)\b", 10, "explicit short recommendation"),
    (r"\b(?:i|we)\s+(?:recommend|suggest)\s+selling\s+(?!(?:puts?|calls?|options?|warrants?)\b)(?:the\s+)?(?:stock|shares?|common|equity)\b", 10, "explicit short recommendation"),
    (r"\b(?:i|we)\s+(?:am|are)\s+pitching\s+[A-Za-z0-9&.,'() /-]{1,80}\s+as\s+(?:a\s+)?short\b", 10, "pitching security as short"),
    (r"\b(?:recommendation|thesis|summary\s+thesis)\s*[-:–—]?\s*short\b", 10, "short recommendation heading"),
    (r"^\s*short(?:\s*[-:–—]\s*|\s+)(?!term\b|duration\b|report\b|case\b|interest\b|sighted\b|background\b|description\b|and\s+sweet\b)[A-Z][A-Za-z0-9&.,'() /-]{1,90}", 10, "short heading"),
    (r"\b[A-Z][A-Za-z0-9&.,'() /-]{1,80}\s+(?:common\s+stock\s+)?is\s+a\s+short(?:\s+because|[.;,]|$)", 10, "security is a short"),
    (r"\b[A-Z][A-Za-z0-9&.,'() /-]{1,80}\s+represents\s+(?:a\s+)?(?:compelling|attractive|interesting)?\s*short\s+(?:opportunity|investment|idea|candidate)\b", 10, "security represents short opportunity"),
]

EXPLICIT_LONG_PATTERNS: list[tuple[str, int, str]] = [
    (r"\b(?:i|we)\s+(?:are|am|remain)\s+long\b", 10, "explicit long position"),
    (r"\b(?:i|we)\s+(?:am|are)\s+recommending\s+(?:a\s+)?long\s+position\s+(?:in|on)\s+[A-Za-z0-9&.,'() /-]{1,90}", 10, "explicit long position recommendation"),
    (r"\b(?:i|we)\s+(?:recommend|suggest)\s+(?:a\s+)?long\s+position\s+(?:in|on)\s+[A-Za-z0-9&.,'() /-]{1,90}", 10, "explicit long position recommendation"),
    (r"\b(?:i|we)\s+advocate\s+(?:a\s+)?long\s+position\s+(?:in|on)\s+[A-Za-z0-9&.,'() /-]{1,90}", 10, "explicit long position recommendation"),
    (r"\brecommend\s+initiating\s+(?:a\s+)?long\s+position\s+(?:in|on)\s+[A-Za-z0-9&.,'() /-]{1,90}", 10, "explicit long position recommendation"),
    (r"\b(?:i|we)\s+(?:recommend|suggest)\s+(?:(?:buying|purchasing|owning)\s+(?:the\s+)?(?:stock|shares?|common|equity|security)\b|going\s+long\b|a\s+long\b)", 10, "explicit long recommendation"),
    (r"\b(?:i|we)\s+recommend\s+(?:the\s+)?(?:purchase|ownership)\b", 10, "explicit purchase recommendation"),
    (r"\b(?:i|we)\s+(?:am|are)\s+pitching\s+[A-Za-z0-9&.,'() /-]{1,80}\s+as\s+(?:a\s+)?long\b", 10, "pitching security as long"),
    (r"\b(?:recommendation|thesis|summary\s+thesis)\s*[-:–—]?\s*(?:long|buy)\b", 10, "long recommendation heading"),
    (r"\brecommend\s+long\s+[A-Za-z0-9&.,'() /-]{1,90}", 10, "recommend long security"),
    (r"^\s*long(?:\s*[-:–—]\s*|\s+)(?!term\b|runway\b|haul\b|duration\b|short\b)[A-Z][A-Za-z0-9&.,'() /-]{1,90}", 10, "long heading"),
    (r"^\s*buy(?:\s*[-:–—]\s*|\s+)[A-Z][A-Za-z0-9&.,'() /-]{1,90}", 10, "buy heading"),
    (r"\b[A-Z][A-Za-z0-9&.,'() /-]{1,80}\s+is\s+a\s+long(?:\s+because|[.;,]|$)", 10, "security is a long"),
    (r"\b[A-Z][A-Za-z0-9&.,'() /-]{1,80}\s+represents\s+(?:a\s+)?(?:compelling|attractive|interesting)?\s*long\s+(?:opportunity|investment|idea|candidate)\b", 10, "security represents long opportunity"),
    (r"\b[A-Z][A-Za-z0-9&.,'() /-]{1,80}\s+is\s+(?:an?\s+)?(?:attractive|compelling|interesting)\s+long\s+(?:investment|opportunity|idea)\b", 10, "security is attractive long investment"),
    (r"\b(?:i|we)\s+own\s+(?:the\s+)?(?:stock|shares|security)\b", 8, "owns stock/shares"),
]

SHORT_PATTERNS: list[tuple[str, int, str]] = [
    (r"\b(?:i|we)\s+(?:are|am|remain)\s+short\b", 9, "explicit short position"),
    (r"\b(?:i|we)\s+(?:recommend|suggest)\s+(?:shorting|a\s+short)\b", 9, "explicit short recommendation"),
    (r"^\s*short\s+[A-Za-z0-9&.,'() -]{1,90}$", 8, "standalone short heading"),
    (r"\bshort\s+(?:idea|case|thesis|recommendation|candidate)\b", 7, "short thesis phrase"),
    (r"\b(?:sell|avoid)\s+(?:recommendation|rating|the\s+stock|shares)\b", 5, "sell/avoid recommendation"),
    (r"\b(?:puts?|put\s+options?)\b.{0,80}\b(?:downside|short|overvalued)\b", 4, "put/downside language"),
    (r"\b(?:overvalued|over-priced|overpriced|bubble|fraud|promotional|bankrupt|worthless)\b", 2, "bearish language"),
    (r"\b(?:downside|price\s+target\s+of\s+\$?0|zero)\b", 2, "downside language"),
]

LONG_PATTERNS: list[tuple[str, int, str]] = [
    (r"\b(?:i|we)\s+(?:are|am|remain)\s+long\b", 9, "explicit long position"),
    (r"\b(?:i|we)\s+(?:recommend|suggest)\s+(?:buying|owning|going\s+long)\b", 9, "explicit long recommendation"),
    (r"\b(?:i|we)\s+(?:recommend|like)\s+[A-Z][A-Za-z0-9&.,'() -]{1,90}\b", 6, "recommend company/security"),
    (r"\b(?:i|we)\s+own\s+(?:the\s+)?(?:stock|shares|security)\b", 7, "owns stock/shares"),
    (r"\b(?:i|we)\s+(?:believe|think)\s+(?:the\s+)?(?:stock|shares|company)\s+(?:is|are)\s+(?:cheap|undervalued|mispriced|attractive)\b", 6, "positive valuation language"),
    (r"\b(?:undervalued|mispriced|cheap|attractive|compelling|upside|margin\s+of\s+safety)\b", 2, "bullish language"),
    (r"\b(?:target|worth|fair\s+value)\b.{0,80}\b(?:upside|higher|above|return)\b", 3, "upside target language"),
    (r"\b(?:buyback|compound|fcf\s+yield|free\s+cash\s+flow|moat|quality\s+business)\b", 1, "positive business language"),
]

NOISE_PHRASES = [
    "short interest",
    "short-term",
    "short term",
    "shortage",
    "shortly",
    "short run",
    "long-term",
    "long term",
    "longer term",
    "long runway",
]

COMPANY_STOPWORDS = {
    "inc",
    "inc.",
    "corp",
    "corp.",
    "corporation",
    "co",
    "company",
    "common",
    "stock",
    "class",
    "ltd",
    "limited",
    "plc",
    "adr",
    "holdings",
    "group",
    "sa",
    "nv",
    "lp",
    "llc",
    "airline",
    "airlines",
    "airways",
}


def clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def load_confirmations() -> dict[str, dict[str, Any]]:
    if not CONFIRMATIONS_JSON.exists():
        return {}
    return json.loads(CONFIRMATIONS_JSON.read_text(encoding="utf-8"))


def load_quality_flags() -> dict[str, dict[str, Any]]:
    if not QUALITY_FLAGS_JSON.exists():
        return {}
    return json.loads(QUALITY_FLAGS_JSON.read_text(encoding="utf-8"))


def confirmed_direction(confirmation: dict[str, Any]) -> bool | None:
    """Return the manually confirmed short flag across old and new schemas."""
    value = confirmation.get("is_short")
    if isinstance(value, bool):
        return value
    value = confirmation.get("confirmed_is_short")
    if isinstance(value, bool):
        return value
    return None


def excerpt(text: str, start: int, end: int) -> str:
    left = max(0, start - CONTEXT_CHARS)
    right = min(len(text), end + CONTEXT_CHARS)
    return clean_space(text[left:right])


def is_direction_false_positive(label: str, match_text: str, snippet: str) -> bool:
    text = f"{match_text} {snippet}".lower()
    match_lower = match_text.lower().strip()
    if re.search(r"^(?:short|long)[-\s]*term\b", match_lower):
        return True
    if label in {"short recommendation heading", "short heading"} and re.search(r"\bshort\s*[-\s]*term\b", text):
        return True
    if label in {"long recommendation heading", "long heading"} and re.search(r"\blong\s*[-\s]*term\b", text):
        return True
    if re.search(r"\bshort[-\s]*sellers?\b|\bshort\s+interest\b|\bshort\s+intro\b", text):
        return True
    if re.search(r"\blong\s+(?:ago|story|years?)\b|\bbuy\s+side\b", text):
        return True
    if re.search(r"\b(?:buying|purchasing|owning)\s+(?:puts?|calls?|options?|warrants?)\b", text):
        return True
    if re.search(r"\bselling\s+(?:puts?|calls?|options?|warrants?)\b", text):
        return True
    if label in {"short heading", "long heading", "buy heading"}:
        if re.search(r"\bshort\s+the\b|\bshort\s+intro\b|\bshort[-\s]*sellers?\b", text):
            return True
    return False


def count_patterns(text: str, patterns: list[tuple[str, int, str]]) -> tuple[int, list[dict[str, Any]]]:
    score = 0
    evidence: list[dict[str, Any]] = []
    lines = text.splitlines()
    line_text = "\n".join(line[:140] for line in lines[:20])
    for pattern, weight, label in patterns:
        flags = re.IGNORECASE | re.MULTILINE
        search_text = line_text if pattern.startswith("^") else text
        for match in re.finditer(pattern, search_text, flags):
            snippet = excerpt(search_text, match.start(), match.end())
            lowered = snippet.lower()
            if any(noise in lowered for noise in NOISE_PHRASES) and weight < 7:
                continue
            if is_direction_false_positive(label, match.group(0), snippet):
                continue
            score += weight
            evidence.append(
                {
                    "label": label,
                    "weight": weight,
                    "match": match.group(0),
                    "excerpt": snippet,
                }
            )
            if len(evidence) >= 8:
                return score, evidence
    return score, evidence


def security_tokens(row: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in ["raw_symbol", "eodhd_symbol"]:
        value = clean_space(row.get(key, ""))
        symbol = value.split(".")[0].upper()
        if len(symbol) >= 2:
            tokens.add(symbol)

    company = clean_space(row.get("company_name", ""))
    for word in re.findall(r"[A-Za-z][A-Za-z&'-]{2,}", company):
        cleaned = word.lower().strip(".")
        if cleaned not in COMPANY_STOPWORDS and len(cleaned) >= 4:
            tokens.add(cleaned.upper())
    return tokens


def evidence_mentions_security(row: dict[str, Any], item: dict[str, Any], *, strict_match: bool) -> bool:
    tokens = security_tokens(row)
    if not tokens:
        return False
    haystack = item.get("match", "") if strict_match else f"{item.get('match', '')} {item.get('excerpt', '')}"
    upper = haystack.upper()
    return any(re.search(rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])", upper) for token in tokens)


def auto_fix_support(row: dict[str, Any], explicit: dict[str, Any]) -> tuple[bool, str]:
    direction = explicit["explicit_direction"]
    evidence = (
        explicit["explicit_short_evidence"]
        if direction == "short"
        else explicit["explicit_long_evidence"]
    )
    if not evidence:
        return False, "no explicit evidence"

    for item in evidence:
        label = item.get("label", "")
        snippet = f"{item.get('match', '')} {item.get('excerpt', '')}".lower()
        if any(
            phrase in snippet
            for phrase in [
                "short the put",
                "short puts",
                "long put",
                "buy side",
                "long history",
                "short write up",
                "short background",
                "short and sweet",
                "long credit",
                "long short hedge fund",
            ]
        ):
            continue

        if label in {
            "long heading",
            "buy heading",
            "short heading",
            "security is a long",
            "security is a short",
            "security represents long opportunity",
            "security represents short opportunity",
            "security is attractive long investment",
            "explicit long position recommendation",
            "explicit short position recommendation",
            "pitching security as long",
            "pitching security as short",
        }:
            if evidence_mentions_security(row, item, strict_match=True):
                return True, f"{label} names the security"
            continue

        if evidence_mentions_security(row, item, strict_match=False):
            return True, f"{label} mentions the security"

    return False, "explicit phrase did not name this security"


def strict_direction(row: dict[str, Any], explicit: dict[str, Any]) -> dict[str, Any]:
    """Only accept direction when explicit language names this row's security."""
    if explicit["explicit_direction"] == "conflict":
        return {
            "strict_direction": "conflict",
            "strict_is_short": None,
            "strict_support_reason": "conflicting explicit long and short evidence",
        }
    if explicit["explicit_is_short"] is None:
        return {
            "strict_direction": "none",
            "strict_is_short": None,
            "strict_support_reason": "no explicit direction evidence",
        }

    supported, reason = auto_fix_support(row, explicit)
    if not supported:
        return {
            "strict_direction": "none",
            "strict_is_short": None,
            "strict_support_reason": reason,
        }

    return {
        "strict_direction": explicit["explicit_direction"],
        "strict_is_short": explicit["explicit_is_short"],
        "strict_support_reason": reason,
    }


def explicit_direction(description: str, catalyst: str) -> dict[str, Any]:
    text = "\n\n".join(part for part in [description or "", catalyst or ""] if part)
    intro = text[:EXPLICIT_CHARS]
    short_score, short_evidence = count_patterns(intro, EXPLICIT_SHORT_PATTERNS)
    long_score, long_evidence = count_patterns(intro, EXPLICIT_LONG_PATTERNS)

    if long_score > 0 and short_score == 0:
        inferred = "long"
        is_short = False
        confidence = "explicit"
    elif short_score > 0 and long_score == 0:
        inferred = "short"
        is_short = True
        confidence = "explicit"
    elif long_score > 0 and short_score > 0:
        inferred = "conflict"
        is_short = None
        confidence = "conflict"
    else:
        inferred = "none"
        is_short = None
        confidence = "none"

    return {
        "explicit_direction": inferred,
        "explicit_is_short": is_short,
        "explicit_confidence": confidence,
        "explicit_long_score": long_score,
        "explicit_short_score": short_score,
        "explicit_long_evidence": long_evidence[:5],
        "explicit_short_evidence": short_evidence[:5],
    }


def classify_direction(description: str, catalyst: str) -> dict[str, Any]:
    text = "\n\n".join(part for part in [description or "", catalyst or ""] if part)
    intro = text[:INTRO_CHARS]
    short_score, short_evidence = count_patterns(intro, SHORT_PATTERNS)
    long_score, long_evidence = count_patterns(intro, LONG_PATTERNS)

    # Disclaimer text often says "I hold a material investment" for both long and
    # short ideas. It is useful context, but not enough to override explicit text.
    all_text = text.lower()
    if "i and/or others i advise hold a material investment" in all_text:
        long_score += 1
        long_evidence.append(
            {
                "label": "disclaimer material investment",
                "weight": 1,
                "match": "hold a material investment",
                "excerpt": "I and/or others I advise hold a material investment in the issuer's securities.",
            }
        )

    margin = abs(long_score - short_score)
    if long_score == 0 and short_score == 0:
        inferred = "unknown"
        confidence = "none"
    elif margin >= 7 and max(long_score, short_score) >= 8:
        inferred = "long" if long_score > short_score else "short"
        confidence = "high"
    elif margin >= 4 and max(long_score, short_score) >= 5:
        inferred = "long" if long_score > short_score else "short"
        confidence = "medium"
    else:
        inferred = "long" if long_score > short_score else "short" if short_score > long_score else "ambiguous"
        confidence = "low"

    return {
        "inferred_direction": inferred,
        "confidence": confidence,
        "long_score": long_score,
        "short_score": short_score,
        "score_margin": margin,
        "long_evidence": long_evidence[:5],
        "short_evidence": short_evidence[:5],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, _, ideas = parse_copy_sections(SQL_DUMP)
    canonical = pd.read_csv(CANONICAL, dtype=str, keep_default_na=False)
    confirmations = load_confirmations()
    quality_flags = load_quality_flags()

    rows = []
    high_conf_overrides: dict[str, dict[str, Any]] = {}
    for row in canonical.to_dict(orient="records"):
        idea_id = row["idea_id"]
        source_meta = ideas.get(idea_id)
        source_is_short = bool(source_meta.is_short) if source_meta else None
        current_is_short = str(row.get("is_short")).lower() == "true"
        confirmation = confirmations.get(idea_id, {})
        quality_flag = quality_flags.get(idea_id, {})
        confirmed_is_short = confirmed_direction(confirmation)
        confirmed_current_label = (
            isinstance(confirmed_is_short, bool) and confirmed_is_short == current_is_short
        )
        classified = classify_direction(row.get("description", ""), row.get("catalyst", ""))
        explicit = explicit_direction(row.get("description", ""), row.get("catalyst", ""))
        strict = strict_direction(row, explicit)
        inferred = classified["inferred_direction"]
        inferred_is_short = True if inferred == "short" else False if inferred == "long" else None
        explicit_is_short = explicit["explicit_is_short"]
        strict_is_short = strict["strict_is_short"]
        mismatch_current = inferred_is_short is not None and inferred_is_short != current_is_short
        mismatch_source = inferred_is_short is not None and source_is_short is not None and inferred_is_short != source_is_short
        explicit_mismatch_current = explicit_is_short is not None and explicit_is_short != current_is_short
        explicit_mismatch_source = explicit_is_short is not None and source_is_short is not None and explicit_is_short != source_is_short
        strict_mismatch_current = strict_is_short is not None and strict_is_short != current_is_short
        strict_mismatch_source = strict_is_short is not None and source_is_short is not None and strict_is_short != source_is_short
        supported, support_reason = auto_fix_support(row, explicit) if explicit_mismatch_current else (False, "")
        auto_fix_eligible = bool(
            explicit_mismatch_current
            and explicit["explicit_confidence"] == "explicit"
            and supported
            and not confirmed_current_label
            and not quality_flag
        )
        audit_row = {
            "idea_id": idea_id,
            "raw_symbol": row.get("raw_symbol"),
            "eodhd_symbol": row.get("eodhd_symbol"),
            "company_name": row.get("company_name"),
            "publication_date": row.get("publication_date"),
            "author_user_id": row.get("author_user_id"),
            "link": row.get("link"),
            "source_sql_is_short": source_is_short,
            "current_is_short": current_is_short,
            "confirmed_current_label": confirmed_current_label,
            "confirmation_source": confirmation.get("source", ""),
            "confirmation_reason": confirmation.get("reason", ""),
            "quality_flag_issue": quality_flag.get("issue", ""),
            "quality_flag_reason": quality_flag.get("reason", ""),
            "inferred_direction": inferred,
            "inferred_is_short": inferred_is_short,
            "confidence": classified["confidence"],
            "long_score": classified["long_score"],
            "short_score": classified["short_score"],
            "score_margin": classified["score_margin"],
            "mismatch_current": mismatch_current,
            "mismatch_source": mismatch_source,
            "explicit_direction": explicit["explicit_direction"],
            "explicit_is_short": explicit_is_short,
            "explicit_confidence": explicit["explicit_confidence"],
            "explicit_long_score": explicit["explicit_long_score"],
            "explicit_short_score": explicit["explicit_short_score"],
            "explicit_mismatch_current": explicit_mismatch_current,
            "explicit_mismatch_source": explicit_mismatch_source,
            "strict_direction": strict["strict_direction"],
            "strict_is_short": strict_is_short,
            "strict_mismatch_current": strict_mismatch_current,
            "strict_mismatch_source": strict_mismatch_source,
            "strict_support_reason": strict["strict_support_reason"],
            "auto_fix_eligible": auto_fix_eligible,
            "auto_fix_support_reason": support_reason,
            "long_evidence": json.dumps(classified["long_evidence"], ensure_ascii=False),
            "short_evidence": json.dumps(classified["short_evidence"], ensure_ascii=False),
            "explicit_long_evidence": json.dumps(explicit["explicit_long_evidence"], ensure_ascii=False),
            "explicit_short_evidence": json.dumps(explicit["explicit_short_evidence"], ensure_ascii=False),
        }
        rows.append(audit_row)

        if auto_fix_eligible:
            explicit_direction_label = explicit["explicit_direction"]
            evidence = (
                explicit["explicit_short_evidence"]
                if explicit_direction_label == "short"
                else explicit["explicit_long_evidence"]
            )
            high_conf_overrides[idea_id] = {
                "is_short": bool(explicit_is_short),
                "reason": (
                    f"Explicit thesis-text direction audit inferred {explicit_direction_label}; "
                    f"current direction flag disagreed. Evidence: "
                    f"{'; '.join(item['excerpt'] for item in evidence[:3])}"
                ),
            }

    frame = pd.DataFrame(rows)
    frame.to_csv(AUDIT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    review = frame[
        (frame["auto_fix_eligible"] != True)
        & (frame["confirmed_current_label"] != True)
        & (frame["quality_flag_issue"].eq(""))
        & ((frame["mismatch_current"] == True)
        | (frame["confidence"].isin(["low", "none"]))
        | (frame["inferred_direction"].isin(["ambiguous", "unknown"]))
        | (frame["explicit_direction"].eq("conflict")))
    ].copy()
    review.to_csv(REVIEW_QUEUE_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    SUGGESTED_OVERRIDES_JSON.write_text(
        json.dumps(high_conf_overrides, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    summary = {
        "total_rows": int(len(frame)),
        "current_short_rows": int(frame["current_is_short"].sum()),
        "source_short_rows": int(frame["source_sql_is_short"].fillna(False).sum()),
        "inferred_direction_counts": frame["inferred_direction"].value_counts(dropna=False).to_dict(),
        "confidence_counts": frame["confidence"].value_counts(dropna=False).to_dict(),
        "mismatch_current_rows": int(frame["mismatch_current"].sum()),
        "confirmed_current_label_rows": int(frame["confirmed_current_label"].sum()),
        "quality_flagged_rows": int((frame["quality_flag_issue"] != "").sum()),
        "unconfirmed_mismatch_current_rows": int(
            (
                (frame["mismatch_current"] == True)
                & (frame["confirmed_current_label"] != True)
                & (frame["quality_flag_issue"] == "")
            ).sum()
        ),
        "mismatch_source_rows": int(frame["mismatch_source"].sum()),
        "explicit_direction_counts": frame["explicit_direction"].value_counts(dropna=False).to_dict(),
        "explicit_mismatch_current_rows": int(frame["explicit_mismatch_current"].sum()),
        "explicit_mismatch_source_rows": int(frame["explicit_mismatch_source"].sum()),
        "strict_direction_counts": frame["strict_direction"].value_counts(dropna=False).to_dict(),
        "strict_mismatch_current_rows": int(frame["strict_mismatch_current"].sum()),
        "strict_mismatch_source_rows": int(frame["strict_mismatch_source"].sum()),
        "high_confidence_auto_override_candidates": len(high_conf_overrides),
        "review_queue_rows": int(len(review)),
        "audit_csv": str(AUDIT_CSV),
        "review_queue_csv": str(REVIEW_QUEUE_CSV),
        "suggested_overrides_json": str(SUGGESTED_OVERRIDES_JSON),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
