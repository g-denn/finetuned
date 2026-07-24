#!/usr/bin/env python3
"""Build fine-tuning datasets from validated investment outcomes and VIC memos.

This joins the final validated performance labels with the original investment
memo/catalyst text from the PostgreSQL dump. The output is designed for two
uses:

1. canonical tabular audit data
2. chat-style JSONL examples for investment-process fine-tuning
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TRAINING_CSV = ROOT / "eodhd_output" / "full_run" / "training_ready_after_sec_yahoo_salvage.csv"
SQL_DUMP = ROOT / "VIC_IDEAS.sql"
OUT_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
DIRECTION_OVERRIDES_PATH = ROOT / "idea_direction_overrides.json"

HORIZONS = ("1y", "3y", "5y", "10y", "20y")
PRIMARY_HORIZON_ORDER = ("3y", "5y", "1y", "10y", "20y")
SFT_DESCRIPTION_CHAR_LIMIT = 12_000
SFT_CATALYST_CHAR_LIMIT = 1_500


@dataclass
class IdeaMeta:
    link: str
    company_id: str
    user_id: str
    publication_date: str
    is_short: bool
    is_contest_winner: bool


def pg_unescape(value: str) -> str | None:
    if value == r"\N":
        return None
    replacements = {
        r"\n": "\n",
        r"\r": "\r",
        r"\t": "\t",
        r"\b": "\b",
        r"\f": "\f",
        r"\\": "\\",
    }
    out: list[str] = []
    index = 0
    while index < len(value):
        if value[index] == "\\" and index + 1 < len(value):
            token = value[index : index + 2]
            if token in replacements:
                out.append(replacements[token])
                index += 2
                continue
        out.append(value[index])
        index += 1
    return "".join(out)


def parse_copy_sections(sql_path: Path) -> tuple[dict[str, str], dict[str, str], dict[str, IdeaMeta]]:
    descriptions: dict[str, str] = {}
    catalysts: dict[str, str] = {}
    ideas: dict[str, IdeaMeta] = {}

    active: str | None = None
    with sql_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "COPY public.descriptions (idea_id, description) FROM stdin;":
                active = "descriptions"
                continue
            if line == "COPY public.catalyst (idea_id, catalysts) FROM stdin;":
                active = "catalysts"
                continue
            if line == "COPY public.ideas (id, link, company_id, user_id, date, is_short, is_contest_winner) FROM stdin;":
                active = "ideas"
                continue
            if active and line == r"\.":
                active = None
                continue
            if not active:
                continue

            parts = line.split("\t")
            if active == "descriptions" and len(parts) >= 2:
                descriptions[parts[0]] = pg_unescape(parts[1]) or ""
            elif active == "catalysts" and len(parts) >= 2:
                catalysts[parts[0]] = pg_unescape(parts[1]) or ""
            elif active == "ideas" and len(parts) >= 7:
                ideas[parts[0]] = IdeaMeta(
                    link=pg_unescape(parts[1]) or "",
                    company_id=pg_unescape(parts[2]) or "",
                    user_id=pg_unescape(parts[3]) or "",
                    publication_date=(pg_unescape(parts[4]) or "")[:10],
                    is_short=parts[5] == "t",
                    is_contest_winner=parts[6] == "t",
                )
    return descriptions, catalysts, ideas


MOJIBAKE_REPLACEMENTS = {
    "Â\xa0": " ",
    "Â ": " ",
    "Â": "",
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "â€¢": "-",
    "ï»¿": "",
}


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    value = text
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        value = value.replace(bad, good)
    value = value.replace("\u00a0", " ")
    value = re.sub(r"(?i)^\s*description\s*", "", value)
    value = re.sub(r"\r\n|\r", "\n", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r"https?://\S+", "[URL]", value)
    return value.strip()


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return parsed


def read_direction_overrides() -> dict[str, dict[str, Any]]:
    if not DIRECTION_OVERRIDES_PATH.exists():
        return {}
    with DIRECTION_OVERRIDES_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def outcome_bucket(multiplier: float | None) -> str:
    if multiplier is None:
        return "missing"
    if multiplier >= 3.0:
        return "excellent"
    if multiplier >= 1.5:
        return "good"
    if multiplier >= 0.8:
        return "neutral"
    if multiplier >= 0.4:
        return "poor"
    return "failed"


def compact_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return (
        text[:head].rstrip()
        + "\n\n[...memo truncated for context budget...]\n\n"
        + text[-tail:].lstrip()
    )


def choose_primary_horizon(row: dict[str, Any]) -> str:
    for horizon in PRIMARY_HORIZON_ORDER:
        if row.get(f"directional_perf_{horizon}") is not None:
            return horizon
    return ""


def build_input(row: dict[str, Any]) -> str:
    direction = "SHORT" if row["is_short"] else "LONG"
    fields = [
        f"Idea date: {row['publication_date']}",
        f"Direction: {direction}",
        f"Company/security: {row['company_name']} ({row['eodhd_symbol']})",
        f"Sector: {row.get('fundamentals_sector') or 'unknown'}",
        f"Industry: {row.get('fundamentals_industry') or 'unknown'}",
        f"Market cap: {row.get('fundamentals_market_cap') or 'unknown'}",
        f"Revenue TTM: {row.get('fundamentals_revenue_ttm') or 'unknown'}",
        f"Profit margin: {row.get('fundamentals_profit_margin') or 'unknown'}",
    ]
    if row.get("catalyst"):
        fields.append(f"Catalyst notes:\n{truncate_middle(row['catalyst'], SFT_CATALYST_CHAR_LIMIT)}")
    fields.append(f"Investment memo:\n{truncate_middle(row['description'], SFT_DESCRIPTION_CHAR_LIMIT)}")
    return "\n\n".join(fields)


def rounded_multiplier(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def build_target(row: dict[str, Any]) -> str:
    """Return the supervised answer as parseable JSON.

    The task is intentionally centered on the 3-year direction-adjusted return
    because closeness to the actual return is more informative than a coarse
    five-class label alone.
    """
    payload = {
        "schema_version": "return_regression_v1",
        "horizon": "3y",
        "direction": "short" if row["is_short"] else "long",
        "raw_stock_multiplier_3y": rounded_multiplier(row.get("raw_perf_3y")),
        "direction_adjusted_multiplier_3y": rounded_multiplier(row.get("directional_perf_3y")),
        "outcome_3y": row.get("outcome_3y"),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def read_training_rows() -> list[dict[str, str]]:
    with TRAINING_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_date(value: str) -> date:
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows = sorted(rows, key=lambda item: (item["publication_date"], item["idea_id"]))
    n = len(rows)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    return {
        "train": rows[:train_end],
        "val": rows[train_end:val_end],
        "test": rows[val_end:],
    }


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    descriptions, catalysts, ideas = parse_copy_sections(SQL_DUMP)
    training_rows = read_training_rows()
    direction_overrides = read_direction_overrides()

    canonical: list[dict[str, Any]] = []
    missing_description = 0
    missing_idea_meta = 0
    for row in training_rows:
        idea_id = row["idea_id"]
        meta = ideas.get(idea_id)
        if not meta:
            missing_idea_meta += 1
            continue

        override = direction_overrides.get(idea_id)
        is_short = bool(override.get("is_short")) if override else meta.is_short

        description = clean_text(descriptions.get(idea_id))
        catalyst = clean_text(catalysts.get(idea_id))
        if not description:
            missing_description += 1

        item: dict[str, Any] = {
            "idea_id": idea_id,
            "raw_symbol": row.get("raw_symbol", ""),
            "eodhd_symbol": row.get("eodhd_symbol", ""),
            "company_name": row.get("fundamentals_name") or row.get("raw_symbol", ""),
            "publication_date": row.get("publication_date") or meta.publication_date,
            "is_short": is_short,
            "is_contest_winner": meta.is_contest_winner,
            "link": meta.link,
            "author_user_id": meta.user_id,
            "description": description,
            "catalyst": catalyst,
            "fundamentals_sector": row.get("fundamentals_sector", ""),
            "fundamentals_industry": row.get("fundamentals_industry", ""),
            "fundamentals_market_cap": row.get("fundamentals_market_cap", ""),
            "fundamentals_revenue_ttm": row.get("fundamentals_revenue_ttm", ""),
            "fundamentals_profit_margin": row.get("fundamentals_profit_margin", ""),
            "math_validation_status": row.get("math_validation_status", ""),
            "review_status": row.get("review_status", ""),
            "reviewed_at": row.get("reviewed_at", ""),
        }
        for horizon in HORIZONS:
            raw = parse_float(row.get(f"validated_perf_{horizon}"))
            directional = (1 / raw if is_short and raw else raw) if raw is not None else None
            item[f"raw_perf_{horizon}"] = raw
            item[f"directional_perf_{horizon}"] = directional
            item[f"outcome_{horizon}"] = outcome_bucket(directional)
        item["primary_horizon"] = choose_primary_horizon(item)
        item["primary_outcome"] = item.get(f"outcome_{item['primary_horizon']}", "missing") if item["primary_horizon"] else "missing"
        canonical.append(item)

    canonical = [row for row in canonical if row["description"]]
    sft_rows = [row for row in canonical if row.get("directional_perf_3y") is not None]
    splits = split_rows(sft_rows)

    canonical_jsonl = OUT_DIR / "investment_canonical.jsonl"
    with canonical_jsonl.open("w", encoding="utf-8") as handle:
        for row in canonical:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    canonical_csv = OUT_DIR / "investment_canonical.csv"
    fieldnames = list(canonical[0].keys()) if canonical else []
    with canonical_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in canonical:
            writer.writerow(row)

    for split_name, split_items in splits.items():
        split_path = OUT_DIR / f"investment_{split_name}.jsonl"
        with split_path.open("w", encoding="utf-8") as handle:
            for row in split_items:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an investment research assistant trained to analyze historical "
                            "investment memos and predict validated future 3-year outcomes. You must "
                            "use only information known at publication time and return parseable JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Analyze this historical investment memo and predict the 3-year "
                            "direction-adjusted return. Return JSON with schema_version, horizon, "
                            "direction, raw_stock_multiplier_3y, direction_adjusted_multiplier_3y, "
                            "and outcome_3y.\n\n"
                            + build_input(row)
                        ),
                    },
                    {"role": "assistant", "content": build_target(row)},
                ]
                handle.write(json.dumps({"messages": messages, "metadata": {"idea_id": row["idea_id"], "split": split_name}}, ensure_ascii=False) + "\n")

    summary = {
        "source_training_csv": str(TRAINING_CSV),
        "source_sql_dump": str(SQL_DUMP),
        "raw_training_rows": len(training_rows),
        "canonical_rows_with_text": len(canonical),
        "sft_rows_with_3y_target": len(sft_rows),
        "missing_idea_meta": missing_idea_meta,
        "missing_description_before_filter": missing_description,
        "split_counts": {name: len(items) for name, items in splits.items()},
        "date_ranges": {
            name: [
                items[0]["publication_date"] if items else None,
                items[-1]["publication_date"] if items else None,
            ]
            for name, items in splits.items()
        },
        "direction_counts": {
            "long": sum(1 for row in canonical if not row["is_short"]),
            "short": sum(1 for row in canonical if row["is_short"]),
        },
        "primary_outcome_counts": {},
        "horizon_nonmissing_counts": {},
    }
    for row in canonical:
        summary["primary_outcome_counts"][row["primary_outcome"]] = summary["primary_outcome_counts"].get(row["primary_outcome"], 0) + 1
    for horizon in HORIZONS:
        summary["horizon_nonmissing_counts"][horizon] = sum(1 for row in canonical if row.get(f"raw_perf_{horizon}") is not None)

    summary_path = REPORTS_DIR / "dataset_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    markdown = [
        "# Investment Fine-Tuning Dataset Audit",
        "",
        f"- Raw validated training rows: {summary['raw_training_rows']}",
        f"- Canonical rows with memo text: {summary['canonical_rows_with_text']}",
        f"- SFT rows with 3y return target: {summary['sft_rows_with_3y_target']}",
        f"- Missing idea metadata skipped: {summary['missing_idea_meta']}",
        f"- Missing descriptions before filtering: {summary['missing_description_before_filter']}",
        f"- Long ideas: {summary['direction_counts']['long']}",
        f"- Short ideas: {summary['direction_counts']['short']}",
        "",
        "## Time-Based Splits",
        "",
    ]
    for name, count in summary["split_counts"].items():
        start, end = summary["date_ranges"][name]
        markdown.append(f"- {name}: {count} rows ({start} to {end})")
    markdown.extend(["", "## Horizon Coverage", ""])
    for horizon, count in summary["horizon_nonmissing_counts"].items():
        markdown.append(f"- {horizon}: {count} rows")
    markdown.extend(["", "## Primary Outcome Counts", ""])
    for name, count in sorted(summary["primary_outcome_counts"].items()):
        markdown.append(f"- {name}: {count}")
    markdown.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{canonical_jsonl}`",
            f"- `{canonical_csv}`",
            f"- `{OUT_DIR / 'investment_train.jsonl'}`",
            f"- `{OUT_DIR / 'investment_val.jsonl'}`",
            f"- `{OUT_DIR / 'investment_test.jsonl'}`",
        ]
    )
    (REPORTS_DIR / "dataset_audit.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
