from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import date, timedelta
from typing import Any

from dateutil.relativedelta import relativedelta

from fetch_yahoo_performance_adjusted import bar_on_or_after, parse_bars, yahoo_chart


PROJECT_REF = "aqfpldvpcoyipkyihuea"
SUPABASE_URL = f"https://{PROJECT_REF}.supabase.co"
LABELS: dict[str, Any] = {
    "1w": timedelta(weeks=1),
    "2w": timedelta(weeks=2),
    "1m": relativedelta(months=1),
    "3m": relativedelta(months=3),
    "6m": relativedelta(months=6),
    "1y": relativedelta(years=1),
    "2y": relativedelta(years=2),
    "3y": relativedelta(years=3),
    "5y": relativedelta(years=5),
}


def service_key() -> str:
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if key:
        return key
    raise RuntimeError("Set SUPABASE_SERVICE_ROLE_KEY before running verify_aapl_yahoo_rows.py")


def fetch_aapl_rows() -> list[dict[str, Any]]:
    columns = ["idea_id", "raw_symbol", "yahoo_symbol", "publication_date", "base_trade_date", "base_adj_close"]
    for label in LABELS:
        columns.extend([f"adj_price_{label}", f"perf_{label}", f"short_perf_{label}", f"trade_date_{label}"])
    query = urllib.parse.urlencode(
        {
            "select": ",".join(columns),
            "yahoo_symbol": "eq.AAPL",
            "source_status": "eq.ok",
            "order": "publication_date.asc",
            "limit": "500",
        }
    )
    req = urllib.request.Request(
        f"{SUPABASE_URL}/rest/v1/performance_yahoo?{query}",
        headers={"apikey": service_key(), "Authorization": f"Bearer {service_key()}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def close_enough(left: float | None, right: float | None) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) <= max(1e-5, 1e-5 * abs(float(right)))


def main() -> int:
    rows = fetch_aapl_rows()
    print(f"AAPL rows: {len(rows)}")
    if not rows:
        return 0

    start = min(date.fromisoformat(row["publication_date"]) for row in rows) - timedelta(days=10)
    end = min(max(date.fromisoformat(row["publication_date"]) for row in rows) + relativedelta(years=5, days=14), date.today())
    bars = parse_bars(yahoo_chart("AAPL", start, end))
    print(f"Fresh Yahoo bars: {len(bars)} ({start} to {end})")

    failures: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    for row in rows:
        pub_date = date.fromisoformat(row["publication_date"])
        base = bar_on_or_after(bars, pub_date)
        base_fresh = base.adj_close if base else None
        item: dict[str, Any] = {
            "idea_id": row["idea_id"],
            "publication_date": row["publication_date"],
            "base_trade_date": row["base_trade_date"],
            "base_adj_close": row["base_adj_close"],
            "base_adj_fresh": base_fresh,
            "horizons": {},
        }
        if not close_enough(row.get("base_adj_close"), base_fresh):
            failures.append({"idea_id": row["idea_id"], "field": "base_adj_close", "stored": row.get("base_adj_close"), "fresh": base_fresh})

        for label, delta in LABELS.items():
            future = bar_on_or_after(bars, pub_date + delta)
            fresh_adj = future.adj_close if future else None
            fresh_perf = fresh_adj / base_fresh if fresh_adj and base_fresh else None
            fresh_short = 1 / fresh_perf if fresh_perf else None
            ok = close_enough(row.get(f"perf_{label}"), fresh_perf) and close_enough(row.get(f"short_perf_{label}"), fresh_short)
            item["horizons"][label] = {
                "trade_date": row.get(f"trade_date_{label}"),
                "stored_perf": row.get(f"perf_{label}"),
                "fresh_perf": fresh_perf,
                "stored_short": row.get(f"short_perf_{label}"),
                "fresh_short": fresh_short,
                "ok": ok,
            }
            if not ok:
                failures.append({"idea_id": row["idea_id"], "field": label, "stored": row.get(f"perf_{label}"), "fresh": fresh_perf})
        compact_rows.append(item)

    print(json.dumps({"failed": len(failures), "failures": failures, "rows": compact_rows}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
