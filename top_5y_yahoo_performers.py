from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any


PROJECT_REF = "aqfpldvpcoyipkyihuea"
SUPABASE_URL = f"https://{PROJECT_REF}.supabase.co/rest/v1"


def service_key() -> str:
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if key:
        return key
    raise RuntimeError("Set SUPABASE_SERVICE_ROLE_KEY before running top_5y_yahoo_performers.py")


def get(path: str, query: dict[str, str]) -> list[dict[str, Any]]:
    key = service_key()
    url = f"{SUPABASE_URL}/{path}?{urllib.parse.urlencode(query)}"
    req = urllib.request.Request(url, headers={"apikey": key, "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    rows = get(
        "performance_yahoo",
        {
            "select": "idea_id,raw_symbol,yahoo_symbol,publication_date,base_adj_close,trade_date_5y,adj_price_5y,perf_5y,source_status",
            "source_status": "eq.ok",
            "perf_5y": "not.is.null",
            "order": "perf_5y.desc",
            "limit": "5",
        },
    )
    for row in rows:
        ideas = get("ideas", {"select": "id,company_id,date,is_short", "id": f"eq.{row['idea_id']}", "limit": "1"})
        row["idea"] = ideas[0] if ideas else {}
    sane: list[dict[str, Any]] = []
    offset = 0
    while len(sane) < 5:
        candidates = get(
            "performance_yahoo",
            {
                "select": "idea_id,raw_symbol,yahoo_symbol,publication_date,base_adj_close,trade_date_5y,adj_price_5y,perf_5y,source_status",
                "source_status": "eq.ok",
                "perf_5y": "not.is.null",
                "perf_5y": "lt.100",
                "order": "perf_5y.desc",
                "limit": "50",
                "offset": str(offset),
            },
        )
        if not candidates:
            break
        for candidate in candidates:
            ideas = get("ideas", {"select": "id,company_id,date,is_short", "id": f"eq.{candidate['idea_id']}", "limit": "1"})
            idea = ideas[0] if ideas else {}
            candidate["idea"] = idea
            if idea.get("is_short") is False:
                sane.append(candidate)
                if len(sane) == 5:
                    break
        offset += 50
    print(json.dumps({"raw_top_5": rows, "sane_long_top_5_under_100x": sane}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
