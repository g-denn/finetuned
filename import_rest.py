import requests
import json
import os
import re
import sys

PROJECT_REF = "aqfpldvpcoyipkyihuea"
SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
if not SERVICE_KEY:
    raise SystemExit("Set SUPABASE_SERVICE_ROLE_KEY before running import_rest.py")
BASE_URL = f"https://{PROJECT_REF}.supabase.co/rest/v1"
HEADERS = {
    "Authorization": f"Bearer {SERVICE_KEY}",
    "apikey": SERVICE_KEY,
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}
BATCH = 500

def cast(val, col_types, col):
    if val == "\\N":
        return None
    t = col_types.get(col, "text")
    if t in ("float8", "float4", "numeric"):
        return float(val)
    if t == "bool":
        return val == "t"
    if t in ("int4", "int8", "int2"):
        return int(val)
    return val

# column types per table (from schema)
TABLE_TYPES = {
    "catalyst": {},
    "companies": {},
    "descriptions": {},
    "ideas": {"is_short": "bool", "is_contest_winner": "bool"},
    "performance": {c: "float8" for c in [
        "nextDayOpen","nextDayClose","oneWeekClosePerf","twoWeekClosePerf",
        "oneMonthPerf","threeMonthPerf","sixMonthPerf","oneYearPerf",
        "twoYearPerf","threeYearPerf","fiveYearPerf"
    ]},
    "users": {},
}

def insert_batch(table, rows):
    r = requests.post(f"{BASE_URL}/{table}", headers=HEADERS, json=rows, timeout=60)
    if r.status_code not in (200, 201):
        print(f"  ERROR {r.status_code}: {r.text[:200]}")
        return False
    return True

print("Reading SQL file...")
with open("VIC_IDEAS.sql", "r", encoding="utf-8") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i]
    m = re.match(r'COPY public\.(\w+) \((.+?)\) FROM stdin;', line)
    if m:
        table = m.group(1)
        # strip quotes from column names
        cols = [c.strip().strip('"') for c in m.group(2).split(",")]
        col_types = TABLE_TYPES.get(table, {})

        data_lines = []
        i += 1
        while i < len(lines) and lines[i].rstrip("\n") != "\\.":
            data_lines.append(lines[i].rstrip("\n"))
            i += 1

        total = len(data_lines)
        print(f"Importing {table} ({total:,} rows)...")

        inserted = 0
        for b_start in range(0, total, BATCH):
            batch_lines = data_lines[b_start:b_start + BATCH]
            rows = []
            for dl in batch_lines:
                vals = dl.split("\t")
                row = {cols[j]: cast(vals[j], col_types, cols[j]) for j in range(len(cols))}
                rows.append(row)
            if not insert_batch(table, rows):
                print(f"  Aborted at batch {b_start}")
                sys.exit(1)
            inserted += len(rows)
            print(f"  {inserted:,}/{total:,}", end="\r")
        print(f"  {table}: {inserted:,} rows done      ")
    i += 1

print("\nVerifying row counts...")
for table in ["ideas", "companies", "descriptions", "catalyst", "performance", "users"]:
    r = requests.get(
        f"{BASE_URL}/{table}?select=*",
        headers={**HEADERS, "Prefer": "count=exact", "Range": "0-0"},
        timeout=30
    )
    count = r.headers.get("content-range", "?/?").split("/")[-1]
    print(f"  {table}: {count} rows")

print("Done!")
