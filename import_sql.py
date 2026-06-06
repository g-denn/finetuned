import psycopg2
import io
import os
import re

password = os.environ.get("SUPABASE_DB_PASSWORD")
if not password:
    raise SystemExit("Set SUPABASE_DB_PASSWORD before running import_sql.py")

conn = psycopg2.connect(
    host="aws-0-ap-southeast-1.pooler.supabase.com",
    port=5432,
    dbname="postgres",
    user="postgres.aqfpldvpcoyipkyihuea",
    password=password,
    sslmode="require",
    connect_timeout=30
)
conn.autocommit = False
cur = conn.cursor()
print("Connected.")

with open("VIC_IDEAS.sql", "r", encoding="utf-8") as f:
    lines = f.readlines()

i = 0
sql_batch = []
while i < len(lines):
    line = lines[i]

    if line.startswith("COPY ") and "FROM stdin" in line:
        # Flush pending SQL first
        if sql_batch:
            stmt = "".join(sql_batch).strip()
            if stmt:
                try:
                    cur.execute(stmt)
                except Exception as e:
                    if "already exists" not in str(e):
                        print(f"SQL warn: {e}")
                    conn.rollback()
                    conn.autocommit = False
            sql_batch = []

        # Collect COPY data until '\.'
        copy_header = line.strip()  # e.g. COPY public.catalyst (idea_id, catalysts) FROM stdin;
        copy_cmd = copy_header.replace("FROM stdin;", "FROM STDIN")
        data_lines = []
        i += 1
        while i < len(lines) and lines[i].rstrip("\n") != "\\.":
            data_lines.append(lines[i])
            i += 1

        data = "".join(data_lines)
        try:
            cur.copy_expert(copy_cmd, io.StringIO(data))
            conn.commit()
            table = re.search(r"COPY (\S+)", copy_cmd).group(1)
            print(f"  Imported {table}: {len(data_lines):,} rows")
        except Exception as e:
            conn.rollback()
            print(f"  COPY error on {copy_cmd[:60]}: {e}")
    else:
        # Skip SET/ALTER/owner lines that would fail on Supabase
        skip = any(line.startswith(p) for p in [
            "ALTER TABLE", "ALTER SCHEMA", "ALTER SEQUENCE",
            "REVOKE", "GRANT", "CREATE ROLE", "SET role",
            "OWNER TO"
        ])
        if not skip:
            sql_batch.append(line)
    i += 1

# Flush any remaining SQL
if sql_batch:
    stmt = "".join(sql_batch).strip()
    if stmt:
        try:
            cur.execute(stmt)
            conn.commit()
        except Exception as e:
            if "already exists" not in str(e):
                print(f"Final SQL warn: {e}")
            conn.rollback()

print("\nRow counts:")
for table in ["ideas", "companies", "descriptions", "catalyst", "performance", "users"]:
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    print(f"  {table}: {cur.fetchone()[0]:,}")

cur.close()
conn.close()
print("Done.")
