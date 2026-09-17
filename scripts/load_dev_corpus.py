from __future__ import annotations

import json
import os
from pathlib import Path

from databricks import sql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

src = ROOT / "data" / "dev_corpus.jsonl"
if not src.exists():
    raise SystemExit(f"Missing {src}; run generate_dev_corpus.py first")

host = os.environ["DATABRICKS_SERVER_HOSTNAME"].replace("https://", "").rstrip("/")
http_path = os.environ["DATABRICKS_HTTP_PATH"]
token = os.environ["DATABRICKS_TOKEN"]
catalog = os.environ.get("DATABRICKS_CATALOG", "p14_rag_llmops")
schema = os.environ.get("DATABRICKS_SCHEMA", "default")

records = []
with src.open("r", encoding="utf-8") as fh:
    for line in fh:
        records.append(json.loads(line))

with sql.connect(
    server_hostname=host,
    http_path=http_path,
    access_token=token,
) as conn:
    with conn.cursor() as cur:
        cur.execute(f"USE CATALOG `{catalog}`")
        cur.execute(f"USE SCHEMA `{schema}`")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS scientific_documents_dev (
            document_id STRING,
            title STRING,
            abstract STRING,
            body STRING,
            category STRING,
            year INT,
            source STRING
        )
        USING DELTA
        """)

        cur.execute("TRUNCATE TABLE scientific_documents_dev")

        insert_sql = """
        INSERT INTO scientific_documents_dev
        (document_id, title, abstract, body, category, year, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        batch_size = 250
        total = len(records)
        for start in range(0, total, batch_size):
            batch = records[start:start + batch_size]
            params = [
                (
                    r["document_id"],
                    r["title"],
                    r["abstract"],
                    r["body"],
                    r["category"],
                    int(r["year"]),
                    r["source"],
                )
                for r in batch
            ]
            cur.executemany(insert_sql, params)
            print(f"LOADED={min(start + len(batch), total)}/{total}")

        cur.execute("""
        SELECT
            count(*) AS rows,
            count(DISTINCT document_id) AS distinct_documents,
            min(year) AS min_year,
            max(year) AS max_year,
            count(DISTINCT category) AS categories
        FROM scientific_documents_dev
        """)
        row = cur.fetchone()

print(f"DATABRICKS_DOCUMENT_ROWS={row[0]}")
print(f"DISTINCT_DOCUMENTS={row[1]}")
print(f"YEAR_RANGE={row[2]}-{row[3]}")
print(f"CATEGORIES={row[4]}")
print("DEV_CORPUS_LOAD=" + ("PASS" if row[0] == total and row[1] == total else "FAIL"))
