from __future__ import annotations

import os
import time
from databricks import sql
from dotenv import load_dotenv

load_dotenv()

host = os.environ["DATABRICKS_SERVER_HOSTNAME"].replace("https://", "").rstrip("/")
http_path = os.environ["DATABRICKS_HTTP_PATH"]
token = os.environ["DATABRICKS_TOKEN"]
catalog = os.environ.get("DATABRICKS_CATALOG", "p14_rag_llmops")
schema = os.environ.get("DATABRICKS_SCHEMA", "default")
volume = "p14_stage"
path = f"/Volumes/{catalog}/{schema}/{volume}/arxiv_real_10k.jsonl"

started = time.perf_counter()

with sql.connect(
    server_hostname=host,
    http_path=http_path,
    access_token=token,
) as conn:
    with conn.cursor() as cur:
        cur.execute(f"USE CATALOG `{catalog}`")
        cur.execute(f"USE SCHEMA `{schema}`")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS scientific_documents_prod (
          document_id STRING,
          abstract STRING,
          authors ARRAY<STRING>,
          created STRING,
          source STRING,
          source_url STRING,
          license STRING,
          ingested_at TIMESTAMP
        ) USING DELTA
        """)

        cur.execute(f"""
        CREATE OR REPLACE TEMP VIEW arxiv_real_10k_stage AS
        SELECT
          CAST(document_id AS STRING) AS document_id,
          CAST(abstract AS STRING) AS abstract,
          CAST(authors AS ARRAY<STRING>) AS authors,
          CAST(created AS STRING) AS created,
          CAST(source AS STRING) AS source,
          CAST(source_url AS STRING) AS source_url,
          CAST(license AS STRING) AS license
        FROM read_files(
          '{path}',
          format => 'json'
        )
        """)

        cur.execute("TRUNCATE TABLE scientific_documents_prod")

        cur.execute("""
        INSERT INTO scientific_documents_prod
        SELECT
          document_id,
          abstract,
          authors,
          created,
          source,
          source_url,
          license,
          current_timestamp()
        FROM arxiv_real_10k_stage
        """)

        cur.execute("""
        SELECT
          count(*) AS rows,
          count(DISTINCT document_id) AS distinct_docs,
          sum(CASE WHEN abstract IS NULL OR trim(abstract) = '' THEN 1 ELSE 0 END) AS empty_abstracts,
          sum(CASE WHEN size(authors) > 0 THEN 1 ELSE 0 END) AS with_authors
        FROM scientific_documents_prod
        """)
        row = cur.fetchone()

elapsed = time.perf_counter() - started

print(f"DELTA_ROWS={row[0]}")
print(f"DISTINCT_DOCUMENTS={row[1]}")
print(f"EMPTY_ABSTRACTS={row[2]}")
print(f"WITH_AUTHORS={row[3]}")
print(f"LOAD_SECONDS={elapsed:.2f}")
print("REAL10K_DELTA_LOAD=" + ("PASS" if row[0] == 10000 and row[1] == 10000 and row[2] == 0 else "FAIL"))
