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

start = time.perf_counter()

with sql.connect(
    server_hostname=host,
    http_path=http_path,
    access_token=token,
) as conn:
    with conn.cursor() as cur:
        cur.execute(f"USE CATALOG `{catalog}`")
        cur.execute(f"USE SCHEMA `{schema}`")

        cur.execute("""
        CREATE OR REPLACE TABLE scientific_documents_dev
        USING DELTA
        AS
        WITH base AS (
          SELECT
            id,
            CASE pmod(id, 8)
              WHEN 0 THEN 'machine_learning'
              WHEN 1 THEN 'distributed_systems'
              WHEN 2 THEN 'databases'
              WHEN 3 THEN 'biology'
              WHEN 4 THEN 'physics'
              WHEN 5 THEN 'climate'
              WHEN 6 THEN 'medicine'
              ELSE 'robotics'
            END AS category
          FROM range(10000)
        ),
        docs AS (
          SELECT
            id,
            category,
            concat(
              'Study ', cast(id AS STRING), ': ',
              replace(category, '_', ' '),
              ' methods and scalable experimental analysis'
            ) AS title,
            concat(
              'This scientific study investigates ',
              replace(category, '_', ' '),
              ' using controlled experiments, measurable baselines, reproducible methods, ',
              'quality evaluation, latency analysis, and computational trade-offs. ',
              'The work compares multiple approaches and reports evidence for scientific retrieval.'
            ) AS abstract
          FROM base
        )
        SELECT
          concat('DOC-', lpad(cast(id AS STRING), 9, '0')) AS document_id,
          title,
          abstract,
          repeat(concat(abstract, ' '), 12 + pmod(id, 5)) AS body,
          category,
          cast(2015 + pmod(id, 12) AS INT) AS year,
          'project14_dev_corpus' AS source
        FROM docs
        """)

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

elapsed = time.perf_counter() - start

print(f"DATABRICKS_DOCUMENT_ROWS={row[0]}")
print(f"DISTINCT_DOCUMENTS={row[1]}")
print(f"YEAR_RANGE={row[2]}-{row[3]}")
print(f"CATEGORIES={row[4]}")
print(f"LOAD_SECONDS={elapsed:.2f}")
print("DEV_CORPUS_LOAD=" + ("PASS" if row[0] == 10000 and row[1] == 10000 else "FAIL"))
