from __future__ import annotations

import os
from databricks import sql
from dotenv import load_dotenv

load_dotenv()

host = os.environ["DATABRICKS_SERVER_HOSTNAME"].replace("https://", "").rstrip("/")
http_path = os.environ["DATABRICKS_HTTP_PATH"]
token = os.environ["DATABRICKS_TOKEN"]
catalog = os.environ.get("DATABRICKS_CATALOG", "p14_rag_llmops")
schema = os.environ.get("DATABRICKS_SCHEMA", "default")

with sql.connect(
    server_hostname=host,
    http_path=http_path,
    access_token=token,
) as conn:
    with conn.cursor() as cur:
        cur.execute(f"USE CATALOG `{catalog}`")
        cur.execute(f"USE SCHEMA `{schema}`")

        cur.execute("""
        CREATE OR REPLACE TABLE retrieval_eval_queries_dev
        USING DELTA
        AS
        WITH q AS (
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
            END AS relevant_category,
            pmod(id, 5) AS phrasing
          FROM range(500)
        )
        SELECT
          concat('Q-', lpad(cast(id AS STRING), 5, '0')) AS query_id,
          relevant_category,
          CASE phrasing
            WHEN 0 THEN concat('scientific research on ', replace(relevant_category, '_', ' '), ' methods')
            WHEN 1 THEN concat('scalable experiments for ', replace(relevant_category, '_', ' '))
            WHEN 2 THEN concat('quality and latency analysis in ', replace(relevant_category, '_', ' '))
            WHEN 3 THEN concat('reproducible evaluation of ', replace(relevant_category, '_', ' '), ' approaches')
            ELSE concat('computational trade offs for ', replace(relevant_category, '_', ' '), ' systems')
          END AS query_text
        FROM q
        """)

        cur.execute("""
        SELECT
          count(*) AS rows,
          count(DISTINCT query_id) AS distinct_queries,
          count(DISTINCT relevant_category) AS categories
        FROM retrieval_eval_queries_dev
        """)
        row = cur.fetchone()

print(f"EVAL_QUERIES={row[0]}")
print(f"DISTINCT_QUERIES={row[1]}")
print(f"GROUND_TRUTH_CATEGORIES={row[2]}")
print("GROUND_TRUTH=" + ("PASS" if row[0] == 500 and row[1] == 500 and row[2] == 8 else "FAIL"))
