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
path = f"/Volumes/{catalog}/{schema}/p14_stage/real_retrieval_eval_500.jsonl"

with sql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
    with conn.cursor() as cur:
        cur.execute(f"USE CATALOG `{catalog}`")
        cur.execute(f"USE SCHEMA `{schema}`")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS retrieval_eval_prod (
          query_id STRING,
          query_text STRING,
          relevant_document_ids ARRAY<STRING>,
          difficulty STRING,
          source STRING,
          created_at TIMESTAMP
        ) USING DELTA
        """)
        cur.execute("TRUNCATE TABLE retrieval_eval_prod")
        cur.execute(f"""
        INSERT INTO retrieval_eval_prod
        SELECT
          CAST(query_id AS STRING),
          CAST(query_text AS STRING),
          CAST(relevant_document_ids AS ARRAY<STRING>),
          CAST(difficulty AS STRING),
          CAST(source AS STRING),
          current_timestamp()
        FROM read_files('{path}', format => 'json')
        """)
        cur.execute("""
        SELECT
          count(*) AS rows,
          count(DISTINCT query_id) AS distinct_queries,
          count(DISTINCT element_at(relevant_document_ids, 1)) AS distinct_docs,
          sum(CASE WHEN query_text IS NULL OR trim(query_text)='' THEN 1 ELSE 0 END) AS empty_queries
        FROM retrieval_eval_prod
        """)
        row = cur.fetchone()

print(f"EVAL_ROWS={row[0]}")
print(f"DISTINCT_QUERIES={row[1]}")
print(f"DISTINCT_RELEVANT_DOCS={row[2]}")
print(f"EMPTY_QUERIES={row[3]}")
print("REAL_EVAL_DELTA_LOAD=" + ("PASS" if row[0] == 500 and row[1] == 500 and row[3] == 0 else "FAIL"))
