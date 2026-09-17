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

chunk_chars = 1800
overlap_chars = 200
step_chars = chunk_chars - overlap_chars

with sql.connect(
    server_hostname=host,
    http_path=http_path,
    access_token=token,
) as conn:
    with conn.cursor() as cur:
        cur.execute(f"USE CATALOG `{catalog}`")
        cur.execute(f"USE SCHEMA `{schema}`")

        cur.execute(f"""
        CREATE OR REPLACE TABLE scientific_chunks_dev
        USING DELTA
        AS
        SELECT
            d.document_id,
            concat(d.document_id, '-CHUNK-', lpad(cast(chunk_position AS STRING), 4, '0')) AS chunk_id,
            d.title,
            d.category,
            d.year,
            chunk_position,
            substring(
                d.body,
                (chunk_position * {step_chars}) + 1,
                {chunk_chars}
            ) AS chunk_text,
            length(
                substring(
                    d.body,
                    (chunk_position * {step_chars}) + 1,
                    {chunk_chars}
                )
            ) AS chunk_char_count,
            '{chunk_chars}_chars_{overlap_chars}_overlap' AS chunking_strategy
        FROM scientific_documents_dev d
        LATERAL VIEW explode(
            sequence(
                0,
                greatest(
                    0,
                    cast(ceil((length(d.body) - {chunk_chars}) / {float(step_chars)}) AS INT)
                )
            )
        ) e AS chunk_position
        """)

        cur.execute("""
        SELECT
            count(*) AS chunks,
            count(DISTINCT document_id) AS documents,
            round(avg(chunk_char_count), 2) AS avg_chunk_chars,
            min(chunk_char_count) AS min_chunk_chars,
            max(chunk_char_count) AS max_chunk_chars
        FROM scientific_chunks_dev
        """)
        summary = cur.fetchone()

        cur.execute("""
        SELECT category, count(*) AS chunks
        FROM scientific_chunks_dev
        GROUP BY category
        ORDER BY category
        """)
        categories = cur.fetchall()

print(f"CHUNK_ROWS={summary[0]}")
print(f"DOCUMENTS_CHUNKED={summary[1]}")
print(f"AVG_CHUNK_CHARS={summary[2]}")
print(f"MIN_CHUNK_CHARS={summary[3]}")
print(f"MAX_CHUNK_CHARS={summary[4]}")
for category, count in categories:
    print(f"CATEGORY_{category}={count}")
print("DEV_CHUNKING=" + ("PASS" if summary[1] == 10000 and summary[0] > 10000 else "FAIL"))
