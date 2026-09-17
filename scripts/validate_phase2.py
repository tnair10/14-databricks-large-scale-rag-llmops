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
        SELECT
            (SELECT count(*) FROM scientific_documents_dev) AS docs,
            (SELECT count(*) FROM scientific_chunks_dev) AS chunks,
            (SELECT count(*) FROM scientific_chunks_dev WHERE chunk_text IS NULL OR trim(chunk_text) = '') AS empty_chunks,
            (SELECT count(*) FROM scientific_chunks_dev WHERE chunk_char_count > 1800) AS oversized_chunks,
            (SELECT count(*) FROM scientific_documents_dev d
             LEFT ANTI JOIN scientific_chunks_dev c
             ON d.document_id = c.document_id) AS documents_without_chunks
        """)
        row = cur.fetchone()

docs, chunks, empty_chunks, oversized, missing = row
passed = (
    docs == 10000
    and chunks > docs
    and empty_chunks == 0
    and oversized == 0
    and missing == 0
)

print(f"DOCUMENTS={docs}")
print(f"CHUNKS={chunks}")
print(f"EMPTY_CHUNKS={empty_chunks}")
print(f"OVERSIZED_CHUNKS={oversized}")
print(f"DOCUMENTS_WITHOUT_CHUNKS={missing}")
print("PHASE2_VALIDATION=" + ("PASS" if passed else "FAIL"))
