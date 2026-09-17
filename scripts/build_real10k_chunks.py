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

# Whitespace-token chunking for the production data pipeline.
# This is deterministic and token-count based, but not tied to a model-specific tokenizer.
target_tokens = 192
overlap_tokens = 32
step_tokens = target_tokens - overlap_tokens

started = time.perf_counter()

with sql.connect(
    server_hostname=host,
    http_path=http_path,
    access_token=token,
) as conn:
    with conn.cursor() as cur:
        cur.execute(f"USE CATALOG `{catalog}`")
        cur.execute(f"USE SCHEMA `{schema}`")

        cur.execute(f"""
        CREATE OR REPLACE TABLE scientific_chunks_prod
        USING DELTA
        AS
        WITH tokenized AS (
          SELECT
            document_id,
            source,
            split(regexp_replace(trim(abstract), '\\\\s+', ' '), ' ') AS tokens
          FROM scientific_documents_prod
          WHERE abstract IS NOT NULL AND trim(abstract) <> ''
        ),
        positions AS (
          SELECT
            document_id,
            source,
            tokens,
            posexplode(
              sequence(
                0,
                greatest(
                  0,
                  cast(ceil((size(tokens) - {target_tokens}) / {float(step_tokens)}) AS INT)
                )
              )
            ) AS (sequence_pos, chunk_position)
          FROM tokenized
        )
        SELECT
          concat(document_id, '-CHUNK-', lpad(cast(chunk_position AS STRING), 4, '0')) AS chunk_id,
          document_id,
          chunk_position,
          array_join(
            slice(
              tokens,
              (chunk_position * {step_tokens}) + 1,
              {target_tokens}
            ),
            ' '
          ) AS chunk_text,
          size(
            slice(
              tokens,
              (chunk_position * {step_tokens}) + 1,
              {target_tokens}
            )
          ) AS token_count,
          '{target_tokens}_whitespace_tokens_{overlap_tokens}_overlap' AS chunking_strategy,
          source
        FROM positions
        """)

        cur.execute("""
        SELECT
          count(*) AS chunks,
          count(DISTINCT document_id) AS documents,
          round(avg(token_count), 2) AS avg_tokens,
          percentile_approx(token_count, 0.50) AS p50_tokens,
          percentile_approx(token_count, 0.95) AS p95_tokens,
          min(token_count) AS min_tokens,
          max(token_count) AS max_tokens,
          sum(CASE WHEN chunk_text IS NULL OR trim(chunk_text) = '' THEN 1 ELSE 0 END) AS empty_chunks
        FROM scientific_chunks_prod
        """)
        row = cur.fetchone()

elapsed = time.perf_counter() - started

print(f"CHUNKS={row[0]}")
print(f"DOCUMENTS_CHUNKED={row[1]}")
print(f"AVG_TOKENS_PER_CHUNK={row[2]}")
print(f"P50_TOKENS={row[3]}")
print(f"P95_TOKENS={row[4]}")
print(f"MIN_TOKENS={row[5]}")
print(f"MAX_TOKENS={row[6]}")
print(f"EMPTY_CHUNKS={row[7]}")
print(f"CHUNK_BUILD_SECONDS={elapsed:.2f}")
print("REAL10K_CHUNKING=" + ("PASS" if row[1] == 10000 and row[7] == 0 and row[6] <= 192 else "FAIL"))
