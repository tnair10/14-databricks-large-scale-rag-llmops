from __future__ import annotations

import os
from pathlib import Path

import mlflow
from databricks import sql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

host = os.environ["DATABRICKS_SERVER_HOSTNAME"].replace("https://", "").rstrip("/")
http_path = os.environ["DATABRICKS_HTTP_PATH"]
token = os.environ["DATABRICKS_TOKEN"]
catalog = os.environ.get("DATABRICKS_CATALOG", "p14_rag_llmops")
schema = os.environ.get("DATABRICKS_SCHEMA", "default")

os.environ["DATABRICKS_HOST"] = "https://" + host
os.environ["DATABRICKS_TOKEN"] = token

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
          (SELECT count(*) FROM scientific_documents_prod) AS documents,
          (SELECT count(*) FROM scientific_chunks_prod) AS chunks,
          (SELECT round(avg(token_count), 4) FROM scientific_chunks_prod) AS avg_tokens,
          (SELECT percentile_approx(token_count, 0.50) FROM scientific_chunks_prod) AS p50_tokens,
          (SELECT percentile_approx(token_count, 0.95) FROM scientific_chunks_prod) AS p95_tokens,
          (SELECT min(token_count) FROM scientific_chunks_prod) AS min_tokens,
          (SELECT max(token_count) FROM scientific_chunks_prod) AS max_tokens,
          (SELECT count(*) FROM scientific_chunks_prod
             WHERE chunk_text IS NULL OR trim(chunk_text) = '') AS empty_chunks,
          (SELECT count(*) FROM scientific_chunks_prod
             WHERE token_count > 192) AS oversized_chunks
        """)
        row = cur.fetchone()

documents, chunks, avg_tokens, p50, p95, min_tokens, max_tokens, empty_chunks, oversized = row

mlflow.set_tracking_uri("databricks")
experiment = "/Shared/project14-large-scale-rag"
mlflow.set_experiment(experiment)

with mlflow.start_run(run_name="real10k_delta_chunking_baseline") as run:
    mlflow.log_params({
        "dataset": "common-pile/arxiv_abstracts",
        "corpus_type": "real",
        "chunking_unit": "whitespace_token",
        "target_tokens": 192,
        "overlap_tokens": 32,
        "step_tokens": 160,
        "documents_table": f"{catalog}.{schema}.scientific_documents_prod",
        "chunks_table": f"{catalog}.{schema}.scientific_chunks_prod",
    })
    mlflow.log_metrics({
        "documents": float(documents),
        "chunks": float(chunks),
        "chunks_per_document": float(chunks) / float(documents),
        "avg_tokens_per_chunk": float(avg_tokens),
        "p50_tokens_per_chunk": float(p50),
        "p95_tokens_per_chunk": float(p95),
        "min_tokens_per_chunk": float(min_tokens),
        "max_tokens_per_chunk": float(max_tokens),
        "empty_chunks": float(empty_chunks),
        "oversized_chunks": float(oversized),
    })
    mlflow.set_tags({
        "project": "project14",
        "phase": "real10k",
        "evidence_type": "measured",
    })
    print(f"TRACKING_URI={mlflow.get_tracking_uri()}")
    print(f"EXPERIMENT={experiment}")
    print(f"MLFLOW_RUN_ID={run.info.run_id}")

print(f"DOCUMENTS={documents}")
print(f"CHUNKS={chunks}")
print(f"CHUNKS_PER_DOCUMENT={chunks/documents:.4f}")
print(f"AVG_TOKENS={avg_tokens}")
print(f"P50_TOKENS={p50}")
print(f"P95_TOKENS={p95}")
print("REAL10K_MLFLOW_LOG=PASS")
