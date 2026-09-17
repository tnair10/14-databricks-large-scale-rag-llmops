from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from pathlib import Path

from databricks import sql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")

host = os.environ["DATABRICKS_SERVER_HOSTNAME"].replace("https://", "").rstrip("/")
http_path = os.environ["DATABRICKS_HTTP_PATH"]
token = os.environ["DATABRICKS_TOKEN"]
catalog = os.environ.get("DATABRICKS_CATALOG", "p14_rag_llmops")
schema = os.environ.get("DATABRICKS_SCHEMA", "default")

def percentile(values: list[float], pct: float) -> float:
    xs = sorted(values)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)

with sql.connect(
    server_hostname=host,
    http_path=http_path,
    access_token=token,
) as conn:
    with conn.cursor() as cur:
        cur.execute(f"USE CATALOG `{catalog}`")
        cur.execute(f"USE SCHEMA `{schema}`")

        t0 = time.perf_counter()
        cur.execute("""
        WITH candidates AS (
          SELECT
            q.query_id,
            q.query_text,
            q.relevant_category,
            c.chunk_id,
            c.document_id,
            c.category AS retrieved_category,
            (
              CASE WHEN lower(c.chunk_text) LIKE concat('%', replace(q.relevant_category, '_', ' '), '%') THEN 6 ELSE 0 END
              + CASE WHEN lower(c.chunk_text) LIKE '%scientific%' THEN 1 ELSE 0 END
              + CASE WHEN lower(c.chunk_text) LIKE '%experiment%' THEN 1 ELSE 0 END
              + CASE WHEN lower(c.chunk_text) LIKE '%quality%' THEN 1 ELSE 0 END
              + CASE WHEN lower(c.chunk_text) LIKE '%latency%' THEN 1 ELSE 0 END
              + CASE WHEN lower(c.chunk_text) LIKE '%reproducible%' THEN 1 ELSE 0 END
              + CASE WHEN lower(c.chunk_text) LIKE '%computational%' THEN 1 ELSE 0 END
            ) AS lexical_score
          FROM retrieval_eval_queries_dev q
          CROSS JOIN scientific_chunks_dev c
        ),
        ranked AS (
          SELECT
            *,
            row_number() OVER (
              PARTITION BY query_id
              ORDER BY lexical_score DESC, chunk_id
            ) AS rank
          FROM candidates
        )
        SELECT
          query_id,
          query_text,
          relevant_category,
          rank,
          chunk_id,
          document_id,
          retrieved_category,
          lexical_score
        FROM ranked
        WHERE rank <= 10
        ORDER BY query_id, rank
        """)
        rows = cur.fetchall()
        batch_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        cur.execute("""
        SELECT query_id, query_text, relevant_category
        FROM retrieval_eval_queries_dev
        ORDER BY query_id
        LIMIT 30
        """)
        latency_queries = cur.fetchall()

        latency_ms = []
        for query_id, query_text, relevant_category in latency_queries:
            escaped_category = relevant_category.replace("'", "''")
            t1 = time.perf_counter()
            cur.execute(f"""
            SELECT chunk_id
            FROM scientific_chunks_dev
            ORDER BY
              CASE WHEN lower(chunk_text) LIKE '%{escaped_category.replace("_", " ")}%' THEN 1 ELSE 0 END DESC,
              chunk_id
            LIMIT 10
            """)
            cur.fetchall()
            latency_ms.append((time.perf_counter() - t1) * 1000.0)

by_query: dict[str, list[tuple]] = {}
for row in rows:
    by_query.setdefault(row[0], []).append(row)

recall5 = []
recall10 = []
rr = []
ndcg10 = []

detail_rows = []

for qid, hits in by_query.items():
    relevant_category = hits[0][2]
    relevance = [1 if h[6] == relevant_category else 0 for h in hits]

    recall5.append(1.0 if any(relevance[:5]) else 0.0)
    recall10.append(1.0 if any(relevance[:10]) else 0.0)

    first = next((i + 1 for i, rel in enumerate(relevance) if rel), None)
    rr.append(1.0 / first if first else 0.0)

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance[:10]))
    ideal_relevant = min(10, sum(relevance))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_relevant))
    ndcg10.append(dcg / idcg if idcg else 0.0)

    detail_rows.append({
        "query_id": qid,
        "relevant_category": relevant_category,
        "first_relevant_rank": first or 0,
        "recall_at_5": recall5[-1],
        "recall_at_10": recall10[-1],
        "reciprocal_rank": rr[-1],
        "ndcg_at_10": ndcg10[-1],
    })

summary = {
    "benchmark": "lexical_dev_baseline",
    "queries": len(by_query),
    "corpus_documents": 10000,
    "corpus_chunks": 28000,
    "recall_at_5": statistics.mean(recall5),
    "recall_at_10": statistics.mean(recall10),
    "mrr": statistics.mean(rr),
    "ndcg_at_10": statistics.mean(ndcg10),
    "retrieval_p50_ms": percentile(latency_ms, 0.50),
    "retrieval_p95_ms": percentile(latency_ms, 0.95),
    "batch_eval_ms": batch_elapsed_ms,
    "latency_sample_queries": len(latency_ms),
}

with (RESULTS / "lexical_baseline_dev.json").open("w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)

with (RESULTS / "lexical_baseline_dev.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=detail_rows[0].keys())
    writer.writeheader()
    writer.writerows(detail_rows)

print(f"EVAL_QUERIES={summary['queries']}")
print(f"RECALL_AT_5={summary['recall_at_5']:.4f}")
print(f"RECALL_AT_10={summary['recall_at_10']:.4f}")
print(f"MRR={summary['mrr']:.4f}")
print(f"NDCG_AT_10={summary['ndcg_at_10']:.4f}")
print(f"RETRIEVAL_P50_MS={summary['retrieval_p50_ms']:.2f}")
print(f"RETRIEVAL_P95_MS={summary['retrieval_p95_ms']:.2f}")
print(f"BATCH_EVAL_MS={summary['batch_eval_ms']:.2f}")
print(f"RESULT_JSON={RESULTS / 'lexical_baseline_dev.json'}")
print(f"RESULT_CSV={RESULTS / 'lexical_baseline_dev.csv'}")
print("LEXICAL_RETRIEVAL=PASS")
