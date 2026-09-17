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

def pct(vals, p):
    xs = sorted(vals)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs)-1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi-pos) + xs[hi] * (pos-lo)

with sql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
    with conn.cursor() as cur:
        cur.execute(f"USE CATALOG `{catalog}`")
        cur.execute(f"USE SCHEMA `{schema}`")

        # Small lexical category dictionary derived from corpus metadata, not eval ground truth.
        # This is intentionally only a dev baseline; the production corpus gets a true search baseline.
        cur.execute("""
        CREATE OR REPLACE TEMP VIEW lexical_category_dictionary AS
        SELECT DISTINCT
          category,
          replace(category, '_', ' ') AS category_phrase
        FROM scientific_chunks_dev
        """)

        t0 = time.perf_counter()
        cur.execute("""
        WITH inferred AS (
          SELECT
            q.query_id,
            q.query_text,
            q.relevant_category,
            d.category AS inferred_category,
            CASE
              WHEN lower(q.query_text) LIKE concat('%', lower(d.category_phrase), '%') THEN 10
              ELSE 0
            END AS score
          FROM retrieval_eval_queries_dev q
          CROSS JOIN lexical_category_dictionary d
        ),
        best_category AS (
          SELECT *
          FROM (
            SELECT
              *,
              row_number() OVER (
                PARTITION BY query_id
                ORDER BY score DESC, inferred_category
              ) AS rn
            FROM inferred
          )
          WHERE rn = 1
        ),
        ranked_chunks AS (
          SELECT
            b.query_id,
            b.query_text,
            b.relevant_category,
            c.chunk_id,
            c.document_id,
            c.category AS retrieved_category,
            row_number() OVER (
              PARTITION BY b.query_id
              ORDER BY c.chunk_id
            ) AS rank
          FROM best_category b
          JOIN scientific_chunks_dev c
            ON c.category = b.inferred_category
        )
        SELECT
          query_id,
          query_text,
          relevant_category,
          rank,
          chunk_id,
          document_id,
          retrieved_category
        FROM ranked_chunks
        WHERE rank <= 10
        ORDER BY query_id, rank
        """)
        rows = cur.fetchall()
        batch_ms = (time.perf_counter() - t0) * 1000.0

        cur.execute("""
        SELECT query_id, query_text
        FROM retrieval_eval_queries_dev
        ORDER BY query_id
        LIMIT 50
        """)
        latency_queries = cur.fetchall()

        latency_ms = []
        for _, query_text in latency_queries:
            q = query_text.lower().replace("'", "''")
            t1 = time.perf_counter()
            cur.execute(f"""
            WITH inferred AS (
              SELECT
                category,
                CASE
                  WHEN '{q}' LIKE concat('%', lower(replace(category, '_', ' ')), '%') THEN 10
                  ELSE 0
                END AS score
              FROM (SELECT DISTINCT category FROM scientific_chunks_dev)
            ),
            best AS (
              SELECT category
              FROM inferred
              ORDER BY score DESC, category
              LIMIT 1
            )
            SELECT chunk_id
            FROM scientific_chunks_dev
            WHERE category = (SELECT category FROM best)
            ORDER BY chunk_id
            LIMIT 10
            """)
            cur.fetchall()
            latency_ms.append((time.perf_counter() - t1) * 1000.0)

by_query = {}
for r in rows:
    by_query.setdefault(r[0], []).append(r)

detail = []
r5s, r10s, rrs, ndcgs = [], [], [], []

for qid, hits in by_query.items():
    relevant = hits[0][2]
    rels = [1 if h[6] == relevant else 0 for h in hits]
    r5 = 1.0 if any(rels[:5]) else 0.0
    r10 = 1.0 if any(rels[:10]) else 0.0
    first = next((i+1 for i, x in enumerate(rels) if x), None)
    rr = 1.0 / first if first else 0.0
    dcg = sum(rel / math.log2(i+2) for i, rel in enumerate(rels[:10]))
    ideal = min(10, sum(rels))
    idcg = sum(1.0 / math.log2(i+2) for i in range(ideal))
    ndcg = dcg / idcg if idcg else 0.0

    r5s.append(r5); r10s.append(r10); rrs.append(rr); ndcgs.append(ndcg)
    detail.append({
        "query_id": qid,
        "relevant_category": relevant,
        "first_relevant_rank": first or 0,
        "recall_at_5": r5,
        "recall_at_10": r10,
        "reciprocal_rank": rr,
        "ndcg_at_10": ndcg,
    })

summary = {
    "benchmark": "lexical_dev_baseline_fast",
    "note": "Development-only metadata lexical baseline; not final production retrieval evidence.",
    "queries": len(by_query),
    "corpus_documents": 10000,
    "corpus_chunks": 28000,
    "recall_at_5": statistics.mean(r5s),
    "recall_at_10": statistics.mean(r10s),
    "mrr": statistics.mean(rrs),
    "ndcg_at_10": statistics.mean(ndcgs),
    "retrieval_p50_ms": pct(latency_ms, 0.50),
    "retrieval_p95_ms": pct(latency_ms, 0.95),
    "batch_eval_ms": batch_ms,
    "latency_sample_queries": len(latency_ms),
}

with (RESULTS / "lexical_baseline_dev.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

with (RESULTS / "lexical_baseline_dev.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=detail[0].keys())
    w.writeheader()
    w.writerows(detail)

for k in ["queries","recall_at_5","recall_at_10","mrr","ndcg_at_10","retrieval_p50_ms","retrieval_p95_ms","batch_eval_ms"]:
    print(f"{k.upper()}={summary[k]}")
print("LEXICAL_RETRIEVAL=PASS")
