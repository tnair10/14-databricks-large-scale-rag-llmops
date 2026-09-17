from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path
from urllib.parse import quote

import mlflow
import requests
from databricks import sql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

host = os.environ["DATABRICKS_SERVER_HOSTNAME"].rstrip("/")
if not host.startswith("http"):
    host = "https://" + host
server_hostname = host.replace("https://", "").replace("http://", "")
http_path = os.environ["DATABRICKS_HTTP_PATH"]
token = os.environ["DATABRICKS_TOKEN"]

catalog = os.environ.get("DATABRICKS_CATALOG", "p14_rag_llmops")
schema = os.environ.get("DATABRICKS_SCHEMA", "default")
index_name = f"{catalog}.{schema}.scientific_chunks_prod_ai_index"

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

def pctl(values, q):
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)

def dcg(labels):
    return sum((1.0 / math.log2(i + 2)) for i, rel in enumerate(labels) if rel)

# One relevant document per query -> ideal DCG is 1 at rank 1.
IDCG = 1.0

status_url = f"{host}/api/2.0/vector-search/indexes/{quote(index_name, safe='')}"
r = requests.get(status_url, headers=headers, timeout=60)
r.raise_for_status()
status = r.json().get("status", {})

print(f"INDEX={index_name}")
print(f"DETAILED_STATE={status.get('detailed_state')}")
print(f"READY={status.get('ready')}")
print(f"INDEXED_ROW_COUNT={status.get('indexed_row_count')}")

if not status.get("ready"):
    raise SystemExit("BENCHMARK_ABORTED: real index is not ready")

with sql.connect(
    server_hostname=server_hostname,
    http_path=http_path,
    access_token=token,
) as conn:
    with conn.cursor() as cur:
        cur.execute(f"USE CATALOG `{catalog}`")
        cur.execute(f"USE SCHEMA `{schema}`")
        cur.execute("""
        SELECT query_id, query_text, relevant_document_ids
        FROM retrieval_eval_prod
        ORDER BY query_id
        LIMIT 500
        """)
        rows = cur.fetchall()

print(f"EVAL_QUERY_ROWS={len(rows)}")

query_url = f"{host}/api/2.0/vector-search/indexes/{quote(index_name, safe='')}/query"
all_results = {}

for query_type in ("ANN", "HYBRID"):
    latencies = []
    hit5 = hit10 = 0
    rr = ndcg10 = 0.0
    failures = 0
    details = []

    print(f"=== REAL {query_type} 500-QUERY BENCHMARK ===")

    for i, (qid, query_text, relevant_document_ids) in enumerate(rows, start=1):
        relevant = set(str(x) for x in (relevant_document_ids or []))

        body = {
            "columns": ["chunk_id", "document_id", "chunk_position", "chunk_text", "token_count", "source"],
            "num_results": 10,
            "query_text": query_text,
            "query_type": query_type,
        }

        started = time.perf_counter()
        resp = requests.post(query_url, headers=headers, json=body, timeout=120)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)

        if resp.status_code != 200:
            failures += 1
            print(f"FAIL query={qid} status={resp.status_code} body={resp.text[:400]}")
            continue

        payload = resp.json()
        cols = [x.get("name") for x in payload.get("manifest", {}).get("columns", [])]
        data = payload.get("result", {}).get("data_array", [])
        recs = [dict(zip(cols, row)) for row in data]

        labels = [str(rec.get("document_id")) in relevant for rec in recs[:10]]

        hit5 += int(any(labels[:5]))
        hit10 += int(any(labels[:10]))

        first_rank = next((rank for rank, rel in enumerate(labels, start=1) if rel), None)
        rr += (1.0 / first_rank) if first_rank else 0.0
        ndcg10 += dcg(labels[:10]) / IDCG

        details.append({
            "query_id": qid,
            "query_text": query_text,
            "relevant_document_ids": sorted(relevant),
            "latency_ms": latency_ms,
            "first_relevant_rank": first_rank,
            "hit_at_5": any(labels[:5]),
            "hit_at_10": any(labels[:10]),
            "top10_document_ids": [rec.get("document_id") for rec in recs[:10]],
        })

        if i % 50 == 0:
            print(f"PROGRESS={i}/{len(rows)}")

    n = len(rows)
    metrics = {
        "queries": n,
        "successful_queries": n - failures,
        "failures": failures,
        "hit_at_5": hit5 / n,
        "hit_at_10": hit10 / n,
        "mrr": rr / n,
        "ndcg_at_10": ndcg10 / n,
        "latency_p50_ms": pctl(latencies, 0.50),
        "latency_p95_ms": pctl(latencies, 0.95),
        "latency_mean_ms": statistics.mean(latencies),
    }

    all_results[query_type] = {"metrics": metrics, "details": details}

    for k, v in metrics.items():
        print(f"{query_type}_{k.upper()}={v}")

results_dir = ROOT / "results"
results_dir.mkdir(exist_ok=True)

out = results_dir / "ai_search_500_real_benchmark.json"
out.write_text(json.dumps({
    "index": index_name,
    "evaluation_table": f"{catalog}.{schema}.retrieval_eval_prod",
    "note": (
        "Real-corpus deterministic retrieval evaluation. Queries are TF-IDF-style keywords "
        "derived from held-out real arXiv abstracts with one known relevant document per query."
    ),
    "results": all_results,
}, indent=2, default=str))

os.environ["DATABRICKS_HOST"] = host
os.environ["DATABRICKS_TOKEN"] = token
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/project14-large-scale-rag")

with mlflow.start_run(run_name="real_ai_search_ann_vs_hybrid_500") as run:
    mlflow.log_params({
        "index": index_name,
        "evaluation_queries": len(rows),
        "dataset_type": "real_arxiv_10k",
        "evaluation_method": "deterministic_tfidf_keywords_known_relevant_document",
    })

    for variant, result in all_results.items():
        prefix = variant.lower()
        for name, value in result["metrics"].items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                mlflow.log_metric(f"{prefix}_{name}", float(value))

    mlflow.log_artifact(str(out), artifact_path="retrieval_benchmarks")
    mlflow.set_tags({
        "project": "project14",
        "phase": "real10k_ai_search",
        "evidence_type": "measured_real_corpus",
    })
    print(f"MLFLOW_RUN_ID={run.info.run_id}")

print(f"RESULT_JSON={out}")
print("REAL_AI_SEARCH_500_BENCHMARK=PASS")
