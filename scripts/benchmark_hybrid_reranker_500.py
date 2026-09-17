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
token = os.environ["DATABRICKS_TOKEN"]
catalog = os.environ.get("DATABRICKS_CATALOG", "p14_rag_llmops")
schema = os.environ.get("DATABRICKS_SCHEMA", "default")
index_name = f"{catalog}.{schema}.scientific_chunks_dev_ai_index"

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def pctl(values, q):
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi-pos) + xs[hi] * (pos-lo)

def dcg(labels):
    return sum((1.0 / math.log2(i + 2)) for i, rel in enumerate(labels) if rel)

IDCG10 = dcg([True] * 10)

status_url = f"{host}/api/2.0/vector-search/indexes/{quote(index_name, safe='')}"
status_resp = requests.get(status_url, headers=headers, timeout=60)
status_resp.raise_for_status()
status = status_resp.json().get("status", {})
print(f"INDEX={index_name}")
print(f"READY={status.get('ready')}")
print(f"INDEXED_ROW_COUNT={status.get('indexed_row_count')}")
if not status.get("ready"):
    raise SystemExit("BENCHMARK_ABORTED: index is not ready")

with sql.connect(server_hostname=server_hostname, http_path=os.environ["DATABRICKS_HTTP_PATH"], access_token=token) as conn:
    with conn.cursor() as cur:
        cur.execute(f"USE CATALOG `{catalog}`")
        cur.execute(f"USE SCHEMA `{schema}`")
        cur.execute("""
        SELECT query_id, relevant_category, query_text
        FROM retrieval_eval_queries_dev
        ORDER BY query_id
        LIMIT 500
        """)
        rows = cur.fetchall()

query_url = f"{host}/api/2.0/vector-search/indexes/{quote(index_name, safe='')}/query"

latencies = []
hit5 = hit10 = 0
precision5 = precision10 = 0.0
rr = ndcg10 = 0.0
failures = 0
details = []

for i, (qid, truth, query_text) in enumerate(rows, start=1):
    body = {
        "columns": ["chunk_id", "document_id", "category", "chunk_text"],
        "num_results": 10,
        "query_text": query_text,
        "query_type": "HYBRID",
        "reranker": {
            "model": "databricks_reranker",
            "parameters": {
                "columns_to_rerank": ["chunk_text", "category"]
            }
        },
        "debug_level": 1
    }

    started = time.perf_counter()
    resp = requests.post(query_url, headers=headers, json=body, timeout=180)
    elapsed = (time.perf_counter() - started) * 1000.0
    latencies.append(elapsed)

    if resp.status_code != 200:
        failures += 1
        print(f"FAIL query={qid} status={resp.status_code} body={resp.text[:500]}")
        continue

    payload = resp.json()
    cols = [x.get("name") for x in payload.get("manifest", {}).get("columns", [])]
    data = payload.get("result", {}).get("data_array", [])
    recs = [dict(zip(cols, row)) for row in data]
    labels = [str(r.get("category")) == str(truth) for r in recs[:10]]

    hit5 += int(any(labels[:5]))
    hit10 += int(any(labels[:10]))
    precision5 += sum(labels[:5]) / 5.0
    precision10 += sum(labels[:10]) / 10.0
    first = next((rank for rank, rel in enumerate(labels, start=1) if rel), None)
    rr += (1.0 / first) if first else 0.0
    ndcg10 += dcg(labels[:10]) / IDCG10

    details.append({
        "query_id": qid,
        "ground_truth_category": truth,
        "query_text": query_text,
        "latency_ms": elapsed,
        "top10_categories": [r.get("category") for r in recs[:10]],
    })

    if i % 50 == 0:
        print(f"PROGRESS={i}/500")

n = len(rows)
metrics = {
    "queries": n,
    "successful_queries": n - failures,
    "failures": failures,
    "hit_at_5": hit5 / n,
    "hit_at_10": hit10 / n,
    "precision_at_5": precision5 / n,
    "precision_at_10": precision10 / n,
    "mrr": rr / n,
    "ndcg_at_10": ndcg10 / n,
    "latency_p50_ms": pctl(latencies, .50),
    "latency_p95_ms": pctl(latencies, .95),
    "latency_mean_ms": statistics.mean(latencies),
}

for k, v in metrics.items():
    print(f"RERANK_{k.upper()}={v}")

out = ROOT / "results" / "ai_search_500_dev_hybrid_reranker.json"
out.write_text(json.dumps({
    "index": index_name,
    "note": "Synthetic development-plumbing benchmark. Reranker uses chunk_text and category.",
    "metrics": metrics,
    "details": details,
}, indent=2, default=str))

os.environ["DATABRICKS_HOST"] = host
os.environ["DATABRICKS_TOKEN"] = token
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/project14-large-scale-rag")

with mlflow.start_run(run_name="dev_ai_search_hybrid_reranker_500") as run:
    mlflow.log_params({
        "index": index_name,
        "query_type": "HYBRID",
        "reranker": "databricks_reranker",
        "columns_to_rerank": "chunk_text,category",
        "evaluation_queries": 500,
        "dataset_type": "synthetic_dev",
    })
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            mlflow.log_metric(k, float(v))
    mlflow.log_artifact(str(out), artifact_path="retrieval_benchmarks")
    mlflow.set_tags({
        "project": "project14",
        "phase": "dev_ai_search_reranker",
        "evidence_type": "measured_dev_plumbing",
    })
    print(f"MLFLOW_RUN_ID={run.info.run_id}")

print(f"RESULT_JSON={out}")
print("HYBRID_RERANKER_500=PASS")
