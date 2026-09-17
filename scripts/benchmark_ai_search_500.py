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

host_raw = os.environ["DATABRICKS_SERVER_HOSTNAME"].rstrip("/")
host = host_raw if host_raw.startswith("http") else "https://" + host_raw
server_hostname = host.replace("https://", "").replace("http://", "")
http_path = os.environ["DATABRICKS_HTTP_PATH"]
token = os.environ["DATABRICKS_TOKEN"]

catalog = os.environ.get("DATABRICKS_CATALOG", "p14_rag_llmops")
schema = os.environ.get("DATABRICKS_SCHEMA", "default")
index_name = f"{catalog}.{schema}.scientific_chunks_dev_ai_index"

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

def pctl(values, q):
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)

def dcg_binary(ranks):
    total = 0.0
    for i, rel in enumerate(ranks, start=1):
        if rel:
            total += 1.0 / math.log2(i + 1)
    return total

# 1) Refuse to benchmark a partial index.
status_url = f"{host}/api/2.0/vector-search/indexes/{quote(index_name, safe='')}"
r = requests.get(status_url, headers=headers, timeout=60)
r.raise_for_status()
status_payload = r.json()
status = status_payload.get("status", {})
ready = bool(status.get("ready"))
detailed_state = status.get("detailed_state")
indexed_row_count = status.get("indexed_row_count")

print(f"INDEX={index_name}")
print(f"DETAILED_STATE={detailed_state}")
print(f"READY={ready}")
print(f"INDEXED_ROW_COUNT={indexed_row_count}")

if not ready:
    raise SystemExit("BENCHMARK_ABORTED: index is not ready; wait for the initial sync to complete.")

# 2) Read the 500 dev evaluation queries.
with sql.connect(
    server_hostname=server_hostname,
    http_path=http_path,
    access_token=token,
) as conn:
    with conn.cursor() as cur:
        cur.execute(f"USE CATALOG `{catalog}`")
        cur.execute(f"USE SCHEMA `{schema}`")
        cur.execute("SELECT * FROM retrieval_eval_queries_dev ORDER BY query_id LIMIT 500")
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]

print("EVAL_COLUMNS=" + json.dumps(cols))
print(f"EVAL_QUERY_ROWS={len(rows)}")

query_col = next((c for c in ("query_text", "query", "text") if c in cols), None)
id_col = next((c for c in ("query_id", "id") if c in cols), None)
gt_col = next(
    (c for c in (
        "ground_truth_category",
        "expected_category",
        "relevant_category",
        "category",
        "ground_truth"
    ) if c in cols),
    None,
)

if not query_col or not gt_col:
    raise SystemExit(
        f"Could not identify query/ground-truth columns. Available columns: {cols}"
    )

query_url = f"{host}/api/2.0/vector-search/indexes/{quote(index_name, safe='')}/query"

all_results = {}

for query_type in ("ANN", "HYBRID"):
    latencies = []
    recall5_hits = 0
    recall10_hits = 0
    rr_sum = 0.0
    ndcg10_sum = 0.0
    failures = 0
    details = []

    print(f"=== {query_type} 500-QUERY BENCHMARK ===")

    for idx, row in enumerate(rows, start=1):
        qid = row.get(id_col) if id_col else idx
        query_text = str(row[query_col])
        truth = str(row[gt_col])

        body = {
            "columns": ["chunk_id", "document_id", "category", "chunk_text"],
            "num_results": 10,
            "query_text": query_text,
            "query_type": query_type,
        }

        started = time.perf_counter()
        resp = requests.post(query_url, headers=headers, json=body, timeout=120)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed_ms)

        if resp.status_code != 200:
            failures += 1
            print(f"FAIL query={qid} status={resp.status_code} body={resp.text[:300]}")
            continue

        payload = resp.json()
        result_cols = [x.get("name") for x in payload.get("manifest", {}).get("columns", [])]
        data = payload.get("result", {}).get("data_array", [])
        records = [dict(zip(result_cols, x)) for x in data]

        rel = [str(rec.get("category")) == truth for rec in records[:10]]

        first_rel_rank = next((i for i, is_rel in enumerate(rel, start=1) if is_rel), None)
        hit5 = any(rel[:5])
        hit10 = any(rel[:10])

        recall5_hits += int(hit5)
        recall10_hits += int(hit10)
        rr_sum += (1.0 / first_rel_rank) if first_rel_rank else 0.0

        # Binary NDCG@10 with one relevant category target: ideal DCG is 1.0.
        ndcg10_sum += dcg_binary(rel[:10])

        details.append({
            "query_id": qid,
            "query_text": query_text,
            "ground_truth_category": truth,
            "latency_ms": elapsed_ms,
            "first_relevant_rank": first_rel_rank,
            "hit_at_5": hit5,
            "hit_at_10": hit10,
            "top10_categories": [rec.get("category") for rec in records[:10]],
        })

        if idx % 50 == 0:
            print(f"PROGRESS={idx}/{len(rows)}")

    n = len(rows)
    success_n = n - failures

    metrics = {
        "queries": n,
        "successful_queries": success_n,
        "failures": failures,
        "recall_at_5": recall5_hits / n,
        "recall_at_10": recall10_hits / n,
        "mrr": rr_sum / n,
        "ndcg_at_10": ndcg10_sum / n,
        "latency_p50_ms": pctl(latencies, 0.50),
        "latency_p95_ms": pctl(latencies, 0.95),
        "latency_mean_ms": statistics.mean(latencies) if latencies else float("nan"),
    }
    all_results[query_type] = {"metrics": metrics, "details": details}

    for k, v in metrics.items():
        print(f"{query_type}_{k.upper()}={v}")

results_dir = ROOT / "results"
results_dir.mkdir(exist_ok=True)

out_json = results_dir / "ai_search_500_dev_benchmark.json"
out_json.write_text(json.dumps({
    "index": index_name,
    "index_state": detailed_state,
    "indexed_row_count": indexed_row_count,
    "note": "Synthetic development plumbing evaluation. Do not treat as production retrieval-quality evidence.",
    "results": all_results,
}, indent=2, default=str))

# 3) Log the measured benchmark to Databricks MLflow.
os.environ["DATABRICKS_HOST"] = host
os.environ["DATABRICKS_TOKEN"] = token
mlflow.set_tracking_uri("databricks")
experiment = "/Shared/project14-large-scale-rag"
mlflow.set_experiment(experiment)

with mlflow.start_run(run_name="dev_ai_search_ann_vs_hybrid_500") as run:
    mlflow.log_params({
        "index": index_name,
        "evaluation_queries": len(rows),
        "dataset_type": "synthetic_dev",
        "evaluation_note": "plumbing_validation_not_production_quality_evidence",
    })

    for query_type, result in all_results.items():
        prefix = query_type.lower()
        for name, value in result["metrics"].items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                mlflow.log_metric(f"{prefix}_{name}", float(value))

    mlflow.log_artifact(str(out_json), artifact_path="retrieval_benchmarks")
    mlflow.set_tags({
        "project": "project14",
        "phase": "dev_ai_search",
        "evidence_type": "measured_dev_plumbing",
    })
    print(f"MLFLOW_RUN_ID={run.info.run_id}")

print(f"RESULT_JSON={out_json}")
print("AI_SEARCH_500_BENCHMARK=PASS")
