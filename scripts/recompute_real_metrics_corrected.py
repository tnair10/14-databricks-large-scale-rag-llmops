from __future__ import annotations

import json
import math
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

src = ROOT / "results" / "ai_search_500_real_benchmark.json"
if not src.exists():
    raise SystemExit(f"Missing {src}")

payload = json.loads(src.read_text())

def dcg(labels):
    return sum((1.0 / math.log2(i + 2)) for i, rel in enumerate(labels) if rel)

corrected = {}

for variant, block in payload["results"].items():
    details = block["details"]
    n = len(details)
    hit5 = hit10 = 0
    rr = ndcg10 = 0.0

    for row in details:
        relevant = set(str(x) for x in row.get("relevant_document_ids", []))
        returned = [str(x) for x in row.get("top10_document_ids", [])]

        labels = [doc_id in relevant for doc_id in returned[:10]]
        hit5 += int(any(labels[:5]))
        hit10 += int(any(labels[:10]))

        first = next((i for i, rel in enumerate(labels, start=1) if rel), None)
        rr += (1.0 / first) if first else 0.0

        # One known relevant document per query. Multiple returned chunks from the
        # same relevant document do not create multiple relevance gains.
        first_relevant_rank = first
        ndcg10 += (1.0 / math.log2(first_relevant_rank + 1)) if first_relevant_rank else 0.0

    corrected[variant] = {
        "queries": n,
        "hit_at_5": hit5 / n,
        "hit_at_10": hit10 / n,
        "mrr": rr / n,
        "ndcg_at_10": ndcg10 / n,
        "latency_p50_ms": block["metrics"]["latency_p50_ms"],
        "latency_p95_ms": block["metrics"]["latency_p95_ms"],
        "latency_mean_ms": block["metrics"]["latency_mean_ms"],
        "failures": block["metrics"]["failures"],
    }

out = ROOT / "results" / "ai_search_500_real_benchmark_corrected.json"
out.write_text(json.dumps({
    "source": str(src),
    "note": (
        "Corrected real-corpus benchmark. NDCG@10 uses the first rank of the one known "
        "relevant document per query, preventing duplicate relevant chunks from producing gain > 1."
    ),
    "results": corrected,
}, indent=2))

for variant, metrics in corrected.items():
    print(f"=== {variant} CORRECTED ===")
    for k, v in metrics.items():
        print(f"{variant}_{k.upper()}={v}")

host = os.environ["DATABRICKS_SERVER_HOSTNAME"].rstrip("/")
if not host.startswith("http"):
    host = "https://" + host
os.environ["DATABRICKS_HOST"] = host
os.environ["DATABRICKS_TOKEN"] = os.environ["DATABRICKS_TOKEN"]

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/project14-large-scale-rag")

with mlflow.start_run(run_name="real_ai_search_ann_vs_hybrid_500_corrected") as run:
    mlflow.log_params({
        "source_run": "real_ai_search_ann_vs_hybrid_500",
        "evaluation_queries": 500,
        "dataset_type": "real_arxiv_10k",
        "metric_correction": "dedupe_relevant_document_for_ndcg",
    })
    for variant, metrics in corrected.items():
        prefix = variant.lower()
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                mlflow.log_metric(f"{prefix}_{k}", float(v))
    mlflow.log_artifact(str(out), artifact_path="retrieval_benchmarks")
    mlflow.set_tags({
        "project": "project14",
        "phase": "real10k_ai_search",
        "evidence_type": "corrected_measured_real_corpus",
    })
    print(f"MLFLOW_RUN_ID={run.info.run_id}")

print(f"RESULT_JSON={out}")
print("REAL_METRICS_CORRECTED=PASS")
