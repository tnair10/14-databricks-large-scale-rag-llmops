from __future__ import annotations

import json
import math
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

src = ROOT / "results" / "ai_search_500_dev_benchmark.json"
if not src.exists():
    raise SystemExit(f"Missing {src}")

payload = json.loads(src.read_text())

def dcg_binary(labels):
    return sum((1.0 / math.log2(i + 2)) for i, rel in enumerate(labels) if rel)

IDCG10 = dcg_binary([True] * 10)

corrected = {}
for variant, block in payload["results"].items():
    details = block["details"]
    n = len(details)

    hit5 = 0
    hit10 = 0
    rr = 0.0
    ndcg = 0.0
    precision5 = 0.0
    precision10 = 0.0

    for row in details:
        truth = str(row["ground_truth_category"])
        cats = [str(x) for x in row.get("top10_categories", [])]
        labels = [c == truth for c in cats[:10]]

        hit5 += int(any(labels[:5]))
        hit10 += int(any(labels[:10]))
        precision5 += sum(labels[:5]) / 5.0
        precision10 += sum(labels[:10]) / 10.0

        first = next((i for i, rel in enumerate(labels, start=1) if rel), None)
        rr += (1.0 / first) if first else 0.0
        ndcg += dcg_binary(labels) / IDCG10

    corrected[variant] = {
        "queries": n,
        "hit_at_5": hit5 / n,
        "hit_at_10": hit10 / n,
        "precision_at_5": precision5 / n,
        "precision_at_10": precision10 / n,
        "mrr": rr / n,
        "ndcg_at_10": ndcg / n,
        "latency_p50_ms": block["metrics"]["latency_p50_ms"],
        "latency_p95_ms": block["metrics"]["latency_p95_ms"],
        "latency_mean_ms": block["metrics"]["latency_mean_ms"],
        "failures": block["metrics"]["failures"],
    }

out = ROOT / "results" / "ai_search_500_dev_benchmark_corrected.json"
out.write_text(json.dumps({
    "source": str(src),
    "note": (
        "Corrected development-plumbing metrics. Previous recall@K was actually hit@K. "
        "NDCG@10 is normalized by ideal DCG with ten relevant binary results and is bounded [0,1]."
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

with mlflow.start_run(run_name="dev_ai_search_ann_vs_hybrid_500_corrected") as run:
    mlflow.log_params({
        "source_run": "dev_ai_search_ann_vs_hybrid_500",
        "evaluation_queries": 500,
        "dataset_type": "synthetic_dev",
        "metric_correction": "recall_to_hit_rate_and_normalized_ndcg",
    })
    for variant, metrics in corrected.items():
        prefix = variant.lower()
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                mlflow.log_metric(f"{prefix}_{k}", float(v))
    mlflow.log_artifact(str(out), artifact_path="retrieval_benchmarks")
    mlflow.set_tags({
        "project": "project14",
        "phase": "dev_ai_search",
        "evidence_type": "corrected_measured_dev_plumbing",
    })
    print(f"MLFLOW_RUN_ID={run.info.run_id}")

print(f"RESULT_JSON={out}")
print("CORRECTED_DEV_METRICS=PASS")
