from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

host = os.environ["DATABRICKS_SERVER_HOSTNAME"].rstrip("/")
if not host.startswith("http"):
    host = "https://" + host

os.environ["DATABRICKS_HOST"] = host
os.environ["DATABRICKS_TOKEN"] = os.environ["DATABRICKS_TOKEN"]

mlflow.set_tracking_uri("databricks")

result_file = ROOT / "results" / "lexical_baseline_dev.json"
csv_file = ROOT / "results" / "lexical_baseline_dev.csv"

if not result_file.exists():
    raise SystemExit(f"Missing {result_file}")

metrics = json.loads(result_file.read_text())

experiment_name = "/Shared/project14-large-scale-rag"
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="dev_lexical_plumbing_validation") as run:
    mlflow.log_params({
        "dataset_type": "synthetic-development",
        "documents": metrics["corpus_documents"],
        "chunks": metrics["corpus_chunks"],
        "retrieval_variant": "metadata_lexical_dev",
        "evidence_scope": "plumbing_validation_only",
    })

    for key in [
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
        "retrieval_p50_ms",
        "retrieval_p95_ms",
        "batch_eval_ms",
    ]:
        mlflow.log_metric(key, float(metrics[key]))

    mlflow.log_artifact(str(result_file))
    if csv_file.exists():
        mlflow.log_artifact(str(csv_file))

    print(f"TRACKING_URI={mlflow.get_tracking_uri()}")
    print(f"EXPERIMENT={experiment_name}")
    print(f"MLFLOW_RUN_ID={run.info.run_id}")
    print("MLFLOW_DATABRICKS_LOG=PASS")
