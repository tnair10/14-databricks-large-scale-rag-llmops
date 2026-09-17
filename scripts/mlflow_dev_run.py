from pathlib import Path
import json, mlflow
ROOT = Path(__file__).resolve().parents[1]
result_file = ROOT / "results" / "lexical_baseline_dev.json"
if not result_file.exists():
    raise SystemExit(f"Missing {result_file}")
metrics = json.loads(result_file.read_text())
mlflow.set_experiment("/Shared/project14-large-scale-rag")
with mlflow.start_run(run_name="dev_lexical_plumbing_validation"):
    mlflow.log_params({
        "dataset_type": "synthetic-development",
        "documents": metrics["corpus_documents"],
        "chunks": metrics["corpus_chunks"],
        "retrieval_variant": "metadata_lexical_dev",
        "evidence_scope": "plumbing_validation_only",
    })
    for key in ["recall_at_5","recall_at_10","mrr","ndcg_at_10","retrieval_p50_ms","retrieval_p95_ms","batch_eval_ms"]:
        mlflow.log_metric(key, float(metrics[key]))
    mlflow.log_artifact(str(result_file))
    print(f"MLFLOW_RUN_ID={mlflow.active_run().info.run_id}")
    print("MLFLOW_DEV_LOG=PASS")
