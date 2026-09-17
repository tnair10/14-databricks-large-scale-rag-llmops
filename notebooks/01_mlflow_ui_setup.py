# Databricks notebook source
# MAGIC %pip install --upgrade "mlflow[databricks]>=3.1.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import mlflow
EXPERIMENT = "/Shared/project14-large-scale-rag"
mlflow.set_experiment(EXPERIMENT)
print(f"EXPERIMENT={EXPERIMENT}")

# COMMAND ----------
with mlflow.start_run(run_name="ui_smoke_test") as run:
    mlflow.log_param("project", "Project 14")
    mlflow.log_metric("smoke_test", 1.0)
    print(f"RUN_ID={run.info.run_id}")
