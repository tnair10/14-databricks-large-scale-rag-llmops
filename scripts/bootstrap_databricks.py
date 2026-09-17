from __future__ import annotations

import os
from pathlib import Path
from databricks import sql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

host = os.environ["DATABRICKS_SERVER_HOSTNAME"].replace("https://", "").rstrip("/")
http_path = os.environ["DATABRICKS_HTTP_PATH"]
token = os.environ["DATABRICKS_TOKEN"]
catalog = os.environ.get("DATABRICKS_CATALOG", "p14_rag_llmops")
schema = os.environ.get("DATABRICKS_SCHEMA", "default")

with sql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT current_user(), current_catalog(), current_schema()")
        who = cur.fetchone()

        cur.execute(f"CREATE CATALOG IF NOT EXISTS `{catalog}`")
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
        cur.execute(f"USE CATALOG `{catalog}`")
        cur.execute(f"USE SCHEMA `{schema}`")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS project14_bootstrap (
            check_name STRING,
            check_value STRING,
            checked_at TIMESTAMP
        ) USING DELTA
        """)
        cur.execute("""
        INSERT INTO project14_bootstrap
        VALUES ('bootstrap', 'PASS', current_timestamp())
        """)
        cur.execute("SELECT check_name, check_value, checked_at FROM project14_bootstrap ORDER BY checked_at DESC LIMIT 1")
        check = cur.fetchone()

print("DATABRICKS_CONNECTION=PASS")
print(f"CURRENT_USER={who[0]}")
print(f"TARGET_CATALOG={catalog}")
print(f"TARGET_SCHEMA={schema}")
print(f"DELTA_WRITE={check[1]}")
