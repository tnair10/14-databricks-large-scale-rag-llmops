from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

host = os.environ["DATABRICKS_SERVER_HOSTNAME"].rstrip("/")
if not host.startswith("http"):
    host = "https://" + host
token = os.environ["DATABRICKS_TOKEN"]
catalog = os.environ.get("DATABRICKS_CATALOG", "p14_rag_llmops")
schema = os.environ.get("DATABRICKS_SCHEMA", "default")

local_file = ROOT / "data" / "real_retrieval_eval_500.jsonl"
volume_path = f"/Volumes/{catalog}/{schema}/p14_stage/real_retrieval_eval_500.jsonl"
url = f"{host}/api/2.0/fs/files{quote(volume_path, safe='/')}"

with local_file.open("rb") as fh:
    r = requests.put(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"},
        params={"overwrite": "true"},
        data=fh,
        timeout=300,
    )

if r.status_code not in (200, 201, 204):
    raise SystemExit(f"UPLOAD_FAILED status={r.status_code} body={r.text[:1000]}")

print(f"LOCAL_FILE={local_file}")
print(f"VOLUME_PATH={volume_path}")
print("REAL_EVAL_UPLOAD=PASS")
