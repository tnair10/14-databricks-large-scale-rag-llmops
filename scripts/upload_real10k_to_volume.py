from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import requests
from databricks import sql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

host = os.environ["DATABRICKS_SERVER_HOSTNAME"].rstrip("/")
if not host.startswith("http"):
    host = "https://" + host

token = os.environ["DATABRICKS_TOKEN"]
http_path = os.environ["DATABRICKS_HTTP_PATH"]
catalog = os.environ.get("DATABRICKS_CATALOG", "p14_rag_llmops")
schema = os.environ.get("DATABRICKS_SCHEMA", "default")
volume = "p14_stage"

local_file = ROOT / "data" / "arxiv_real_10k.jsonl"
if not local_file.exists():
    raise SystemExit(f"Missing {local_file}")

with sql.connect(
    server_hostname=host.replace("https://", ""),
    http_path=http_path,
    access_token=token,
) as conn:
    with conn.cursor() as cur:
        cur.execute(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.`{volume}`")

volume_path = f"/Volumes/{catalog}/{schema}/{volume}/arxiv_real_10k.jsonl"
api_url = f"{host}/api/2.0/fs/files{quote(volume_path, safe='/')}"

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/octet-stream",
}

with local_file.open("rb") as fh:
    r = requests.put(
        api_url,
        headers=headers,
        params={"overwrite": "true"},
        data=fh,
        timeout=300,
    )

if r.status_code not in (200, 201, 204):
    raise SystemExit(f"UPLOAD_FAILED status={r.status_code} body={r.text[:1000]}")

print(f"LOCAL_FILE={local_file}")
print(f"LOCAL_BYTES={local_file.stat().st_size}")
print(f"VOLUME_PATH={volume_path}")
print("VOLUME_UPLOAD=PASS")
