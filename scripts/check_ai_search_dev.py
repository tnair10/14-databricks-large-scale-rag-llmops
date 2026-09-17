from __future__ import annotations

import json
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
index_name = "p14_rag_llmops.default.scientific_chunks_dev_ai_index"

url = f"{host}/api/2.0/vector-search/indexes/{quote(index_name, safe='')}"
r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=60)

if r.status_code != 200:
    raise SystemExit(f"AI_SEARCH_STATUS_FAILED status={r.status_code} body={r.text[:1500]}")

payload = r.json()
status = payload.get("status", {})
print(f"INDEX={index_name}")
print(f"DETAILED_STATE={status.get('detailed_state')}")
print(f"READY={status.get('ready')}")
print(f"INDEX_URL={status.get('index_url')}")
print("AI_SEARCH_STATUS=PASS")
print("RAW_STATUS=" + json.dumps(status, sort_keys=True))
