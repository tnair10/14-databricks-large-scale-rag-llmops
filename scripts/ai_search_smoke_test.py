from __future__ import annotations

import os
import time
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
url = f"{host}/api/2.0/vector-search/indexes/{quote(index_name, safe='')}/query"
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

queries = [
    "scalable experiments for distributed systems",
    "quality and latency analysis in databases",
    "reproducible evaluation of robotics approaches",
]

for query_type in ("ANN", "HYBRID"):
    print(f"=== {query_type} ===")
    for q in queries:
        body = {
            "columns": ["chunk_id", "document_id", "category", "chunk_text"],
            "num_results": 5,
            "query_text": q,
            "query_type": query_type,
        }
        started = time.perf_counter()
        r = requests.post(url, headers=headers, json=body, timeout=120)
        elapsed_ms = (time.perf_counter() - started) * 1000

        if r.status_code != 200:
            raise SystemExit(
                f"QUERY_FAILED type={query_type} status={r.status_code} body={r.text[:1800]}"
            )

        payload = r.json()
        columns = [x.get("name") for x in payload.get("manifest", {}).get("columns", [])]
        rows = payload.get("result", {}).get("data_array", [])

        print(f"QUERY={q}")
        print(f"LATENCY_MS={elapsed_ms:.2f}")
        print(f"RESULTS={len(rows)}")
        print(f"COLUMNS={columns}")
        for i, row in enumerate(rows[:3], start=1):
            record = dict(zip(columns, row))
            text = str(record.get("chunk_text", "")).replace("\n", " ")
            print(
                f"TOP{i} chunk_id={record.get('chunk_id')} "
                f"document_id={record.get('document_id')} "
                f"category={record.get('category')} "
                f"text={text[:100]}"
            )

print("AI_SEARCH_SMOKE=PASS")
