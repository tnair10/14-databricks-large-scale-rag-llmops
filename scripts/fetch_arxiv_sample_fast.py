from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

API = "https://datasets-server.huggingface.co/rows"
DATASET = "common-pile/arxiv_abstracts"
CONFIG = "default"
SPLIT = "train"

def normalize(raw: dict, idx: int) -> dict:
    meta = raw.get("meta") or raw.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}

    doc_id = raw.get("id") or raw.get("paper_id") or meta.get("id") or f"ARXIV-{idx:09d}"
    text = raw.get("text") or raw.get("abstract") or raw.get("content") or ""
    title = raw.get("title") or meta.get("title") or f"ArXiv document {doc_id}"
    categories = raw.get("categories") or meta.get("categories") or meta.get("category") or []
    if isinstance(categories, str):
        categories = [x for x in categories.replace(",", " ").split() if x]

    return {
        "document_id": str(doc_id),
        "title": str(title),
        "abstract": str(text),
        "categories": categories or [],
        "published_date": str(meta.get("published") or meta.get("date") or "") or None,
        "source": DATASET,
        "source_url": meta.get("url"),
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--out", default="data/arxiv_sample.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    batch_size = 100
    written = 0
    started = time.perf_counter()

    with out.open("w", encoding="utf-8") as fh:
        while written < args.rows:
            length = min(batch_size, args.rows - written)
            params = {
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "offset": written,
                "length": length,
            }
            r = requests.get(API, params=params, headers=headers, timeout=60)
            r.raise_for_status()
            payload = r.json()
            rows = payload.get("rows", [])
            if not rows:
                raise RuntimeError(f"No rows returned at offset {written}: {payload}")

            for item in rows:
                raw = item.get("row", item)
                rec = normalize(raw, written)
                if rec["abstract"]:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= args.rows:
                        break

            print(f"FETCHED={written}/{args.rows}", flush=True)

    elapsed = time.perf_counter() - started
    print(f"DATASET={DATASET}")
    print(f"WRITTEN_ROWS={written}")
    print(f"OUTPUT={out}")
    print(f"BYTES={out.stat().st_size}")
    print(f"ELAPSED_SECONDS={elapsed:.2f}")
    print("ARXIV_SAMPLE_FAST=PASS")

if __name__ == "__main__":
    main()
