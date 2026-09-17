from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

API = "https://datasets-server.huggingface.co/rows"
DATASET = "common-pile/arxiv_abstracts"
CONFIG = "default"
SPLIT = "train"
RETRYABLE = {429, 500, 502, 503, 504}


def _meta(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def normalize(row: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    meta = _meta(row)
    document_id = row.get("id") or meta.get("id") or f"ARXIV-{fallback_index:09d}"
    abstract = row.get("text") or row.get("abstract") or ""
    authors = meta.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]

    return {
        "document_id": str(document_id),
        "abstract": str(abstract),
        "authors": authors,
        "created": row.get("created"),
        "source": row.get("source") or DATASET,
        "source_url": meta.get("url"),
        "license": meta.get("license"),
    }


def request_batch(
    session: requests.Session,
    headers: dict[str, str],
    offset: int,
    length: int,
    max_retries: int = 12,
) -> list[dict[str, Any]]:
    params = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "offset": offset,
        "length": length,
    }

    for attempt in range(max_retries + 1):
        try:
            r = session.get(API, params=params, headers=headers, timeout=60)

            if r.status_code in RETRYABLE:
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_s = max(1.0, float(retry_after))
                    except ValueError:
                        sleep_s = min(60.0, 2 ** attempt)
                else:
                    sleep_s = min(60.0, 2 ** attempt)

                if attempt >= max_retries:
                    r.raise_for_status()

                print(
                    f"RETRY offset={offset} attempt={attempt+1}/{max_retries} "
                    f"http={r.status_code} sleep={sleep_s:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            payload = r.json()
            rows = payload.get("rows", [])
            if not rows:
                raise RuntimeError(f"No rows returned at source offset {offset}")
            return rows

        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt >= max_retries:
                raise
            sleep_s = min(60.0, 2 ** attempt)
            print(
                f"RETRY offset={offset} attempt={attempt+1}/{max_retries} "
                f"reason={type(exc).__name__} sleep={sleep_s:.1f}s",
                flush=True,
            )
            time.sleep(sleep_s)

    raise RuntimeError("retry loop exhausted")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--request-delay", type=float, default=1.0)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(out) + ".partial")
    state_file = Path(str(out) + ".state.json")

    if args.fresh:
        partial.unlink(missing_ok=True)
        state_file.unlink(missing_ok=True)

    if partial.exists() and state_file.exists():
        state = json.loads(state_file.read_text())
        written = int(state["written"])
        source_offset = int(state["source_offset"])
        mode = "a"
        print(f"RESUME written={written} source_offset={source_offset}", flush=True)
    else:
        written = 0
        source_offset = 0
        mode = "w"

    headers: dict[str, str] = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    session = requests.Session()
    started = time.perf_counter()

    with partial.open(mode, encoding="utf-8") as fh:
        while written < args.rows:
            length = min(args.batch_size, args.rows - written)
            rows = request_batch(session, headers, source_offset, length)

            accepted = 0
            for item in rows:
                raw = item.get("row", item)
                rec = normalize(raw, source_offset)
                source_offset += 1

                if not rec["document_id"] or not rec["abstract"].strip():
                    continue

                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                accepted += 1
                if written >= args.rows:
                    break

            fh.flush()
            state_file.write_text(json.dumps({
                "dataset": DATASET,
                "written": written,
                "source_offset": source_offset,
                "target_rows": args.rows,
            }, indent=2))

            print(
                f"FETCHED={written}/{args.rows} SOURCE_OFFSET={source_offset} "
                f"ACCEPTED_LAST_BATCH={accepted}",
                flush=True,
            )

            if written < args.rows and args.request_delay > 0:
                time.sleep(args.request_delay)

    partial.replace(out)
    state_file.unlink(missing_ok=True)

    elapsed = time.perf_counter() - started
    print(f"DATASET={DATASET}")
    print(f"WRITTEN_ROWS={written}")
    print(f"OUTPUT={out}")
    print(f"BYTES={out.stat().st_size}")
    print(f"ELAPSED_SECONDS={elapsed:.2f}")
    print("ARXIV_FETCH=PASS" if written == args.rows else "ARXIV_FETCH=FAIL")


if __name__ == "__main__":
    main()
