from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import quote

import mlflow
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

INDEX_NAME = f"{catalog}.{schema}.scientific_chunks_dev_ai_index"
MODEL_SERVICE = f"{catalog}.{schema}.p14-rag-llm-dev"
EXPERIMENT = "/Shared/project14-large-scale-rag"

HEADERS = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

os.environ["DATABRICKS_HOST"] = host
os.environ["DATABRICKS_TOKEN"] = token
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(EXPERIMENT)


def _extract_response_text(payload: dict) -> str:
    """
    Return only assistant final output_text.

    GPT-OSS is a reasoning model. Unity Gateway/Open Responses returns
    reasoning separately as output items with type='reasoning' and
    content type='reasoning_text'. Those items are intentionally ignored.
    """
    pieces = []

    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue

        if item.get("type") != "message":
            continue

        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue

            if content.get("type") == "output_text" and content.get("text"):
                pieces.append(str(content["text"]))

    if pieces:
        return "\n".join(pieces).strip()

    # Defensive fallback only if the service provides a top-level final answer.
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()

    raise RuntimeError(
        "No assistant output_text found in model response. "
        "Reasoning items were intentionally excluded."
    )


@mlflow.trace(name="retrieve_hybrid", span_type="RETRIEVER")
def retrieve_hybrid(question: str, k: int = 5) -> list[dict]:
    url = f"{host}/api/2.0/vector-search/indexes/{quote(INDEX_NAME, safe='')}/query"
    body = {
        "columns": ["chunk_id", "document_id", "category", "chunk_text"],
        "num_results": k,
        "query_text": question,
        "query_type": "HYBRID",
    }
    started = time.perf_counter()
    r = requests.post(url, headers=HEADERS, json=body, timeout=120)
    latency_ms = (time.perf_counter() - started) * 1000
    r.raise_for_status()

    payload = r.json()
    columns = [x.get("name") for x in payload.get("manifest", {}).get("columns", [])]
    data = payload.get("result", {}).get("data_array", [])
    rows = [dict(zip(columns, row)) for row in data]

    span = mlflow.get_current_active_span()
    if span is not None:
        span.set_attributes({
            "retrieval.index": INDEX_NAME,
            "retrieval.query_type": "HYBRID",
            "retrieval.k": k,
            "retrieval.latency_ms": round(latency_ms, 3),
            "retrieval.result_count": len(rows),
        })
    return rows


@mlflow.trace(name="build_grounded_prompt", span_type="CHAIN")
def build_grounded_prompt(question: str, rows: list[dict]) -> str:
    blocks = []
    for i, row in enumerate(rows, start=1):
        blocks.append(
            f"[{i}] chunk_id={row.get('chunk_id')} "
            f"document_id={row.get('document_id')} "
            f"category={row.get('category')}\n"
            f"{row.get('chunk_text', '')}"
        )
    context = "\n\n".join(blocks)

    return f"""You are a scientific research assistant.

Use ONLY the retrieved context below to answer the question.

Retrieved context:
{context}

Question:
{question}

Requirements:
- Answer only from the retrieved context.
- If the retrieved context is insufficient, say so.
- Do not invent facts, citations, paper titles, authors, or references.
- Cite supporting chunks inline using [1], [2], etc.
- Return only the final answer.
- Do not include reasoning, analysis, planning, scratch work, or commentary about how you formed the answer.
- Keep the answer concise.
"""


@mlflow.trace(name="generate_answer", span_type="LLM")
def generate_answer(prompt: str) -> tuple[str, dict]:
    url = f"{host}/ai-gateway/mlflow/v1/responses"
    body = {
        "model": MODEL_SERVICE,
        "max_output_tokens": 256,
        "input": [{
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}],
        }],
    }

    started = time.perf_counter()
    r = requests.post(url, headers=HEADERS, json=body, timeout=180)
    latency_ms = (time.perf_counter() - started) * 1000
    if r.status_code != 200:
        raise RuntimeError(f"MODEL_CALL_FAILED status={r.status_code} body={r.text[:2000]}")

    payload = r.json()
    answer = _extract_response_text(payload)
    usage = payload.get("usage") or {}

    span = mlflow.get_current_active_span()
    if span is not None:
        span.set_attributes({
            "llm.model_service": MODEL_SERVICE,
            "llm.latency_ms": round(latency_ms, 3),
            "llm.input_tokens": usage.get("input_tokens"),
            "llm.output_tokens": usage.get("output_tokens"),
            "llm.total_tokens": usage.get("total_tokens"),
        })

    return answer, {"latency_ms": latency_ms, "usage": usage, "raw_id": payload.get("id")}


@mlflow.trace(name="project14_dev_rag_final_only", span_type="CHAIN")
def rag(question: str) -> dict:
    mlflow.update_current_trace(tags={
        "project": "project14",
        "phase": "dev_rag_e2e_final_only",
        "retrieval_mode": "HYBRID",
        "model_service": MODEL_SERVICE,
    })
    started = time.perf_counter()
    rows = retrieve_hybrid(question, k=5)
    prompt = build_grounded_prompt(question, rows)
    answer, model_meta = generate_answer(prompt)
    end_to_end_ms = (time.perf_counter() - started) * 1000

    return {
        "question": question,
        "answer": answer,
        "retrieved": [{
            "rank": i,
            "chunk_id": row.get("chunk_id"),
            "document_id": row.get("document_id"),
            "category": row.get("category"),
            "score": row.get("score"),
        } for i, row in enumerate(rows, start=1)],
        "model": MODEL_SERVICE,
        "retrieval_index": INDEX_NAME,
        "model_meta": model_meta,
        "end_to_end_ms": end_to_end_ms,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--question",
        default="What aspects of distributed systems are evaluated in these scientific studies?",
    )
    args = ap.parse_args()
    result = rag(args.question)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "rag_dev_e2e_final_only_result.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("QUESTION=" + result["question"])
    print("MODEL_SERVICE=" + result["model"])
    print("RETRIEVAL_INDEX=" + result["retrieval_index"])
    print(f"END_TO_END_MS={result['end_to_end_ms']:.2f}")
    print("ANSWER:")
    print(result["answer"])
    print("USAGE=" + json.dumps(result["model_meta"].get("usage") or {}, sort_keys=True))
    print(f"RESULT_JSON={out_file}")
    print("RAG_DEV_E2E_FINAL_ONLY=PASS")
