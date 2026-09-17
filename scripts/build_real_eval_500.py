from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path

STOP = {
    "the","and","for","that","with","from","this","these","those","are","was","were","been","being",
    "into","onto","over","under","between","within","without","using","used","use","their","there",
    "which","where","when","what","who","why","how","can","could","may","might","will","would","should",
    "such","than","then","also","into","through","based","study","paper","results","method","methods",
    "analysis","approach","approaches","data","model","models","system","systems","we","our","they","it",
    "its","a","an","of","to","in","on","by","as","at","or","is","be","has","have","had"
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")

def toks(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="data/arxiv_real_10k.jsonl")
ap.add_argument("--rows", type=int, default=500)
ap.add_argument("--output", default="data/real_retrieval_eval_500.jsonl")
args = ap.parse_args()

src = Path(args.input)
docs = []
with src.open(encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        if r.get("document_id") and (r.get("abstract") or "").strip():
            docs.append(r)

N = len(docs)
if N < args.rows:
    raise SystemExit(f"Need at least {args.rows} docs; found {N}")

df = collections.Counter()
per_doc_terms = []
for d in docs:
    terms = [t for t in toks(d["abstract"]) if t not in STOP and not t.isdigit()]
    unique = set(terms)
    df.update(unique)
    per_doc_terms.append(terms)

# Deterministic spread through the 10K corpus.
step = N / args.rows
selected_idx = [min(N - 1, int(i * step)) for i in range(args.rows)]

out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)

rows = []
for qnum, idx in enumerate(selected_idx):
    d = docs[idx]
    tf = collections.Counter(per_doc_terms[idx])

    scored = []
    for term, freq in tf.items():
        if df[term] < 2:
            continue
        idf = math.log((N + 1) / (df[term] + 1)) + 1.0
        scored.append((freq * idf, term))
    scored.sort(reverse=True)

    chosen = []
    for _, term in scored:
        if term not in chosen:
            chosen.append(term)
        if len(chosen) == 6:
            break

    if len(chosen) < 4:
        # fallback to frequent non-stop words
        for term, _ in tf.most_common():
            if term not in chosen and term not in STOP:
                chosen.append(term)
            if len(chosen) == 6:
                break

    query_text = " ".join(chosen)
    difficulty = "medium" if len(chosen) >= 6 else "easy"

    rows.append({
        "query_id": f"REAL-Q-{qnum:05d}",
        "query_text": query_text,
        "relevant_document_ids": [d["document_id"]],
        "difficulty": difficulty,
        "source": "common-pile/arxiv_abstracts_tfidf_keyword_eval",
        "created_at": None,
    })

with out.open("w", encoding="utf-8") as fh:
    for row in rows:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"SOURCE_DOCUMENTS={N}")
print(f"EVAL_QUERIES={len(rows)}")
print(f"DISTINCT_RELEVANT_DOCS={len({r['relevant_document_ids'][0] for r in rows})}")
print(f"OUTPUT={out}")
print(f"BYTES={out.stat().st_size}")
print("SAMPLE_QUERY_1=" + rows[0]["query_text"])
print("SAMPLE_DOC_1=" + rows[0]["relevant_document_ids"][0])
print("REAL_EVAL_BUILD=PASS")
