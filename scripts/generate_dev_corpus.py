from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TOPICS = [
    ("machine_learning", ["transformer", "representation learning", "optimization", "classification", "embedding"]),
    ("distributed_systems", ["consensus", "replication", "scheduling", "fault tolerance", "storage"]),
    ("databases", ["query optimization", "transactions", "indexing", "lakehouse", "columnar storage"]),
    ("biology", ["genomics", "protein", "cellular", "sequence", "expression"]),
    ("physics", ["quantum", "particle", "field", "energy", "simulation"]),
    ("climate", ["temperature", "precipitation", "forecast", "carbon", "atmosphere"]),
    ("medicine", ["clinical", "therapy", "diagnosis", "outcomes", "patient"]),
    ("robotics", ["control", "navigation", "perception", "planning", "manipulation"]),
]

TEMPLATES = [
    "We study {a} for {topic} systems and evaluate {b} under varying workloads. The method combines {c} with controlled experiments and reports reproducible measurements.",
    "This work presents a scalable approach to {a} using {b}. Results show how {c} changes accuracy, efficiency, and robustness across representative datasets.",
    "We investigate {a} in the context of {topic}. Our experiments compare {b} and {c}, highlighting trade-offs in quality, latency, and computational cost.",
]

def build_doc(i: int, rng: random.Random) -> dict:
    topic, terms = TOPICS[i % len(TOPICS)]
    a, b, c = rng.sample(terms, 3)
    title = f"{a.title()} and {b.title()} for {topic.replace('_', ' ').title()}"
    abstract = rng.choice(TEMPLATES).format(a=a, b=b, c=c, topic=topic.replace("_", " "))
    body = " ".join([abstract] * rng.randint(8, 16))
    return {
        "document_id": f"DOC-{i:09d}",
        "title": title,
        "abstract": abstract,
        "body": body,
        "category": topic,
        "year": 2015 + (i % 12),
        "source": "project14_dev_corpus",
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--out", default="data/dev_corpus.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(14)
    with out.open("w", encoding="utf-8") as fh:
        for i in range(args.rows):
            fh.write(json.dumps(build_doc(i, rng), ensure_ascii=False) + "\n")

    print(f"CORPUS_ROWS={args.rows}")
    print(f"CORPUS_BYTES={out.stat().st_size}")
    print(f"CORPUS_PATH={out}")
    print("CORPUS_GENERATION=PASS")

if __name__ == "__main__":
    main()
