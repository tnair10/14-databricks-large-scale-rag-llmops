from __future__ import annotations
import argparse, json
from pathlib import Path
from datasets import load_dataset

DATASET = "common-pile/arxiv_abstracts"

def first_value(row, names, default=None):
    for n in names:
        if n in row and row[n] not in (None, "", []):
            return row[n]
    return default

def normalize(row, i):
    doc_id = first_value(row, ["id","paper_id","arxiv_id","article_id"], f"ARXIV-{i:09d}")
    title = first_value(row, ["title"], "")
    abstract = first_value(row, ["abstract","text","content"], "")
    cats = first_value(row, ["categories","category","subjects"], [])
    if isinstance(cats, str):
        cats = [x for x in cats.replace(",", " ").split() if x]
    published = first_value(row, ["update_date","published","published_date","date"], None)
    url = first_value(row, ["url","source_url"], None)
    return {
        "document_id": str(doc_id),
        "title": str(title),
        "abstract": str(abstract),
        "categories": cats or [],
        "published_date": None if published is None else str(published),
        "source": DATASET,
        "source_url": url,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--out", default="data/arxiv_sample.jsonl")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET, split="train", streaming=True)
    written = 0
    with out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            rec = normalize(row, i)
            if not rec["title"] or not rec["abstract"]:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written >= args.rows:
                break
    print(f"DATASET={DATASET}")
    print(f"WRITTEN_ROWS={written}")
    print(f"OUTPUT={out}")
    print(f"BYTES={out.stat().st_size}")
    print("ARXIV_SAMPLE=PASS" if written == args.rows else "ARXIV_SAMPLE=PARTIAL")

if __name__ == "__main__":
    main()
