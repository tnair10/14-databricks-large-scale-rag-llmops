from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True)
ap.add_argument("--expected-rows", type=int, required=True)
args = ap.parse_args()

p = Path(args.file)
rows = []
with p.open(encoding="utf-8") as fh:
    for line in fh:
        rows.append(json.loads(line))

ids = [r.get("document_id") for r in rows]
abstract_lengths = [len((r.get("abstract") or "").strip()) for r in rows]

empty_ids = sum(not x for x in ids)
empty_abstracts = sum(n == 0 for n in abstract_lengths)
duplicate_ids = len(ids) - len(set(ids))
with_authors = sum(bool(r.get("authors")) for r in rows)
with_url = sum(bool(r.get("source_url")) for r in rows)
with_license = sum(bool(r.get("license")) for r in rows)

print(f"ROWS={len(rows)}")
print(f"EXPECTED_ROWS={args.expected_rows}")
print(f"EMPTY_IDS={empty_ids}")
print(f"EMPTY_ABSTRACTS={empty_abstracts}")
print(f"DUPLICATE_IDS={duplicate_ids}")
print(f"WITH_AUTHORS={with_authors}")
print(f"WITH_SOURCE_URL={with_url}")
print(f"WITH_LICENSE={with_license}")
print(f"AVG_ABSTRACT_CHARS={mean(abstract_lengths):.2f}")
print(f"MIN_ABSTRACT_CHARS={min(abstract_lengths)}")
print(f"MAX_ABSTRACT_CHARS={max(abstract_lengths)}")
print(f"SAMPLE_ID={ids[0] if ids else ''}")
print(f"SAMPLE_ABSTRACT={(rows[0].get('abstract') or '')[:160] if rows else ''}")

passed = (
    len(rows) == args.expected_rows
    and empty_ids == 0
    and empty_abstracts == 0
    and duplicate_ids == 0
)
print("CORPUS_VALIDATION=" + ("PASS" if passed else "FAIL"))
