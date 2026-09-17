from datasets import load_dataset
DATASET = "common-pile/arxiv_abstracts"
ds = load_dataset(DATASET, split="train", streaming=True)
row = next(iter(ds))
print(f"DATASET={DATASET}")
print("FIELDS=" + ",".join(sorted(row.keys())))
for k, v in row.items():
    s = repr(v)
    if len(s) > 220:
        s = s[:220] + "..."
    print(f"{k}={s}")
print("HF_SCHEMA_INSPECTION=PASS")
