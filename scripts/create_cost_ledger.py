from pathlib import Path
import csv
root = Path(__file__).resolve().parents[1]
path = root / "results" / "cost_ledger.csv"
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists():
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "timestamp_utc","phase","resource","operation",
            "quantity","cost_usd","cost_type","notes"
        ])
print(f"COST_LEDGER={path}")
