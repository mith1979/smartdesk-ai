import json
from pathlib import Path

from datasets import load_dataset

# Save relative to the project folder.
project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("strova-ai/hr-policies-qa-dataset")

for split_name, records in dataset.items():
    print(f"\nSplit: {split_name}")
    print(f"Number of records: {len(records)}")
    print(f"Columns: {records.column_names}")

    output_file = raw_dir / f"hr_{split_name}.jsonl"
    records.to_json(str(output_file), force_ascii=False)
    print(f"Saved to: {output_file}")

    print("\nFirst record:")
    print(json.dumps(records[0], indent=2, ensure_ascii=False))