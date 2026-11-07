import json
from pathlib import Path

from datasets import load_dataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

dataset_name = "Console-AI/IT-helpdesk-synthetic-tickets"
dataset = load_dataset(dataset_name)

for split_name, records in dataset.items():
    print(f"\nDataset: {dataset_name}")
    print(f"Split: {split_name}")
    print(f"Records: {len(records)}")
    print(f"Columns: {records.column_names}")

    output_file = RAW_DIR / f"it_{split_name}.jsonl"
    records.to_json(str(output_file), force_ascii=False)
    print(f"Saved: {output_file}")

    for index in range(min(3, len(records))):
        print(f"\n--- Record {index + 1} ---")
        print(json.dumps(records[index], indent=2, ensure_ascii=False))