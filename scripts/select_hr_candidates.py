import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "data/raw/hr_train.jsonl"
OUTPUT_FILE = PROJECT_DIR / "data/raw/hr_candidates.json"

# These are 1-based record numbers from our earlier exploration.
SELECTED_RECORDS = {1, 2, 5, 8, 9, 229, 251, 282, 433, 559, 605}

candidates = []

with INPUT_FILE.open(encoding="utf-8") as source:
    for record_number, line in enumerate(source, start=1):
        if not line.strip():
            continue

        if record_number not in SELECTED_RECORDS:
            continue

        record = json.loads(line)

        candidate = {
            "source_dataset": "strova-ai/hr-policies-qa-dataset",
            "source_file": "data/raw/hr_train.jsonl",
            "source_record": record_number,
            "review_status": "candidate",
            "messages": record["messages"],
        }
        candidates.append(candidate)

OUTPUT_FILE.write_text(
    json.dumps(candidates, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Selected {len(candidates)} candidate records.")
print(f"Saved: {OUTPUT_FILE}")

for candidate in candidates:
    print(f"\n--- Source record {candidate['source_record']} ---")

    for message in candidate["messages"]:
        if message["role"] == "user":
            print(f"Question: {message['content']}")
        elif message["role"] == "assistant":
            print(f"Answer: {message['content']}")