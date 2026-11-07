import json
import random
from pathlib import Path

project_dir = Path(__file__).resolve().parents[1]
input_file = project_dir / "data" / "raw" / "hr_train.jsonl"

with input_file.open(encoding="utf-8") as file:
    records = [json.loads(line) for line in file if line.strip()]

# A fixed seed gives us the same sample each time.
rng = random.Random(42)
indices = sorted(rng.sample(range(len(records)), min(15, len(records))))

print(f"Total records: {len(records)}")
print(f"Inspecting {len(indices)} examples across the dataset")

for index in indices:
    print(f"\n--- Source record {index + 1} ---")

    for message in records[index]["messages"]:
        if message["role"] == "user":
            print(f"Question: {message['content']}")
        elif message["role"] == "assistant":
            print(f"Answer:   {message['content']}")