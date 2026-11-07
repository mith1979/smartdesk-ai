import json
from pathlib import Path

project_dir = Path(__file__).resolve().parents[1]
input_file = project_dir / "data/raw/hr_train.jsonl"
output_file = project_dir / "data/raw/hr_questions.txt"

with (
    input_file.open(encoding="utf-8") as source,
    output_file.open("w", encoding="utf-8") as output,
):
    for record_number, line in enumerate(source, start=1):
        if not line.strip():
            continue

        record = json.loads(line)

        for message in record["messages"]:
            if message["role"] == "user":
                output.write(
                    f"{record_number}: {message['content']}\n"
                )

print(f"Saved question list: {output_file}")