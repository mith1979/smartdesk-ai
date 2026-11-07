import json
from pathlib import Path

project_dir = Path(__file__).resolve().parents[1]
input_file = project_dir / "data/processed/chunks.jsonl"

with input_file.open(encoding="utf-8") as file:
    chunks = [json.loads(line) for line in file if line.strip()]

if not chunks:
    raise ValueError("No chunks found")

sizes = [len(chunk["text"].split()) for chunk in chunks]

print(f"Total chunks: {len(chunks)}")
print(f"Words per chunk: min={min(sizes)}, "
      f"max={max(sizes)}, average={sum(sizes) / len(sizes):.1f}")

print("\nLeave-policy chunks:")
for chunk in chunks:
    if chunk["document_id"] == "hr-leave-001":
        words = len(chunk["text"].split())
        print(f"\n{chunk['chunk_id']} | {words} words")
        print(chunk["text"])