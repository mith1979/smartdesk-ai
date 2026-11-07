import json
from collections import Counter
from pathlib import Path

project_dir = Path(__file__).resolve().parents[1]
input_file = project_dir / "data/raw/it_train.jsonl"

with input_file.open(encoding="utf-8") as source:
    tickets = [json.loads(line) for line in source if line.strip()]

counts = Counter(ticket.get("category", "Unknown") for ticket in tickets)

print(f"Total tickets: {len(tickets)}")
print("\nCategories:")
for category, count in counts.most_common():
    print(f"  {category}: {count}")

print("\nFirst 3 examples per category:")

for category in sorted(counts):
    examples = [
        ticket for ticket in tickets
        if ticket.get("category", "Unknown") == category
    ][:3]

    print(f"\n=== {category} ===")
    for ticket in examples:
        print(f"\nID: {ticket['id']}")
        print(f"Subject: {ticket['subject']}")
        print(f"Description: {ticket['description']}")