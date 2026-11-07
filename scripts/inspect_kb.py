from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
KB_DIR = PROJECT_DIR / "data" / "kb"


def read_document(path):
    text = path.read_text(encoding="utf-8")

    # Separate the YAML metadata header from the document body.
    parts = text.split("---", 2)

    if len(parts) != 3 or parts[0].strip():
        raise ValueError(f"Missing metadata header: {path}")

    metadata = {}
    for line in parts[1].strip().splitlines():
        key, separator, value = line.partition(":")
        if separator:
            metadata[key.strip()] = value.strip().strip('"').strip("'")

    body = parts[2].strip()
    return metadata, body


def main():
    paths = sorted(KB_DIR.rglob("*.md"))
    if not paths:
        raise ValueError(f"No Markdown documents found in {KB_DIR}")

    seen_ids = set()

    for path in paths:
        metadata, body = read_document(path)

        for field in ("id", "domain", "topic", "review_status"):
            if not metadata.get(field):
                raise ValueError(f"Missing '{field}': {path}")

        if metadata["id"] in seen_ids:
            raise ValueError(f"Duplicate document ID: {metadata['id']}")
        seen_ids.add(metadata["id"])

        if metadata["review_status"] != "reviewed":
            raise ValueError(f"Document is not marked reviewed: {path}")

        if not body:
            raise ValueError(f"Empty document: {path}")

        sections = [
            line[3:].strip()
            for line in body.splitlines()
            if line.startswith("## ")
        ]

        print(f"\nFile: {path.relative_to(PROJECT_DIR)}")
        print(f"ID: {metadata['id']}")
        print(f"Domain: {metadata['domain']}")
        print(f"Topic: {metadata['topic']}")
        print(f"Words: {len(body.split())}")
        print(f"Sections: {', '.join(sections)}")

    print(f"\nSuccessfully inspected {len(paths)} reviewed documents.")


if __name__ == "__main__":
    main()