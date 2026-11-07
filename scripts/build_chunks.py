# Convert Python dictionaries into JSON text for saving.
import json

# Use regular expressions to split Markdown at section headings.
import re

# Work with file and directory paths.
from pathlib import Path

# Reuse our existing function from scripts/inspect_kb.py.
# It returns:
#   metadata: the fields between the opening --- markers
#   body:     the Markdown content below that header
from inspect_kb import read_document


# __file__ is this script's path.
# resolve() makes it an absolute path.
# parents[0] is scripts/; parents[1] is smartdesk_capstone/.
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Only reviewed KB documents are read—not data/raw/.
KB_DIR = PROJECT_DIR / "data" / "kb"

# Generated chunks will be written here.
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "chunks.jsonl"


def main():
    # Collect all chunks from all documents in this list.
    chunks = []

    # rglob searches this folder AND its subfolders for Markdown files.
    # sorted makes the processing order predictable.
    for path in sorted(KB_DIR.rglob("*.md")):

        # Separate the metadata header from the document text.
        metadata, body = read_document(path)

        # Stop if a document has not been marked reviewed.
        # get() returns None if the field is missing.
        if metadata.get("review_status") != "reviewed":
            raise ValueError(f"Document is not reviewed: {path}")

        # Find the first top-level Markdown heading, such as:
        #   # NovaTech Solutions — Leave Policy
        #
        # line[2:] removes the "# " prefix.
        # strip() removes whitespace around the title.
        #
        # next() returns the first matching title.
        # If no title exists, use the filename without ".md".
        title = next(
            (
                line[2:].strip()
                for line in body.splitlines()
                if line.startswith("# ")
            ),
            path.stem,
        )

        # Split the body at Markdown section headings:
        #   ## Eligibility
        #   ## Leave Allowances
        #
        # Regex:
        #   ^       start of a line
        #   ##      literal heading markers
        #   \s+     one or more whitespace characters
        #   (.+)    capture the heading text
        #   $       end of a line
        #
        # MULTILINE lets ^ and $ match individual lines.
        #
        # Because the heading is captured in parentheses,
        # re.split() keeps it in the resulting list.
        parts = re.split(r"^##\s+(.+)$", body, flags=re.MULTILINE)

        # The first item is everything BEFORE the first ## heading:
        # the document title and introductory paragraphs.
        introduction = parts[0].strip()

        # If nothing was split, the document has no ## sections.
        if len(parts) == 1:
            raise ValueError(f"No sections found: {path}")

        # The list alternates between headings and their content:
        #
        # parts[0] = introduction
        # parts[1] = first heading
        # parts[2] = first section's content
        # parts[3] = second heading
        # parts[4] = second section's content
        #
        # range(1, len(parts), 2) selects heading positions: 1, 3, 5...
        # enumerate(..., start=1) also supplies section numbers: 1, 2, 3...
        for section_number, index in enumerate(
            range(1, len(parts), 2), start=1
        ):
            heading = parts[index].strip()
            content = parts[index + 1].strip()

            # Stop rather than create a chunk with an empty section.
            if not content:
                raise ValueError(f"Empty section '{heading}': {path}")

            # Create one dictionary representing this section.
            chunks.append({
                # Example: hr-leave-001::section-01
                # :02d formats the section number with at least two digits.
                "chunk_id": (
                    f"{metadata['id']}::section-{section_number:02d}"
                ),

                # Identify the parent document and section.
                "document_id": metadata["id"],
                "title": title,
                "section": heading,

                # Copy classification metadata from the document header.
                "domain": metadata["domain"],
                "topic": metadata["topic"],

                # Optional metadata: use an empty string when absent.
                "version": metadata.get("version", ""),
                "source_type": metadata.get("source_type", ""),

                # Preserve origins for dataset-derived documents.
                "source_dataset": metadata.get("source_dataset", ""),
                "source_records": metadata.get("source_records", ""),

                # Preserve inspiration for synthetic IT guides.
                "inspiration_dataset": metadata.get(
                    "inspiration_dataset", ""
                ),
                "inspiration_ticket_ids": metadata.get(
                    "inspiration_ticket_ids", ""
                ),

                # Store a project-relative source path, for example:
                # data/kb/hr/leave_policy.md
                # as_posix() uses forward slashes.
                "source": path.relative_to(PROJECT_DIR).as_posix(),

                # This is the searchable text:
                # document introduction + section heading + section content.
                #
                # Repeat the introduction so the chunk retains
                # the document title and fictional-company context.
                "text": (
                    f"{introduction}\n\n"
                    f"## {heading}\n{content}"
                ),
            })

    # Stop if no documents produced any chunks.
    if not chunks:
        raise ValueError("No chunks generated")

    # Extract all chunk IDs.
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    # A set removes duplicates.
    # Different lengths therefore indicate duplicate IDs.
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs found")

    # Create data/processed/ if it does not exist.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # "w" replaces the previous chunks.jsonl file.
    # The original Markdown documents remain unchanged.
    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for chunk in chunks:

            # JSONL = one complete JSON object per line.
            # ensure_ascii=False preserves characters such as × and —.
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Count distinct parent document IDs.
    print(f"Documents: {len(set(c['document_id'] for c in chunks))}")

    # Count all generated section chunks.
    print(f"Chunks: {len(chunks)}")
    print(f"Saved: {OUTPUT_FILE}")

    # Display the first chunk in readable, indented JSON.
    # This formatting is for the terminal, not the JSONL file.
    print("\nFirst chunk:")
    print(json.dumps(chunks[0], indent=2, ensure_ascii=False))


# Run main() when this file is executed directly:
#   python scripts/build_chunks.py
#
# Do not run main() automatically if another script imports this module.
if __name__ == "__main__":
    main()