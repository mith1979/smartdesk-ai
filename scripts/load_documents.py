"""
File: scripts/load_documents.py

Purpose:
    Read our prepared chunks and convert each into a LangChain
    Document containing page_content and metadata.

Input:
    data/processed/chunks.jsonl — currently 109 chunks.

Output:
    An in-memory list of LangChain Documents.
    Prints the count and first Document for inspection.

FDE connection:
    Week 2: LangChain Documents and metadata.
    These Documents will later support Week 3 evaluation
    and Week 6 retrieval experiments.

Pipeline position:
    Reviewed KB → chunks → Documents → embeddings → Chroma

Side effects:
    Reads a local file. No API calls or file writes.
"""



import json
from pathlib import Path

from langchain_core.documents import Document

PROJECT_DIR = Path(__file__).resolve().parents[1]
CHUNKS_FILE = PROJECT_DIR / "data/processed/chunks.jsonl"


def load_documents():
    """Load our prepared chunks as LangChain Document objects."""
    documents = []

    with CHUNKS_FILE.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            chunk = json.loads(line)

            # This is the actual policy or troubleshooting passage.
            text = chunk["text"]

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Missing chunk text on line {line_number}"
                )

            # Everything except the text describes the chunk:
            # IDs, domain, section, source path, and provenance.
            metadata = {
                key: value
                for key, value in chunk.items()
                if key != "text"
            }

            # One existing chunk becomes one LangChain Document.
            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

    if not documents:
        raise ValueError("No documents loaded")

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print(f"LangChain Documents: {len(documents)}")

    print("\nFirst document metadata:")
    print(json.dumps(
        documents[0].metadata,
        indent=2,
        ensure_ascii=False,
    ))

    print("\nFirst document text:")
    print(documents[0].page_content)