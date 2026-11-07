"""
File: scripts/build_index.py

Purpose:
    Load prepared LangChain Documents, generate OpenAI embeddings,
    and save text, vectors, and metadata in a local Chroma index.

Input:
    data/processed/chunks.jsonl, through load_documents().
    OPENAI_API_KEY from the environment or an existing .env file.

Output:
    A persistent Chroma collection under storage/chroma/.

FDE connection:
    Week 2: Embeddings, Chroma, persistence, and document metadata.
    Creates the dense-search baseline for Week 3 evaluation
    and later Week 6 hybrid-search experiments.

Pipeline:
    KB Markdown → chunks → Documents → embeddings → Chroma

Side effects:
    Sends chunk text to OpenAI and incurs embedding API usage.
    Writes local database files.
    Re-running embeds the Documents again and updates matching IDs.
    It does not automatically remove entries deleted from the KB.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from load_documents import load_documents

# breakpoint()
PROJECT_DIR = Path(__file__).resolve().parents[1]

# For your current layout:
# fde_program/notebooks/week_11/smartdesk_capstone/
# FDE_DIR = PROJECT_DIR.parents[2]

# Use the shared FDE .env when running inside the course workspace.
# In a standalone clone, fall back to the project's own .env directory.
FDE_DIR = next(
    (parent for parent in PROJECT_DIR.parents if parent.name == "fde_program"),
    PROJECT_DIR,
)

INDEX_DIR = PROJECT_DIR / "storage" / "chroma"

# Keep the collection name tied to this embedding model.
COLLECTION_NAME = "smartdesk_sections_openai_small_v1"
EMBEDDING_MODEL = "text-embedding-3-small"


def main():
    # Prefer project-specific settings, then your existing FDE .env.
    # Neither call overwrites variables already in the environment.
    load_dotenv(PROJECT_DIR / ".env", override=False)
    load_dotenv(FDE_DIR / ".env", override=False)

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is missing. Set it in your environment "
            "or your project/FDE .env file."
        )

    # Reuse the loader you already tested.
    documents = load_documents()

    # Use our chunk IDs as the database entry IDs.
    ids = [doc.metadata["chunk_id"] for doc in documents]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs found")

    print(f"Documents to index: {len(documents)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    # This configures the embedding client.
    # Embedding requests happen when we add the Documents below.
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        max_retries=2,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Open or create our persistent local collection.
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(INDEX_DIR),
    )

    # LangChain generates embeddings from page_content.
    # Chroma stores the vectors, text, and metadata.
    # Existing matching IDs are updated rather than duplicated.
    stored_ids = vector_store.add_documents(
        documents=documents,
        ids=ids,
    )

    print(f"Indexed entries: {len(stored_ids)}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Saved under: {INDEX_DIR}")


if __name__ == "__main__":
    main()