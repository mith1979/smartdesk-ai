"""
File: scripts/search_kb.py

Purpose:
    Search the existing Chroma index and display relevant KB passages.

Input:
    A question supplied as a command-line argument.
    Existing index in storage/chroma/.
    OPENAI_API_KEY from the environment or .env.

Output:
    Top matching chunks with source, section, and distance.

FDE connection:
    Week 2: Chroma similarity search and source metadata.
    Week 3: Inspect retrieved evidence before evaluating answers.
    Establishes the dense baseline for Week 6 hybrid search.

Side effects:
    Sends the question to OpenAI for embedding, incurring API usage.
    Does not re-embed KB documents or generate an answer.
"""

import argparse
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Reuse the exact configuration used during indexing.
# Importing build_index does not execute its main() function.
from build_index import (
    PROJECT_DIR,
    FDE_DIR,
    INDEX_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


def main():
    parser = argparse.ArgumentParser(description="Search SmartDesk KB")
    parser.add_argument("question", help="Employee question")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1")

    load_dotenv(PROJECT_DIR / ".env", override=False)
    load_dotenv(FDE_DIR / ".env", override=False)

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing")

    if not INDEX_DIR.exists():
        raise FileNotFoundError("Run scripts/build_index.py first")

    # Use the same embedding model as the stored document vectors.
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Open the existing collection; do not silently create a new one.
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(INDEX_DIR),
        create_collection_if_not_exists=False,
    )

    # breakpoint()
    # Embed the question and retrieve the nearest stored chunks.
    results = store.similarity_search_with_score(
        args.question,
        k=args.k,
    )

    print(f"\nQuestion: {args.question}")

    for rank, (document, distance) in enumerate(results, start=1):
        metadata = document.metadata

        print(f"\n--- Result {rank} | Distance: {distance:.4f} ---")
        print(f"Chunk: {metadata['chunk_id']}")
        print(f"Source: {metadata['source']}")
        print(f"Section: {metadata['section']}")
        print(document.page_content)

    if not results:
        print("No results found.")


if __name__ == "__main__":
    main()