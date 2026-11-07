"""
File: scripts/bm25_search.py

Purpose:
    Search the indexed KB passages using BM25 keyword retrieval.

Input:
    Text and metadata from the existing Chroma collection.
    A question supplied on the command line.

Output:
    Ranked LangChain Documents and their BM25 scores.

FDE connection:
    Week 2: Documents and metadata.
    Week 6: Sparse lexical retrieval, to complement dense search.

Pipeline position:
    Question → tokenize → BM25 rank → candidate passages.
    These candidates will later be combined with dense results.

Side effects:
    Reads the existing local Chroma collection.
    Builds a temporary BM25 index in memory.
    No embedding calls, LLM calls, or KB updates.

Limitations:
    Basic lowercase word tokenization.
    No stemming, synonym expansion, or stopword removal yet.
    BM25 scores are ranking scores, not confidence probabilities.
"""

import argparse
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from build_index import INDEX_DIR, COLLECTION_NAME


def tokenize(text):
    """Apply the same simple tokenization to passages and questions."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25():
    """Read the same passages used by dense search."""
    if not INDEX_DIR.exists():
        raise FileNotFoundError("Run build_index.py first")

    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(INDEX_DIR),
        create_collection_if_not_exists=False,
    )
    # breakpoint()
    # Reading stored text does not generate embeddings.
    stored = store.get(include=["documents", "metadatas"])

    if not stored["ids"]:
        raise ValueError("The Chroma collection is empty")

    # Sort by ID to make input order predictable.
    rows = sorted(
        zip(
            stored["ids"],
            stored["documents"],
            stored["metadatas"],
        ),
        key=lambda row: row[0],
    )

    documents = []
    tokenized_passages = []

    for chunk_id, text, metadata in rows:
        if not text or not text.strip():
            raise ValueError(f"Empty text for {chunk_id}")

        tokens = tokenize(text)
        if not tokens:
            raise ValueError(f"No searchable tokens for {chunk_id}")

        metadata = dict(metadata or {})
        metadata["chunk_id"] = chunk_id

        documents.append(
            Document(page_content=text, metadata=metadata)
        )
        tokenized_passages.append(tokens)

    bm25 = BM25Okapi(tokenized_passages)
    return bm25, documents, tokenized_passages


def search_bm25(question, bm25, documents, tokenized_passages, k=4):
    """Return keyword-matching Documents with their BM25 scores."""
    if k < 1:
        raise ValueError("k must be at least 1")
    # breakpoint()
    query_tokens = tokenize(question)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    query_terms = set(query_tokens)

    # Exclude passages with no query-term overlap.
    candidates = [
        index
        for index, tokens in enumerate(tokenized_passages)
        if query_terms.intersection(tokens)
    ]

    # Higher BM25 scores rank first.
    ranked = sorted(
        candidates,
        key=lambda index: (
            -float(scores[index]),
            documents[index].metadata["chunk_id"],
        ),
    )

    return [
        (documents[index], float(scores[index]))
        for index in ranked[:k]
    ]


def main():
    parser = argparse.ArgumentParser(description="Search KB with BM25")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    if not args.question.strip():
        parser.error("Question must not be empty")
    if args.k < 1:
        parser.error("--k must be at least 1")

    bm25, documents, tokenized_passages = build_bm25()

    results = search_bm25(
        args.question,
        bm25,
        documents,
        tokenized_passages,
        k=args.k,
    )

    print(f"Indexed passages: {len(documents)}")
    print(f"Question: {args.question}")

    for rank, (document, score) in enumerate(results, start=1):
        metadata = document.metadata
        print(f"\n--- Result {rank} | BM25 score: {score:.4f} ---")
        print(f"Chunk: {metadata['chunk_id']}")
        print(f"Section: {metadata.get('section', '')}")
        print(document.page_content)

    if not results:
        print("No keyword matches found.")


if __name__ == "__main__":
    main()