"""
File: scripts/hybrid_search.py

Purpose:
    Combine dense and BM25 retrieval using reciprocal rank fusion.

Input:
    Employee question.
    Existing Chroma collection containing our indexed KB passages.

Output:
    Top hybrid Documents, with RRF scores and original ranks.

FDE connection:
    Week 2: Reuse LangChain Documents and source metadata.
    Week 6: Combine semantic and lexical retrieval.
    Week 3: Prepare results for comparison with the dense baseline.

Pipeline:
    Question → dense candidates + BM25 candidates
             → deduplicate by chunk ID → RRF → top Documents

Side effects:
    Calls OpenAI to embed the question for dense retrieval.
    Reads Chroma and builds BM25 in memory.
    No chat-model calls, document re-embedding, or index updates.

Limitations:
    Equal contribution from both retrieval methods.
    RRF is rank fusion, not a learned semantic re-ranker.
    Improvement must be measured; it is not guaranteed.
"""

import argparse

from dotenv import load_dotenv

from build_index import PROJECT_DIR, FDE_DIR
from bm25_search import build_bm25, search_bm25
from rag import retrieve as retrieve_dense


def fuse_rankings(dense_documents, bm25_documents, k=4, constant=60):
    """Merge ranked Documents and return top entries."""
    if k < 1 or constant < 1:
        raise ValueError("k and constant must be positive")

    entries = {}

    for method, documents in (
        ("dense", dense_documents),
        ("bm25", bm25_documents),
    ):
        seen = set()

        for rank, document in enumerate(documents, start=1):
            chunk_id = document.metadata["chunk_id"]

            # Count a chunk at most once per retrieval method.
            if chunk_id in seen:
                continue
            seen.add(chunk_id)

            if chunk_id not in entries:
                entries[chunk_id] = {
                    "document": document,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                }

            entry = entries[chunk_id]
            entry["rrf_score"] += 1.0 / (constant + rank)
            entry[f"{method}_rank"] = rank

    # Higher RRF is better. Chunk ID breaks exact ties predictably.
    ranked = sorted(
        entries.values(),
        key=lambda entry: (
            -entry["rrf_score"],
            entry["document"].metadata["chunk_id"],
        ),
    )

    return ranked[:k]


def retrieve_hybrid(question, k=4, fetch_k=10):
    """Retrieve candidates from both methods, then fuse them."""
    if k < 1 or fetch_k < k:
        raise ValueError("Require fetch_k >= k >= 1")

    # Fetch more candidates than we ultimately return.
    dense_documents = retrieve_dense(question, k=fetch_k)

    bm25, documents, tokenized_passages = build_bm25()
    bm25_results = search_bm25(
        question,
        bm25,
        documents,
        tokenized_passages,
        k=fetch_k,
    )

    # RRF needs rank order, not the original BM25 score.
    bm25_documents = [doc for doc, score in bm25_results]

    return fuse_rankings(
        dense_documents,
        bm25_documents,
        k=k,
    )


def main():
    parser = argparse.ArgumentParser(description="Hybrid KB retrieval")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--fetch-k", type=int, default=10)
    args = parser.parse_args()

    if not args.question.strip():
        parser.error("Question must not be empty")
    if args.k < 1 or args.fetch_k < args.k:
        parser.error("Require --fetch-k >= --k >= 1")

    load_dotenv(PROJECT_DIR / ".env", override=False)
    load_dotenv(FDE_DIR / ".env", override=False)

    results = retrieve_hybrid(
        args.question,
        k=args.k,
        fetch_k=args.fetch_k,
    )

    print(f"\nQuestion: {args.question}")
    print(f"Candidates per method: up to {args.fetch_k}")

    for rank, entry in enumerate(results, start=1):
        document = entry["document"]
        metadata = document.metadata

        print(
            f"\n--- Hybrid result {rank} | "
            f"RRF: {entry['rrf_score']:.6f} ---"
        )
        print(f"Chunk: {metadata['chunk_id']}")
        print(
            f"Dense rank: {entry['dense_rank']} | "
            f"BM25 rank: {entry['bm25_rank']}"
        )
        print(f"Section: {metadata.get('section', '')}")
        print(document.page_content)

    if not results:
        print("No candidates retrieved.")


if __name__ == "__main__":
    main()