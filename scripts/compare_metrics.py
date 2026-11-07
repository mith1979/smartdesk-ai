"""
File: scripts/compare_metrics.py

Purpose:
    Compare squared L2 distance and cosine similarity using
    the SAME existing document embeddings and query embedding.

Input:
    Existing Chroma collection in storage/chroma/.
    A question supplied on the command line.
    OPENAI_API_KEY from the environment or .env.

Output:
    Vector norms, top results for each metric, and top-k overlap.

FDE connection:
    Week 2: Embeddings, vector comparisons, and metadata.
    Week 3: Inspect whether rankings retrieve required evidence.
    Provides a baseline before Week 6 hybrid-search experiments.

Method:
    Read stored vectors from Chroma.
    Embed the question once using LangChain OpenAIEmbeddings.
    Calculate both metrics with NumPy across every stored vector.
    This is exact metric comparison, not Chroma's index search.

Side effects:
    Sends the question to OpenAI and incurs embedding API usage.
    Does not re-embed documents or add/update/delete KB entries.
"""

import argparse
import os

import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Importing these constants does not execute build_index.main().
from build_index import (
    PROJECT_DIR,
    FDE_DIR,
    INDEX_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compare L2 and cosine on stored KB vectors"
    )
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1")

    if not args.question.strip():
        parser.error("Question must not be empty")

    load_dotenv(PROJECT_DIR / ".env", override=False)
    load_dotenv(FDE_DIR / ".env", override=False)

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing")

    if not INDEX_DIR.exists():
        raise FileNotFoundError("Run build_index.py first")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(INDEX_DIR),
        create_collection_if_not_exists=False,
    )

    # Retrieve existing vectors and their matching metadata.
    # This operation does not generate embeddings.
    stored = store.get(include=["embeddings", "metadatas"])

    if not stored["ids"]:
        raise ValueError("The collection is empty")

    if stored["embeddings"] is None:
        raise ValueError("Stored embeddings were not returned")

    # Matrix shape: number_of_chunks × embedding_dimensions.
    vectors = np.asarray(stored["embeddings"], dtype=np.float64)

    # One query embedding is reused for BOTH metric calculations.
    query = np.asarray(
        embeddings.embed_query(args.question),
        dtype=np.float64,
    )

    if vectors.ndim != 2 or vectors.shape[1] != query.size:
        raise ValueError("Query and document dimensions do not match")

    # Norm means vector length.
    document_norms = np.linalg.norm(vectors, axis=1)
    query_norm = np.linalg.norm(query)

    if query_norm == 0 or np.any(document_norms == 0):
        raise ValueError("Cosine similarity requires nonzero vectors")

    # Squared L2: sum of squared coordinate differences.
    # axis=1 produces one distance per document vector.
    l2_distances = np.sum((vectors - query) ** 2, axis=1)

    # Cosine: dot product divided by the two vector lengths.
    # @ performs matrix-vector multiplication.
    cosine_scores = (vectors @ query) / (
        document_norms * query_norm
    )

    # Remove tiny floating-point excursions outside [-1, 1].
    cosine_scores = np.clip(cosine_scores, -1.0, 1.0)

    k = min(args.k, len(vectors))

    # Lower L2 is better; higher cosine similarity is better.
    l2_order = np.argsort(l2_distances, kind="stable")
    cosine_order = np.argsort(-cosine_scores, kind="stable")

    print(f"\nQuestion: {args.question}")
    print(f"Stored chunks: {len(vectors)}")
    print(f"Dimensions: {vectors.shape[1]}")
    print(f"Query norm: {query_norm:.6f}")
    print(
        "Document norms: "
        f"min={document_norms.min():.6f}, "
        f"max={document_norms.max():.6f}"
    )

    # Show both scores for every selected result.
    rankings = [
        ("SQUARED L2 — lower is better", l2_order),
        ("COSINE SIMILARITY — higher is better", cosine_order),
    ]

    for label, order in rankings:
        print(f"\n=== {label} ===")

        for rank, index in enumerate(order[:k], start=1):
            metadata = stored["metadatas"][index] or {}
            chunk_id = stored["ids"][index]

            print(
                f"{rank}. {chunk_id}\n"
                f"   L2²={l2_distances[index]:.6f} | "
                f"Cosine={cosine_scores[index]:.6f}\n"
                f"   Section: {metadata.get('section', '')}"
            )

    # Compare membership and order of the top-k results.
    overlap = len(set(l2_order[:k]) & set(cosine_order[:k]))
    same_order = np.array_equal(l2_order[:k], cosine_order[:k])

    print(f"\nTop-{k} overlap: {overlap}/{k}")
    print(f"Identical top-{k} order: {same_order}")


if __name__ == "__main__":
    main()