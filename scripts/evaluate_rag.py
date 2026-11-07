"""
File: scripts/evaluate_rag.py

Purpose:
    Run the saved development cases against our dense RAG pipeline.

Input:
    data/evaluation/rag_cases.json
    Existing Chroma index and OpenAI credentials.

Output:
    Per-case results and summary in the terminal.
    data/evaluation/dense_results.json containing answers,
    retrieved IDs, metrics, errors, and elapsed time.

FDE connection:
    Week 3: Separate retrieval evaluation from generation evaluation.
    Week 6: Establish a baseline for later hybrid-search comparison.

Metrics:
    Required-evidence coverage:
        Expected chunk IDs retrieved / expected chunk IDs.
    Answerability accuracy:
        Predicted answerable flag matches the expected flag.
    Citation validity:
        Cited IDs exist in the retrieved evidence, with at least
        one citation required for an answerable response.

Side effects:
    Calls query embedding and chat APIs for each case.
    Incurs API usage and overwrites dense_results.json.
    Does not rebuild the index or create tickets.

Limitations:
    These are development cases, not an independent test set.
    Expected chunks describe required evidence, not an exhaustive
    relevance judgment for every retrieved chunk.
    Factual correctness and completeness require manual review.
"""

import json
from time import perf_counter

from dotenv import load_dotenv

from build_index import PROJECT_DIR, FDE_DIR, EMBEDDING_MODEL
from rag import retrieve, generate_answer
import argparse


CASES_FILE = PROJECT_DIR / "data/evaluation/rag_cases.json"
# RESULTS_FILE = PROJECT_DIR / "data/evaluation/dense_results.json"
# TOP_K = 4


def main():
    # Select retrieval mode and output settings.
    parser = argparse.ArgumentParser(description="Evaluate SmartDesk RAG")
    parser.add_argument(
        "--retrieval",
        choices=["dense", "hybrid"],
        default="dense",
    )
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1")

    results_file = (
        PROJECT_DIR / "data/evaluation"
        / f"{args.retrieval}_results.json"
    )
    # Load credentials before calling the imported RAG functions.
    load_dotenv(PROJECT_DIR / ".env", override=False)
    load_dotenv(FDE_DIR / ".env", override=False)

    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))

    if not cases:
        raise ValueError("No evaluation cases found")

    results = []

    for case in cases:
        started = perf_counter()

        row = {
            "id": case["id"],
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "expected_answerable": case["answerable"],
        }

        try:
            # Both modes feed the same answer-generation function.
            if args.retrieval == "hybrid":
                from hybrid_search import retrieve_hybrid

                entries = retrieve_hybrid(
                    case["question"],
                    k=args.k,
                    fetch_k=max(10, args.k),
                )
                documents = [entry["document"] for entry in entries]
            else:
                documents = retrieve(case["question"], args.k)

            retrieved_ids = [
                doc.metadata["chunk_id"] for doc in documents
            ]
            expected_ids = set(case["expected_chunk_ids"])
            retrieved_set = set(retrieved_ids)

            # Missing expected evidence is a retrieval problem.
            missing_ids = sorted(expected_ids - retrieved_set)
            coverage = (
                len(expected_ids & retrieved_set) / len(expected_ids)
                if expected_ids else None
            )

            if not documents:
                raise ValueError("No Documents retrieved")

            answer = generate_answer(case["question"], documents)

            # Evaluate the decision separately from retrieval coverage.
            decision_correct = (
                answer.answerable == case["answerable"]
            )

            cited_ids = set(answer.source_ids)
            citations_valid = (
                cited_ids.issubset(retrieved_set)
                and (not answer.answerable or bool(cited_ids))
            )

            row.update({
                "retrieved_ids": retrieved_ids,
                "expected_chunk_ids": sorted(expected_ids),
                "missing_chunk_ids": missing_ids,
                "required_evidence_coverage": coverage,
                "answerability_correct": decision_correct,
                "predicted_answerable": answer.answerable,
                "answer": answer.answer,
                "source_ids": answer.source_ids,
                "citations_valid": citations_valid,
                "error": None,
            })

            coverage_label = (
                f"{coverage:.0%}" if coverage is not None else "N/A"
            )
            print(
                f"{case['id']}: "
                f"evidence={coverage_label} | "
                f"decision={'PASS' if decision_correct else 'FAIL'} | "
                f"citations={'PASS' if citations_valid else 'FAIL'}"
            )

        except Exception as error:
            # Preserve failed cases instead of silently excluding them.
            row["error"] = type(error).__name__
            print(f"{case['id']}: ERROR — {type(error).__name__}")

        row["elapsed_seconds"] = round(
            perf_counter() - started, 2
        )
        results.append(row)

    total = len(results)
    completed = [row for row in results if not row["error"]]

    # Count operational failures as unsuccessful cases in these totals.
    evidence_passes = sum(
        row.get("required_evidence_coverage") == 1.0
        for row in results
    )
    decision_passes = sum(
        row.get("answerability_correct", False)
        for row in results
    )
    citation_passes = sum(
        row.get("citations_valid", False)
        for row in results
    )

    summary = {
        "cases": total,
        "completed": len(completed),
        "errors": total - len(completed),
        "all_required_evidence_cases": evidence_passes,
        "correct_answerability_cases": decision_passes,
        "valid_citation_cases": citation_passes,
    }

    report = {
        "retrieval_mode": args.retrieval,
        "embedding_model": EMBEDDING_MODEL,
        "top_k": args.k,
        "summary": summary,
        "results": results,
    }

    # RESULTS_FILE.write_text(
    #     json.dumps(report, indent=2, ensure_ascii=False),
    #     encoding="utf-8",
    # )


    # Save the report to the mode-specific path defined above.
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nCompleted: {len(completed)}/{total}")
    print(f"All required evidence: {evidence_passes}/{total}")
    print(f"Correct answerability: {decision_passes}/{total}")
    print(f"Valid citations: {citation_passes}/{total}")
    # print(f"Saved: {RESULTS_FILE}")
    print(f"Saved: {results_file.resolve()}")


if __name__ == "__main__":
    main()