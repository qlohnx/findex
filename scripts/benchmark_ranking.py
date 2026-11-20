"""Compare scalar and NumPy BM25 ranking on the same loaded index.

Usage: uv run python scripts/benchmark_ranking.py data/index.json --repeats 100
"""

import argparse
import math
import statistics
import time
from pathlib import Path

from findex.query import parse
from findex.ranking import BM25, search
from findex.store import open_index
from findex.vectorized import view_for


def measure(query: str, index, repeats: int, vectorized: bool) -> float:
    batches: list[float] = []
    for _ in range(3):
        started = time.perf_counter()
        for _ in range(repeats):
            search(query, index, BM25(), k=10, vectorized=vectorized)
        batches.append((time.perf_counter() - started) * 1000 / repeats)
    return statistics.median(batches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    with open_index(args.index) as index:
        rare = next(
            term
            for term in sorted(index)
            if term.isalpha()
            and 7 <= len(term) <= 15
            and len(set(term)) >= 4
            and index.df(term) == 1
        )
        queries = {"common": "the", "rare": rare, "three-term": "the AND of AND in"}
        print(f"documents={index.num_docs} terms={len(index)} rare_term={rare}")
        print(
            "| Query | Docs matched | Scalar ms | NumPy ms | Speedup | First conversion ms |"
        )
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for label, query in queries.items():
            scalar_results = search(query, index, BM25(), vectorized=False)
            matches = len(parse(query).evaluate(index))
            started = time.perf_counter()
            numpy_results = search(query, index, BM25(), vectorized=True)
            conversion_ms = (time.perf_counter() - started) * 1000
            assert [r.doc_id for r in scalar_results] == [
                r.doc_id for r in numpy_results
            ]
            assert all(
                math.isclose(a.score, b.score, rel_tol=1e-10, abs_tol=1e-10)
                for a, b in zip(scalar_results, numpy_results, strict=True)
            )
            scalar_ms = measure(query, index, args.repeats, False)
            numpy_ms = measure(query, index, args.repeats, True)
            print(
                f"| {label} (`{query}`) | {matches} | {scalar_ms:.3f} | {numpy_ms:.3f} | "
                f"{scalar_ms / numpy_ms:.2f}x | {conversion_ms:.3f} |"
            )
        view = view_for(index)
        array_bytes = (
            view.doc_ids.nbytes
            + view.doc_lengths.nbytes
            + sum(ids.nbytes + tfs.nbytes for ids, tfs in view.posting_arrays.values())
        )
        print(f"cached_numpy_array_bytes={array_bytes}")


if __name__ == "__main__":
    main()
