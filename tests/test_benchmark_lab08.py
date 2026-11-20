"""Repeatable small-corpus microbenchmarks; full-corpus timings live in docs."""

import statistics
import time

import pytest

from findex.index import build_index
from findex.ranking import search


@pytest.fixture(scope="module")
def benchmark_index():
    pages = [
        (f"Page {i}", "common alpha beta " * (i % 5 + 1) + f"rare{i}")
        for i in range(200)
    ]
    return build_index(pages)


@pytest.mark.parametrize("query", ["common", "rare123", "common AND alpha AND beta"])
def test_benchmark_search(benchmark, benchmark_index, query):
    # Warm the lazy posting cache before timing a warm-query request.
    search(query, benchmark_index)
    result = benchmark(search, query, benchmark_index)
    assert result


def test_benchmark_build_index(benchmark):
    pages = [(f"Page {i}", "alpha beta gamma " * 3) for i in range(200)]
    result = benchmark(build_index, pages)
    assert result.num_docs == 200


def test_benchmark_vectorized_regression_guard(benchmark):
    """A gross NumPy regression should fail on the same CI machine."""
    pages = [(f"Page {i}", "common alpha beta " * (i % 4 + 1)) for i in range(3000)]
    index = build_index(pages)
    search("common", index, vectorized=True)  # warm the posting arrays
    scalar_batches = []
    for _ in range(3):
        started = time.perf_counter()
        for _ in range(30):
            search("common", index, vectorized=False)
        scalar_batches.append((time.perf_counter() - started) / 30)
    numpy_results = benchmark(search, "common", index, vectorized=True)
    assert numpy_results
    # A lenient relative guard avoids imposing this laptop's absolute times
    # on Linux and Windows CI runners.
    if benchmark.enabled:
        assert benchmark.stats.stats.mean < statistics.median(scalar_batches) * 1.5
