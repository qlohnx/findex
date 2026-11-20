# Lab 08: baseline profiling and optimization hypotheses

This page records the measurements **before changing the index or ranking code**.
The source is the local Simple Wikipedia XML dump, indexed with `--limit 5000
--positions`. This is the largest existing project index, **not the entire dump**;
full-corpus results must not be inferred from these numbers. All measurements
were made on Windows with Python 3.14. The benchmark index has 5,000 documents
and 200,303 terms.

## Baseline

| Workflow | cProfile cumulative time | Main cost |
| --- | ---: | --- |
| CLI cold search (`the AND of AND in`) | 17.219 s | JSON load 16.344 s; validation 4.547 s is part of load |
| CLI build and save, 5,000 documents | 56.567 s | JSON save 43.286 s; `json.dump` 35.082 s is part of save |
| 100 warm BM25 searches | 2.569 s (25.687 ms/search, instrumented) | 1,011,600 scalar BM25 calls, 1.893 s cumulative |

The cProfile times are **instrumented** and should not be compared directly to
unprofiled wall-clock latency. In the warm-search profile, `Index.df` consumed
0.515 s cumulative, `Index.num_docs` 0.256 s, snippet generation 0.114 s,
Boolean evaluation 0.081 s and `heapq.nlargest` 0.037 s. The top three
actionable costs are JSON serialization on build, JSON deserialization on cold
search, and Python-per-posting BM25 scoring on warm search. The first two are
separate workflows, so their times must not be summed.

Sampling profiles: [build flame graph](lab08-index-before.svg) and
[warm-search flame graph](lab08-search-before.svg). `py-spy` recorded 2,432
and 3,368 samples respectively, with zero sampling errors. The warm-search
sample ran 3,000 searches in 26.822 s (8.941 ms/search), without cProfile.
The CLI build flame graph includes XML parsing, indexing and JSON saving.

Reproduce the instrumented profiles with the same commands and index size
(the output files under `.cache/` are intentionally not versioned):

```bash
python -m cProfile -o .cache/profiles/index.prof -m findex index data/ --out data/profile-index.json --limit 5000 --positions
python -m cProfile -o .cache/profiles/search.prof -m findex search data/index.json "the AND of AND in" --k 5 --json
python scripts/profile_ranking.py data/index.json "the AND of AND in" --repeats 100 --profile-out .cache/profiles/rank.prof
```

The SVGs were recorded with `py-spy record -o <output.svg> -- <command>`.
On this Windows setup, `py-spy` could not inspect the venv Python launcher;
the base Python interpreter plus the project `src` and venv `site-packages`
on `PYTHONPATH` was used for the sampled child process. The application code
and indexed data were unchanged.

Scalene was also run with `--cpu-only` on ranking. It attributes work to the
scoring expression, `Index.df`, and `math.log`, but its percentages include
the one-time JSON load and its Python/native split is therefore only supporting
evidence, not a precise per-query breakdown.

For the build workflow, a separate Scalene run took 56.0 s. It attributed
76.0% native CPU to `_save_json`, versus 5.1% native CPU to
`Index.add_document` and 3.8% to `build_partial`. These are whole-process
sampling percentages; the native label does not mean the JSON operation is
free of Python-level cost. This agrees with cProfile that serialization is
the primary build bottleneck.

| Hotspot | cProfile cumulative time | Scalene Python / native / system CPU share |
| --- | ---: | ---: |
| Build: `_save_json` | 43.286 s | 0.22% / 76.03% / 5.56% |
| Build: `Index.add_document` | 6.285 s | 0.38% / 5.11% / 0.72% |
| Warm search: `BM25.__call__` | 1.893 s across 100 searches | 0.78% / 8.77% / 0.13% |

The Scalene percentages use **different whole-process runs** for build and
warm search; they are not additive. The search run also loaded JSON once,
which accounts for most of its native CPU samples. In BM25, the repeated
Python calls and property lookups are still actionable even though native
`math.log` work appears inside the function.

## Hypotheses recorded before implementation

1. Keep the on-disk format and public scalar scorer API, but score postings in
   NumPy arrays (`int32` doc IDs, term frequencies and document lengths).
   This should reduce Python call and property-lookup overhead for common-term
   and three-term warm queries. Conversion and cache memory may offset the
   gain for rare terms.
2. Use `np.argpartition` for top-k selection with stable tie handling. This
   should help when many documents match, but the measured heap cost is small;
   it may offer little or no improvement by itself.
3. JSON read/write dominates cold and build workflows. A future compact or
   binary format could help substantially, but changing persistence requires
   compatibility and size measurements first. It is not assumed to be solved
   by vectorizing ranking.

The next benchmark must measure rare, common and three-term warm queries,
record conversion time and memory, and compare result order against the scalar
implementation before claiming a speedup.

## Outcome after NumPy ranking

The `scripts/benchmark_ranking.py` run loaded the same 5,000-document index
once. It warmed each path, then took the median of three batches of 100
searches per query. The scalar and NumPy paths returned identical document
orders and matching scores within floating-point tolerance.

| Query | Docs matched | Scalar ms | NumPy ms | Speedup | First vector conversion ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `the` (common) | 3,666 | 3.368 | 0.827 | 4.07× | 1.772 |
| `aaaacattle` (rare) | 1 | 0.122 | 0.142 | 0.86× | 0.212 |
| `the AND of AND in` | 3,372 | 9.554 | 2.116 | 4.52× | 2.783 |

The arrays cached for these three terms, including document IDs and lengths,
occupy 126,832 bytes of raw NumPy buffers. The normal `search` entry point
uses the scalar path for queries with fewer than 64 matches, avoiding the
rare-term regression; the table deliberately forces NumPy to expose its cost.
The common and three-term gains support the hypothesis about Python-per-posting
overhead. They do **not** improve the separate JSON load/save bottlenecks.

The benchmark fixture in `tests/test_benchmark_lab08.py` checks search and
build behavior with `pytest-benchmark`. Its timings are not comparable to the
5,000-document table because it uses a 200-document in-memory fixture.
