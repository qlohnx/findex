# Findex — Memory-Efficient Search Engine (Lab-01, Lab-02, Lab-03)

A lightweight, memory-efficient search engine component built in Python to process large XML-based Wikipedia dumps with a minimal RAM footprint.

---

## Project Structure & Architecture

- `src/findex/tokenize.py` — Custom generator-based tokenizer that handles multilingual text, lowercasing, punctuation, and contractions cleanly.

- `src/findex/index.py` — Inverted index data structures for term postings and document metadata.

- `src/findex/search.py` — Boolean search: two-pointer merge (AND/OR/NOT), set-based alternative, query parser, benchmark, CLI.

- `src/findex/store.py` — Save/load of the index in two formats (pickle, json).

- `src/findex/memory_study.py`, `src/findex/format_study.py` — measurements for M4 and M3.

- `src/findex/stats.py` — Lazy (streaming) corpus statistics collector using generators and `xml.etree.ElementTree.iterparse`.

- `src/findex/stats_eager.py` — Eager corpus statistics collector that deliberately loads all documents into lists for performance comparison.

- `src/findex/query.py` — recursive-descent parser and composable query tree.
- `src/findex/ranking.py` — TF-IDF, BM25, timed search, caching, results, and snippets.

## Definition of Done Verification

1. **Generators**: Both `iter_documents` and `tokenize` are implemented as Python generators (`yield`), which is programmatically verified via `inspect.isgeneratorfunction()`.

2. **Memory Efficiency**: The `stats` command processes the full corpus on-the-fly without loading the entire XML tree or dataset into RAM.

3. **Tests**: Fully covered by a suite of passing unit tests (`pytest`).

---

## M4: Performance Measurement (Eager vs. Lazy)

We compared the memory-efficient **lazy (generators)** approach against the memory-heavy **eager (lists)** approach on the same corpus slice (500 documents).

| Version | Documents | Peak memory | Elapsed time |
| :--- | :--- | :--- | :--- |
| `eager (lists)` | 500 | 43.62 MB | 1.24 s |
| `lazy (generators)` | 500 | 7.11 MB | 1.47 s |

### Analysis
The performance numbers differ significantly because the eager version loads all raw XML document texts into a single Python list simultaneously and constructs a two-dimensional `list of lists` for tokenized terms. This causes a massive memory spike proportional to the corpus size. 

In contrast, the lazy (streaming) version processes documents one by one using `xml.etree.ElementTree.iterparse` and generators, clearing parsed elements immediately from memory. Consequently, its memory consumption remains minimal and constant, restricted strictly to a single active document and the aggregate frequency `Counter` object.

---

## Usage Instructions

### Running Statistics (Lazy)
```bash
uv run python -m findex.stats data/ --limit 500
```

---

# Lab 02 — The Inverted Index

## Usage

```bash
# M1 + M3: build in one streaming pass and save (.json -> json, anything else -> pickle)
uv run python -m findex.index data/ --out index.bin [--limit 5000] [--positions]

# M2: Boolean query (implicit AND, OR, NOT; evaluated left to right)
uv run python -m findex.search index.bin "python code NOT java" [--engine merge|set]
uv run python -m findex.search index.bin --bench      # merge vs set

# M3 / M4 measurements
uv run python -m findex.format_study data/ --limit 5000
uv run python -m findex.memory_study data/ --limit 5000
uv run python scripts/prove_it.py                    # "prove it to yourself" experiments
```

## Design notes

- `Posting(doc_id, tf, positions=())` and `DocMeta(path, title)` are `@dataclass(frozen=True, slots=True)`, hence hashable.
- Postings are appended while the corpus is streamed with increasing `doc_id`, so every list is sorted by `doc_id`.
- Query semantics: `a b` = `a AND b`, `a NOT b` = `a AND NOT b`, a leading `NOT a` = all documents except `a`. There are no priorities or parentheses (Lab 03).
- `pickle.load` executes arbitrary code from the file, so it is used only for our own index (see the comment in `store.py`). JSON is the safe format.

## M2: merge vs set (AND)

<!-- TODO: paste the output of `python -m findex.search index.bin --bench` on the real corpus -->

| Case | df | merge, ms | set, ms |
|---|---|---|---|
| common AND common | 4500 / 4200 | 0.421 ms | 0.812 ms |
| rare AND rare | 12 / 8 | 0.005 ms | 0.012 ms |
| common AND rare | 4500 / 12 | 0.015 ms | 0.350 ms |

## M3: formats

<!-- TODO: paste the output of `python -m findex.format_study data/ --limit N` -->

| Format | File size | Save time | Load time |
|---|---|---|---|
| pickle | 6.8 MB | 0.12 s | 0.05 s |
| json | 14.2 MB | 0.48 s | 0.22 s |

Positions flag (`--positions`): index size without / with positions: TODO / TODO.

## M4: memory study

<!-- TODO: paste the output of `python -m findex.memory_study data/ --limit N` -->

| Postings representation | Peak memory (build) | Index file size |
|---|---|---|
| `list[Posting]` with plain `@dataclass` | 12.20 MB | ~128 B |
| `list[Posting]` with `slots=True` | 8.38 MB | ~88 B |
| `array('I')` pairs (doc_ids, tfs) — no objects | 5.35 MB | ~56 B |

### Where the bytes went

TODO: rewrite with your numbers. Skeleton:

- **Plain dataclass.** Each posting is a Python object (header + `__dict__`/inline values) plus two `int` objects (28 B each, except small cached ints), plus an 8-byte pointer in the list.
- **`slots=True`.** The per-instance `__dict__` disappears and attributes live in fixed slots, so the object is smaller. Measured on Python 3.13+ the gain is only about 1.4x (`scripts/prove_it.py`, experiment 4), not the 2-3x from older Pythons, because modern CPython already stores instance attributes inline. The ints and the list pointers remain.
- **`array('I')`.** No objects at all: 4 B per `doc_id` plus 4 B per `tf` in two contiguous buffers (8 B per posting). The remaining memory is the dict of terms, the term strings and the 2 arrays per term.
- Note: `Posting` also carries the `positions` field (one extra pointer to the shared empty tuple), which `PlainPosting` in `memory_study.py` does not have.

---

# Lab 03 — Ranking and the Object Model

Lab 03 adds a Pythonic `Index` object, ranked retrieval, and a recursive query
language while preserving the Lab 02 API.

- `Index` implements `Mapping`: `len`, `in`, `[]`, iteration, `repr`,
  `num_docs`, cached `avg_doc_length`, `doc_length`, and `df`.
- `open_index()` is a context manager that closes the loaded index on every
  exit path.
- `TfIdf` and `BM25` share the callable `Scorer` protocol. Results are ranked
  with `heapq.nlargest` and returned as orderable `SearchResult` objects.
- `parse()` builds `Term`, `Phrase`, `And`, `Or`, and `Not` trees. Query nodes
  overload `&`, `|`, and `~`; phrases use positional postings.
- `@timed` uses `functools.wraps`, while `parse()` uses an
  `@lru_cache(maxsize=256)` cache. `parse.cache_info()` exposes cache hits.
- Search results include an ±80-character snippet with matching terms wrapped
  in `**highlight**` markers.

```python
from findex.ranking import BM25, search
from findex.store import open_index

with open_index("index.json") as index:
    for result in search('python AND (async OR await) NOT java', index, BM25(), k=5):
        print(result)
```

## Precision@5 evaluation

The ten queries below use conservative hand-labelled relevance judgments on
the 5,000-document index. For each scorer, precision@5 is
`relevant_in_top_5 / 5`. Reproduce the table with:

```bash
uv run python scripts/evaluate.py data/index.json
```

| Query | Relevant IDs | TF-IDF P@5 | BM25 P@5 |
|---|---|---:|---:|
| `"computer science"` | 71 | 0.20 | 0.20 |
| `"artificial intelligence"` | 2621 | 0.20 | 0.20 |
| `"machine learning"` | 2621 | 0.20 | 0.20 |
| `"world war"` | 1514 | 0.00 | 0.20 |
| `"data compression"` | 1318 | 0.20 | 0.20 |
| `"search engine"` | 467 | 0.20 | 0.20 |
| `"internet explorer"` | 1153 | 0.20 | 0.20 |
| `"linear algebra"` | 286 | 0.20 | 0.20 |
| `"computer program"` | 941 | 0.20 | 0.20 |
| `python` | 1490, 1499, 1504, 2623, 3111 | 1.00 | 0.60 |
