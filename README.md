# Findex — Memory-Efficient Search Engine (Lab-01)

A lightweight, memory-efficient search engine component built in Python to process large XML-based Wikipedia dumps with a minimal RAM footprint.

---

## Project Structure & Architecture

- `src/findex/tokenize.py` — Custom generator-based tokenizer that handles multilingual text, lowercasing, punctuation, and contractions cleanly.

- `src/findex/index.py` — Inverted index data structures for term postings and document metadata.

- `src/findex/stats.py` — Lazy (streaming) corpus statistics collector using generators and `xml.etree.ElementTree.iterparse`.

- `src/findex/stats_eager.py` — Eager corpus statistics collector that deliberately loads all documents into lists for performance comparison.

---

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