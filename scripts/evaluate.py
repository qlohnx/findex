"""Print a reproducible precision@5 comparison for the Lab 03 corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from findex.ranking import BM25, TfIdf, search
from findex.store import open_index

JUDGMENTS: dict[str, set[int]] = {
    '"computer science"': {71},
    '"artificial intelligence"': {2621},
    '"machine learning"': {2621},
    '"world war"': {1514},
    '"data compression"': {1318},
    '"search engine"': {467},
    '"internet explorer"': {1153},
    '"linear algebra"': {286},
    '"computer program"': {941},
    "python": {2623, 1490, 1499, 1504, 3111},
}


def precision_at_5(doc_ids: list[int], relevant: set[int]) -> float:
    """Return the fraction of relevant documents in the first five results."""
    return len(set(doc_ids[:5]) & relevant) / 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    args = parser.parse_args()

    print("| Query | Relevant IDs | TF-IDF P@5 | BM25 P@5 |")
    print("|---|---|---:|---:|")
    with open_index(args.index) as index:
        for query, relevant in JUDGMENTS.items():
            tfidf = search(query, index, TfIdf(), k=5)
            bm25 = search(query, index, BM25(), k=5)
            tfidf_ids = [result.doc_id for result in tfidf]
            bm25_ids = [result.doc_id for result in bm25]
            print(
                f"| `{query}` | {', '.join(map(str, sorted(relevant)))} | "
                f"{precision_at_5(tfidf_ids, relevant):.2f} | "
                f"{precision_at_5(bm25_ids, relevant):.2f} |"
            )


if __name__ == "__main__":
    main()
