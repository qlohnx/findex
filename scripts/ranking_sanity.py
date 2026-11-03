"""Demonstrate the three BM25/TF-IDF ranking sanity checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from findex.index import InvertedIndex
from findex.ranking import BM25, TfIdf
from findex.store import open_index


def corpus_rare_vs_common(index) -> tuple[str, str, float, float]:
    terms = sorted(index, key=index.df)
    rare, common = terms[0], terms[-1]
    rare_score = TfIdf()(rare, index[rare][0], index)
    common_score = TfIdf()(common, index[common][0], index)
    return rare, common, rare_score, common_score


def controlled_bm25() -> tuple[float, float, float, float, float]:
    index = InvertedIndex()
    index.add_document(1, "short.txt", ["term", "filler"])
    index.add_document(2, "long.txt", ["term"] + ["filler"] * 99)
    index.add_document(3, "tf1.txt", ["term"] + ["x"] * 19)
    index.add_document(4, "tf19.txt", ["term"] * 19 + ["x"])
    index.add_document(5, "tf20.txt", ["term"] * 20)
    scorer = BM25()
    postings = index["term"]
    scores = {posting.doc_id: scorer("term", posting, index) for posting in postings}
    return scores[1], scores[2], scores[3], scores[4], scores[5]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    args = parser.parse_args()
    with open_index(args.index) as index:
        rare, common, rare_score, common_score = corpus_rare_vs_common(index)
        print(f"rare term: {rare!r} score={rare_score:.4f}")
        print(f"common term: {common!r} score={common_score:.4f}")
    short, long, tf1, tf19, tf20 = controlled_bm25()
    print(f"short one-match score={short:.4f}; long one-match score={long:.4f}")
    print(f"BM25 tf=1: {tf1:.4f}; tf=19: {tf19:.4f}; tf=20: {tf20:.4f}")
    print(f"20th marginal gain: {tf20 - tf19:.4f}")


if __name__ == "__main__":
    main()
