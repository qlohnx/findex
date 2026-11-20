"""Equivalence and cache tests for the NumPy ranking path."""

import pytest

from findex.index import InvertedIndex
from findex.ranking import BM25, TfIdf, search


@pytest.fixture
def varied_index() -> InvertedIndex:
    index = InvertedIndex(positions=True)
    for doc_id in range(1, 101):
        tokens = ["common"] * (doc_id % 7 + 1)
        tokens += ["rare"] * (doc_id % 3 + 1) if doc_id % 11 == 0 else []
        tokens += ["other"] * (doc_id % 5)
        index.add_document(doc_id, f"{doc_id}.txt", tokens, text=" ".join(tokens))
    return index


@pytest.mark.parametrize("scorer", [BM25(), TfIdf()])
@pytest.mark.parametrize(
    "query", ["common", "rare", "common OR rare", "common AND rare"]
)
@pytest.mark.parametrize("k", [0, 1, 5, 1000])
def test_vectorized_matches_scalar(varied_index, scorer, query, k):
    scalar = search(query, varied_index, scorer, k=k, vectorized=False)
    vector = search(query, varied_index, scorer, k=k, vectorized=True)
    assert [r.doc_id for r in vector] == [r.doc_id for r in scalar]
    assert [r.score for r in vector] == pytest.approx([r.score for r in scalar])
    assert [r.snippet for r in vector] == [r.snippet for r in scalar]


def test_boundary_ties_preserve_original_match_order():
    index = InvertedIndex()
    for doc_id in [1, 4, 8, 12, 16, 20, 24, 28, 32, 36]:
        index.add_document(doc_id, f"{doc_id}.txt", ["same"])
    scalar = search("same", index, k=3, vectorized=False)
    vector = search("same", index, k=3, vectorized=True)
    assert [r.doc_id for r in vector] == [r.doc_id for r in scalar]


def test_cached_arrays_are_invalidated_when_index_changes(varied_index):
    search("common", varied_index, vectorized=True)
    varied_index.add_document(101, "new.txt", ["common"] * 50)
    scalar = search("common", varied_index, vectorized=False)
    vector = search("common", varied_index, vectorized=True)
    assert [r.doc_id for r in vector] == [r.doc_id for r in scalar]
    assert vector[0].doc_id == 101


def test_large_document_ids_fall_back_to_scalar():
    index = InvertedIndex()
    index.add_document(2**31 + 1, "large.txt", ["word"])
    assert search("word", index, vectorized=True)[0].doc_id == 2**31 + 1
