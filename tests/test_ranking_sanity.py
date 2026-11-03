import runpy

import pytest

from findex.index import InvertedIndex
from findex.ranking import BM25, TfIdf


@pytest.mark.parametrize("scorer", [BM25(), TfIdf()])
def test_rare_terms_score_above_common_terms(scorer):
    index = InvertedIndex()
    index.add_document(1, "a", ["rare", "common"])
    index.add_document(2, "b", ["common", "other"])
    assert scorer("rare", index["rare"][0], index) > scorer(
        "common", index["common"][0], index
    )


def test_tfidf_tf_weight_has_diminishing_returns():
    index = InvertedIndex()
    for doc_id, tf in enumerate([1, 2, 3], start=1):
        index.add_document(doc_id, str(doc_id), ["term"] * tf)
    scores = [TfIdf()("term", p, index) for p in index["term"]]
    assert scores[2] > scores[1] > scores[0]
    assert scores[2] - scores[1] < scores[1] - scores[0]


def test_bm25_sanity_checks():
    module = runpy.run_path("scripts/ranking_sanity.py")
    short, long, tf1, tf19, tf20 = module["controlled_bm25"]()
    assert short > long
    assert tf20 - tf19 < tf19 - tf1
