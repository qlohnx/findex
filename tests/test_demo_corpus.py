"""The packaged demo corpus must remain searchable in the container image."""

from pathlib import Path

from findex.corpus import iter_local_pages
from findex.index import build_index
from findex.ranking import search


def test_demo_corpus_is_searchable() -> None:
    corpus = Path(__file__).resolve().parents[1] / "demo" / "corpus"
    index = build_index(iter_local_pages(corpus), positions=True)
    assert index.num_docs == 8
    assert len({meta.title for meta in index.doc_meta.values()}) == 8
    assert len(search("python", index, k=8)) >= 4
    assert search('"inverted index"', index)[0].title == "02 Inverted index"
