from findex.index import InvertedIndex
from findex.ranking import BM25, TfIdf, search


def make_index() -> InvertedIndex:
    index = InvertedIndex(positions=True)
    index.add_document(1, "a.txt", ["rare", "common"], "A", "Rare common")
    index.add_document(2, "b.txt", ["common"] * 20, "B", "common " * 20)
    return index


def test_ranking_and_snippet():
    index = make_index()
    result = search("rare common", index, TfIdf(), k=2)[0]
    assert result.doc_id == 1
    assert "**Rare**" in result.snippet
    assert search(index, "rare", BM25())[0].doc_id == 1


def test_search_observes_index_changes():
    index = make_index()
    assert [r.doc_id for r in search("rare", index)] == [1]
    index.add_document(3, "c.txt", ["rare"])
    assert {r.doc_id for r in search("rare", index)} == {1, 3}


def test_search_accepts_query_tree(tiny_index):
    from findex.query import Term

    assert {r.doc_id for r in search(Term("python"), tiny_index)} == {1, 3}
