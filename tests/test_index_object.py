import pytest

from findex.index import InvertedIndex


def test_index_protocol_and_properties():
    index = InvertedIndex(positions=True)
    index.add_document(1, "a.txt", ["rare", "common", "phrase"], "A")
    index.add_document(2, "b.txt", ["common"] * 20, "B")

    assert len(index) == 3
    assert "rare" in index
    assert index["rare"][0].doc_id == 1
    with pytest.raises(KeyError):
        index["missing"]
    assert index.num_docs == 2
    assert index.avg_doc_length == 11.5
    assert index.df("common") == 2
    assert repr(index) == "Index(terms=3, docs=2)"


def test_average_length_updates_after_adding_document():
    index = InvertedIndex()
    index.add_document(1, "a", ["a"])
    assert index.avg_doc_length == 1
    index.add_document(2, "b", ["b"] * 3)
    assert index.avg_doc_length == 2


@pytest.mark.parametrize("doc_id", [0, 1])
def test_duplicate_or_decreasing_document_ids_are_rejected(doc_id):
    index = InvertedIndex()
    index.add_document(1, "a", ["a"])
    with pytest.raises(ValueError):
        index.add_document(doc_id, "b", ["b"])
    assert index.num_docs == 1
    assert "b" not in index
