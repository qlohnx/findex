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
