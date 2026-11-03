from findex.index import InvertedIndex
from findex.store import open_index, save_index


def test_open_index_closes_loaded_resource(tmp_path):
    source = InvertedIndex()
    source.add_document(1, "a.txt", ["python"], "A", "Python")
    path = tmp_path / "index.json"
    save_index(source, path)

    with open_index(path) as index:
        assert index.num_docs == 1
        loaded = index
    assert loaded._closed is True
