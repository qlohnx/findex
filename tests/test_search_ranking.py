from findex.index import InvertedIndex
from findex.search import BM25, TfIdf, search


def test_search_module_exports_ranking_api():
    index = InvertedIndex()
    index.add_document(1, "a.txt", ["python"], "A", "Python")
    assert search("python", index, TfIdf())[0].doc_id == 1
    assert search(index, "python", BM25())[0].doc_id == 1
