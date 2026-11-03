from findex.index import InvertedIndex
from findex.search import BooleanSearch


def test_boolean_search_operations():
    index = InvertedIndex()
    index.add_document(1, "doc1.txt", ["python", "code", "fast"])
    index.add_document(2, "doc2.txt", ["python", "search"])
    index.add_document(3, "doc3.txt", ["java", "code"])

    searcher = BooleanSearch(index)

    # 1. Тест AND: "python" AND "code" -> тільки doc 1
    res_and = searcher.search_and("python", "code")
    assert len(res_and) == 1
    assert res_and[0].doc_id == 1

    # 2. Тест OR: "python" OR "java" -> doc 1, doc 2, doc 3
    res_or = searcher.search_or("python", "java")
    assert len(res_or) == 3
    assert [p.doc_id for p in res_or] == [1, 2, 3]

    # 3. Тест NOT: "python" AND NOT "fast" -> тільки doc 2
    res_not = searcher.search_not("python", "fast")
    assert len(res_not) == 1
    assert res_not[0].doc_id == 2
