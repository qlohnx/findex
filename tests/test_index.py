from findex.index import InvertedIndex, Posting


def test_inverted_index_add_document():
    index = InvertedIndex()

    # Додаємо два документи
    index.add_document(
        doc_id=1, path="data/doc1.txt", tokens=["python", "search", "engine", "python"]
    )
    index.add_document(
        doc_id=2, path="data/doc2.txt", tokens=["python", "data", "structures"]
    )

    # Перевіряємо кількість унікальних слів (словник)
    assert len(index) == 5  # python, search, engine, data, structures

    # Перевіряємо постінги для слова "python"
    postings_python = index.get_postings("python")
    assert len(postings_python) == 2
    assert postings_python[0] == Posting(doc_id=1, tf=2)
    assert postings_python[1] == Posting(doc_id=2, tf=1)

    # Перевіряємо постінги для слова "engine"
    postings_engine = index.get_postings("engine")
    assert len(postings_engine) == 1
    assert postings_engine[0] == Posting(doc_id=1, tf=1)

    # Перевіряємо запит за неіснуючим словом
    assert index.get_postings("java") == []


def test_doc_meta_and_lengths():
    index = InvertedIndex()
    index.add_document(doc_id=1, path="data/doc1.txt", tokens=["hello", "world"])

    assert index.doc_meta[1].path == "data/doc1.txt"
    assert index.doc_lengths[1] == 2
