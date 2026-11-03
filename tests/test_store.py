from pathlib import Path
from findex.index import InvertedIndex, Posting
from findex.store import save_index, load_index
from findex.search import BooleanSearch


def test_index_save_and_load(tmp_path: Path):
    # 1. Створюємо індекс і заповнюємо його
    index = InvertedIndex()
    index.add_document(doc_id=1, path="data/doc1.txt", tokens=["python", "search"])
    index.add_document(doc_id=2, path="data/doc2.txt", tokens=["python", "fast"])

    # 2. Зберігаємо у тимчасовий файл (tmp_path надається pytest)
    save_path = tmp_path / "index.pkl"
    save_index(index, save_path)
    assert save_path.exists()

    # 3. Завантажуємо індекс назад
    loaded_index = load_index(save_path)

    # 4. Перевіряємо цілісність даних
    assert len(loaded_index) == len(index)
    assert loaded_index.doc_meta[1].path == "data/doc1.txt"
    assert loaded_index.get_postings("python") == [
        Posting(doc_id=1, tf=1),
        Posting(doc_id=2, tf=1),
    ]

    # 5. Перевіряємо, що пошук працює по завантаженому індексу
    searcher = BooleanSearch(loaded_index)
    results = searcher.search_and("python", "fast")
    assert len(results) == 1
    assert results[0].doc_id == 2
