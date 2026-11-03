import json
import pickle
from contextlib import contextmanager
from pathlib import Path

from findex.index import DocMeta, Index, InvertedIndex, Posting

FORMATS = ("pickle", "json")


def _detect_format(path: Path, fmt: str | None) -> str:
    if fmt is None:
        return "json" if path.suffix == ".json" else "pickle"
    if fmt not in FORMATS:
        raise ValueError(f"Невідомий формат {fmt!r}; доступні: {FORMATS}")
    return fmt


def save_index(
    index: InvertedIndex, path: Path | str, fmt: str | None = None
) -> None:
    """Зберігає InvertedIndex у файл (pickle або json)."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if _detect_format(file_path, fmt) == "json":
        _save_json(index, file_path)
        return

    with open(file_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path: Path | str, fmt: str | None = None) -> InvertedIndex:
    """Завантажує InvertedIndex з файлу (pickle або json)."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Файл індексу не знайдено за шляхом: {file_path}")

    if _detect_format(file_path, fmt) == "json":
        return _load_json(file_path)

    # УВАГА (безпека): pickle.load виконує довільний код, який закодовано у файлі
    # (через __reduce__ об'єкт може змусити викликати, наприклад, os.system).
    # Тому завантажувати можна ЛИШЕ власний індекс, створений нами самими.
    # Ніколи не робіть pickle.load для файлу, отриманого з мережі чи від інших людей.
    with open(file_path, "rb") as f:
        index = pickle.load(f)

    if not isinstance(index, Index):
        raise TypeError("Завантажений об'єкт не є екземпляром InvertedIndex")

    return index


@contextmanager
def open_index(path: Path | str, fmt: str | None = None):
    """Load an index and guarantee cleanup when leaving the context."""
    index = load_index(path, fmt=fmt)
    try:
        yield index
    finally:
        index.close()


def _save_json(index: InvertedIndex, path: Path) -> None:
    # json: int-ключі стають рядками, tuple -> list; тому структура явна,
    # а при завантаженні типи відновлюються вручну.
    data = {
        "positions": index.positions,
        "doc_meta": {
            str(i): {"path": m.path, "title": m.title} for i, m in index.doc_meta.items()
        },
        "doc_lengths": {str(i): n for i, n in index.doc_lengths.items()},
        "doc_texts": {str(i): text for i, text in index.doc_texts.items()},
        "index": {
            term: [
                [p.doc_id, p.tf, list(p.positions)] if index.positions else [p.doc_id, p.tf]
                for p in postings
            ]
            for term, postings in index.index.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _load_json(path: Path) -> InvertedIndex:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    index = InvertedIndex(positions=data["positions"])
    index.doc_meta = {
        int(i): DocMeta(path=m["path"], title=m["title"])
        for i, m in data["doc_meta"].items()
    }
    index.doc_lengths = {int(i): n for i, n in data["doc_lengths"].items()}
    index.doc_texts = {int(i): text for i, text in data.get("doc_texts", {}).items()}
    for term, postings in data["index"].items():
        index.index[term] = [
            Posting(p[0], p[1], tuple(p[2])) if len(p) == 3 else Posting(p[0], p[1])
            for p in postings
        ]
    return index
