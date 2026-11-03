import json
import logging
import pickle
from collections.abc import Generator
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from typing import Final, Literal, cast

from findex.index import DocMeta, Index, InvertedIndex, Posting

type Format = Literal["pickle", "json"]
type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)
FORMATS: Final[tuple[Format, ...]] = ("pickle", "json")
log = logging.getLogger(__name__)


def _detect_format(path: Path, fmt: Format | None) -> Format:
    if fmt is None:
        return "json" if path.suffix == ".json" else "pickle"
    if fmt not in FORMATS:
        raise ValueError(f"Невідомий формат {fmt!r}; доступні: {FORMATS}")
    return fmt


def save_index(index: Index, path: Path | str, fmt: Format | None = None) -> None:
    """Зберігає InvertedIndex у файл (pickle або json)."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    log.debug("saving index to %s", file_path)

    if _detect_format(file_path, fmt) == "json":
        _save_json(index, file_path)
        return

    with open(file_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path: Path | str, fmt: Format | None = None) -> Index:
    """Завантажує InvertedIndex з файлу (pickle або json)."""
    file_path = Path(path)
    log.debug("loading index from %s", file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Файл індексу не знайдено за шляхом: {file_path}")

    if _detect_format(file_path, fmt) == "json":
        return _load_json(file_path)

    # УВАГА (безпека): pickle.load виконує довільний код, який закодовано у файлі
    # (через __reduce__ об'єкт може змусити викликати, наприклад, os.system).
    # Тому завантажувати можна ЛИШЕ власний індекс, створений нами самими.
    # Ніколи не робіть pickle.load для файлу, отриманого з мережі чи від інших людей.
    with open(file_path, "rb") as f:
        index: object = pickle.load(f)

    if not isinstance(index, Index):
        raise TypeError("Завантажений об'єкт не є екземпляром InvertedIndex")

    return index


@contextmanager
def open_index(path: Path | str, fmt: Format | None = None) -> Generator[Index]:
    """Load an index and guarantee cleanup when leaving the context."""
    index = load_index(path, fmt=fmt)
    try:
        yield index
    finally:
        index.close()


def _save_json(index: Index, path: Path) -> None:
    # json: int-ключі стають рядками, tuple -> list; тому структура явна,
    # а при завантаженні типи відновлюються вручну.
    data = {
        "positions": index.positions,
        "doc_meta": {
            str(i): {"path": m.path, "title": m.title}
            for i, m in index.doc_meta.items()
        },
        "doc_lengths": {str(i): n for i, n in index.doc_lengths.items()},
        "doc_texts": {str(i): text for i, text in index.doc_texts.items()},
        "index": {
            term: [
                [p.doc_id, p.tf, list(p.positions)]
                if index.positions
                else [p.doc_id, p.tf]
                for p in postings
            ]
            for term, postings in index.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _object(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError("invalid index: expected a JSON object")
    return value


def _integer(value: JsonValue) -> int:
    if type(value) is not int:
        raise ValueError("invalid index: expected an integer")
    return value


def _string(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid index: expected a string")
    return value


def _load_json(path: Path) -> InvertedIndex:
    with open(path, encoding="utf-8") as f:
        # The stdlib returns Any; JSON has this recursive shape. Fields are
        # validated below before they enter the typed Index API.
        data = _object(cast(JsonValue, json.load(f)))

    positions = data["positions"]
    if not isinstance(positions, bool):
        raise ValueError("invalid index: positions must be a boolean")
    index = InvertedIndex(positions=positions)
    index.doc_meta = {
        int(i): DocMeta(
            path=_string(_object(m)["path"]), title=_string(_object(m)["title"])
        )
        for i, m in _object(data["doc_meta"]).items()
    }
    index.doc_lengths = {
        int(i): _integer(n) for i, n in _object(data["doc_lengths"]).items()
    }
    index.doc_texts = {
        int(i): _string(text) for i, text in _object(data.get("doc_texts", {})).items()
    }
    for term, postings in _object(data["index"]).items():
        if not isinstance(postings, list):
            raise ValueError("invalid index: postings must be a list")
        for p in postings:
            if not isinstance(p, list) or len(p) not in (2, 3):
                raise ValueError("invalid index: malformed posting")
            offsets: tuple[int, ...] = ()
            if len(p) == 3:
                raw_offsets = p[2]
                if not isinstance(raw_offsets, list):
                    raise ValueError("invalid index: positions must be a list")
                offsets = tuple(_integer(offset) for offset in raw_offsets)
            index.index[term].append(Posting(_integer(p[0]), _integer(p[1]), offsets))
    _validate_index(index)
    return index


def _validate_index(index: Index) -> None:
    """Reject inconsistent JSON data before statistics or scorers consume it."""
    if index.doc_meta.keys() != index.doc_lengths.keys():
        raise ValueError("invalid index: document metadata and lengths disagree")
    if any(length < 0 for length in index.doc_lengths.values()):
        raise ValueError("invalid index: document lengths must be nonnegative")
    for postings in index.values():
        previous_id: int | None = None
        for posting in postings:
            length = index.doc_lengths.get(posting.doc_id)
            if length is None:
                raise ValueError("invalid index: posting refers to an unknown document")
            if posting.tf <= 0 or posting.tf > length:
                raise ValueError(
                    "invalid index: term frequency must be positive and fit the document"
                )
            if previous_id is not None and posting.doc_id <= previous_id:
                raise ValueError(
                    "invalid index: postings must have increasing document ids"
                )
            previous_id = posting.doc_id
            offsets = posting.positions
            if index.positions:
                if len(offsets) != posting.tf or any(
                    offset < 0 or offset >= length for offset in offsets
                ):
                    raise ValueError(
                        "invalid index: positions do not match frequency or document length"
                    )
                if any(left >= right for left, right in pairwise(offsets)):
                    raise ValueError(
                        "invalid index: positions must be strictly increasing"
                    )
            elif offsets:
                raise ValueError(
                    "invalid index: unexpected positions in a non-positional index"
                )
