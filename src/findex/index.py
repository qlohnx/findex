import argparse
import functools
import itertools
import logging
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Posting:
    """Запис про одне входження терміна в документ."""

    doc_id: int
    tf: int  # Term Frequency (скільки разів слово зустрілося в документі)
    positions: tuple[int, ...] = ()  # зміщення токенів (лише з positions=True)


@dataclass(frozen=True, slots=True)
class DocMeta:
    """Метадані документа для відображення у результатах пошуку."""

    path: str
    title: str = ""


class Index(Mapping[str, list[Posting]]):
    """Обернений індекс для швидкого пошуку документів за термінами."""

    def __init__(self, positions: bool = False) -> None:
        self.positions = positions
        self.index: dict[str, list[Posting]] = defaultdict(list)
        self.doc_meta: dict[int, DocMeta] = {}
        self.doc_lengths: dict[int, int] = {}
        self.doc_texts: dict[int, str] = {}
        self._closed = False

    def add_document(
        self,
        doc_id: int,
        path: str,
        tokens: list[str],
        title: str = "",
        text: str | None = None,
    ) -> None:
        """Додає один документ до індексу.

        Постінги лишаються відсортованими за doc_id, якщо doc_id зростають
        (так і є при потоковій побудові).
        """
        if doc_id in self.doc_meta:
            raise ValueError(f"duplicate document id: {doc_id}")
        if self.doc_meta and doc_id <= next(reversed(self.doc_meta)):
            raise ValueError("document ids must increase to keep postings sorted")
        self.__dict__.pop("avg_doc_length", None)
        self.doc_meta[doc_id] = DocMeta(path=path, title=title)
        self.doc_lengths[doc_id] = len(tokens)
        if text is not None:
            self.doc_texts[doc_id] = text

        if self.positions:
            where: dict[str, list[int]] = defaultdict(list)
            for offset, term in enumerate(tokens):
                where[term].append(offset)
            for term, offsets in where.items():
                self.index[term].append(
                    Posting(doc_id=doc_id, tf=len(offsets), positions=tuple(offsets))
                )
        else:
            for term, tf in Counter(tokens).items():
                self.index[term].append(Posting(doc_id=doc_id, tf=tf))

    def get_postings(self, term: str) -> list[Posting]:
        """Повертає список постінгів для терміна."""
        return self.index.get(term, [])

    def __len__(self) -> int:
        """Повертає кількість унікальних слів в індексі."""
        return len(self.index)

    def __contains__(self, term: object) -> bool:
        return term in self.index

    def __getitem__(self, term: str) -> list[Posting]:
        if term not in self.index:
            raise KeyError(term)
        return self.index[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self.index)

    def __repr__(self) -> str:
        return f"Index(terms={len(self):,}, docs={self.num_docs:,})"

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    @functools.cached_property
    def avg_doc_length(self) -> float:
        return sum(self.doc_lengths.values()) / self.num_docs if self.num_docs else 0.0

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        return len(self.index.get(term, ()))

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.close()
        return False


class InvertedIndex(Index):
    """Backward-compatible name retained from Lab 2."""


def build_index(
    pages: Iterable[tuple[str, str]],
    source: str = "corpus",
    positions: bool = False,
) -> InvertedIndex:
    """Один прохід по потоку (title, text); у пам'яті лише індекс і поточний документ."""
    from findex.tokenize import tokenize

    index = InvertedIndex(positions=positions)
    for doc_id, (title, text) in enumerate(pages, start=1):
        index.add_document(
            doc_id, f"{source}#{doc_id}", list(tokenize(text)), title=title, text=text
        )
    log.info("indexed %d documents, %d terms", index.num_docs, len(index))
    return index


def main() -> None:
    # Важливо: при `python -m findex.index` цей файл виконується як __main__,
    # і pickle запам'ятав би класи як __main__.InvertedIndex (файл не завантажився б
    # з іншого модуля). Тому будуємо через імпортований findex.index.
    from findex.index import build_index as build
    from findex.stats import iter_pages
    from findex.store import save_index

    parser = argparse.ArgumentParser(description="Build and save the inverted index.")
    parser.add_argument("data_dir", type=Path, help="Папка з *.xml.bz2")
    parser.add_argument("--out", type=Path, required=True, help="Куди зберегти індекс")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--positions", action="store_true", help="Зберігати позиції токенів"
    )
    parser.add_argument(
        "--format",
        choices=["pickle", "json"],
        default=None,
        help="Формат файлу (за замовчуванням за розширенням: .json -> json, інакше pickle)",
    )
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*.xml.bz2"))
    if not files:
        raise SystemExit(f"Не знайдено *.xml.bz2 у {args.data_dir}")
    pages = iter_pages(files[0])
    if args.limit is not None:
        pages = itertools.islice(pages, args.limit)

    tracemalloc.start()
    t0 = time.perf_counter()
    index = build(pages, source=files[0].name, positions=args.positions)
    build_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0 = time.perf_counter()
    save_index(index, args.out, fmt=args.format)
    save_s = time.perf_counter() - t0

    print(f"Documents: {len(index.doc_meta):,}   Terms: {len(index):,}")
    print(f"Build time: {build_s:.2f} s   Peak memory: {peak / 1024 / 1024:.1f} MB")
    print(
        f"Saved to {args.out}: {args.out.stat().st_size / 1024 / 1024:.1f} MB "
        f"in {save_s:.2f} s"
    )


if __name__ == "__main__":
    main()
