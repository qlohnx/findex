import argparse
import itertools
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


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


class InvertedIndex:
    """Обернений індекс для швидкого пошуку документів за термінами."""

    def __init__(self, positions: bool = False) -> None:
        self.positions = positions
        self.index: dict[str, list[Posting]] = defaultdict(list)
        self.doc_meta: dict[int, DocMeta] = {}
        self.doc_lengths: dict[int, int] = {}

    def add_document(
        self, doc_id: int, path: str, tokens: list[str], title: str = ""
    ) -> None:
        """Додає один документ до індексу.

        Постінги лишаються відсортованими за doc_id, якщо doc_id зростають
        (так і є при потоковій побудові).
        """
        self.doc_meta[doc_id] = DocMeta(path=path, title=title)
        self.doc_lengths[doc_id] = len(tokens)

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
            doc_id, f"{source}#{doc_id}", list(tokenize(text)), title=title
        )
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
