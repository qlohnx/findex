"""M4: дослідження пам'яті — три представлення постінгів.

Той самий корпус будується тричі, змінюється лише спосіб зберігання постінгів:

1. ``list[PlainPosting]`` — звичайний ``@dataclass`` (у кожного екземпляра є ``__dict__``);
2. ``list[Posting]`` — ``findex.index.Posting`` (``frozen=True, slots=True``);
3. ``array('I')`` — два масиви на термін (doc_ids, tfs), жодних об'єктів-постінгів.

Для кожного варіанта вимірюється: пікова пам'ять побудови (tracemalloc),
розмір pickle-файлу, час завантаження.

Запуск:
    uv run python -m findex.memory_study data/ --limit 5000
    uv run python -m findex.memory_study --synthetic 5000
"""

import argparse
import gc
import itertools
import random
import statistics
import tempfile
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from findex.index import DocMeta, InvertedIndex
from findex.stats import iter_documents
from findex.store import load_index, save_index
from findex.tokenize import tokenize

Doc = tuple[int, str, str]  # (doc_id, path, text)


# ---------------------------------------------------------------------------
# Варіант 1: звичайний dataclass (з __dict__ у кожного екземпляра)
# ---------------------------------------------------------------------------
@dataclass
class PlainPosting:
    doc_id: int
    tf: int


class PlainInvertedIndex(InvertedIndex):
    """Те саме, що InvertedIndex, але постінги — звичайні dataclass-об'єкти."""

    def add_document(self, doc_id: int, path: str, tokens: list[str]) -> None:
        self.doc_meta[doc_id] = DocMeta(path=path)
        self.doc_lengths[doc_id] = len(tokens)
        for term, tf in Counter(tokens).items():
            self.index[term].append(PlainPosting(doc_id, tf))


# ---------------------------------------------------------------------------
# Варіант 2: findex.index.InvertedIndex як є (Posting: frozen + slots)
# ---------------------------------------------------------------------------
class SlotsInvertedIndex(InvertedIndex):
    """Ваша реалізація без змін; підклас потрібен лише для підпису в таблиці."""


# ---------------------------------------------------------------------------
# Варіант 3: array('I') — без об'єктів-постінгів
# ---------------------------------------------------------------------------
def _new_pair() -> tuple[array, array]:
    # Має бути функцією верхнього рівня, а не lambda: lambda не серіалізується pickle.
    return (array("I"), array("I"))


class ArrayInvertedIndex(InvertedIndex):
    """Постінги терміна — пара масивів (doc_ids, tfs) по 4 байти на елемент."""

    def __init__(self) -> None:
        super().__init__()
        self.index = defaultdict(_new_pair)  # type: ignore[assignment]

    def add_document(self, doc_id: int, path: str, tokens: list[str]) -> None:
        self.doc_meta[doc_id] = DocMeta(path=path)
        self.doc_lengths[doc_id] = len(tokens)
        for term, tf in Counter(tokens).items():
            doc_ids, tfs = self.index[term]  # type: ignore[misc]
            doc_ids.append(doc_id)
            tfs.append(tf)


VARIANTS: list[tuple[str, type[InvertedIndex]]] = [
    ("`list[Posting]`, plain `@dataclass`", PlainInvertedIndex),
    ("`list[Posting]`, `slots=True` (frozen)", SlotsInvertedIndex),
    ("`array('I')` pairs, no objects", ArrayInvertedIndex),
]


def postings_as_pairs(index: InvertedIndex, term: str) -> list[tuple[int, int]]:
    """Єдиний вигляд постінгів для звірки варіантів між собою."""
    entry = index.index[term]
    if isinstance(entry, tuple):  # array-варіант
        doc_ids, tfs = entry
        return list(zip(doc_ids, tfs, strict=True))
    return [(p.doc_id, p.tf) for p in entry]


# ---------------------------------------------------------------------------
# Джерела документів
# ---------------------------------------------------------------------------
def real_docs(data_dir: Path, limit: int | None) -> Iterator[Doc]:
    files = sorted(data_dir.glob("*.xml.bz2"))
    if not files:
        raise SystemExit(f"Не знайдено *.xml.bz2 у {data_dir}")
    source = iter_documents(files[0])
    if limit is not None:
        source = itertools.islice(source, limit)
    for doc_id, text in enumerate(source, start=1):
        yield doc_id, f"{files[0].name}#{doc_id}", text


def synthetic_docs(n_docs: int, vocab: int = 50_000, seed: int = 42) -> Iterator[Doc]:
    """Корпус із розподілом Ціпфа: лише для перевірки скрипта, не для README."""
    rng = random.Random(seed)
    words = [f"w{i}" for i in range(vocab)]
    cum_weights = list(itertools.accumulate(1 / (rank + 1) for rank in range(vocab)))
    for doc_id in range(1, n_docs + 1):
        length = rng.randint(200, 800)
        text = " ".join(rng.choices(words, cum_weights=cum_weights, k=length))
        yield doc_id, f"synthetic/{doc_id}.txt", text


# ---------------------------------------------------------------------------
# Вимірювання
# ---------------------------------------------------------------------------
@dataclass
class Row:
    label: str
    peak_mb: float
    file_mb: float
    load_s: float
    build_s: float
    n_postings: int
    n_terms: int
    fingerprint: dict[str, list[tuple[int, int]]]

    @property
    def peak_bytes_per_posting(self) -> float:
        return self.peak_mb * 1024 * 1024 / self.n_postings


def measure(
    label: str,
    index_cls: type[InvertedIndex],
    make_docs: Callable[[], Iterator[Doc]],
    loads: int,
) -> Row:
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    index = index_cls()
    for doc_id, path, text in make_docs():
        index.add_document(doc_id, path, list(tokenize(text)))
    build_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_terms = len(index)
    n_postings = sum(len(postings_as_pairs(index, t)) for t in index.index)
    # Відбиток для звірки варіантів: перші 300 термінів за алфавітом
    sample_terms = sorted(index.index)[:300]
    fingerprint = {t: postings_as_pairs(index, t) for t in sample_terms}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.pkl"
        save_index(index, path)
        file_size = path.stat().st_size
        del index
        gc.collect()

        times = []
        for _ in range(loads):
            t0 = time.perf_counter()
            loaded = load_index(path)
            times.append(time.perf_counter() - t0)
            del loaded
            gc.collect()

    return Row(
        label=label,
        peak_mb=peak / 1024 / 1024,
        file_mb=file_size / 1024 / 1024,
        load_s=statistics.median(times),
        build_s=build_s,
        n_postings=n_postings,
        n_terms=n_terms,
        fingerprint=fingerprint,
    )


def print_report(rows: list[Row]) -> None:
    base = rows[0]
    print(f"\nТерміни: {base.n_terms:,} · постінги: {base.n_postings:,}\n")
    print(
        "| Postings representation | Peak memory (build) | Index file size | Load time |"
    )
    print("|---|---|---|---|")
    for r in rows:
        print(
            f"| {r.label} | {r.peak_mb:.1f} MB | {r.file_mb:.1f} MB | {r.load_s:.2f} s |"
        )

    print("\nДодатково (для абзацу «куди пішли байти»):")
    for r in rows:
        print(
            f"- {r.label}: {r.peak_bytes_per_posting:.0f} B пікової пам'яті на постінг, "
            f"{r.peak_mb / base.peak_mb:.2f}× від рядка 1, build {r.build_s:.1f} s"
        )

    same = all(r.fingerprint == base.fingerprint for r in rows)
    print(
        f"\nВаріанти дають однакові постінги (перевірка вибірки): {'так' if same else 'НІ!'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M4: memory study of posting representations."
    )
    parser.add_argument("data_dir", type=Path, nargs="?", help="Папка з *.xml.bz2")
    parser.add_argument(
        "--limit", type=int, default=None, help="Макс. кількість документів"
    )
    parser.add_argument(
        "--synthetic",
        type=int,
        default=None,
        metavar="N",
        help="Синтетичний корпус із N документів (лише для перевірки скрипта)",
    )
    parser.add_argument(
        "--loads", type=int, default=3, help="Скільки разів міряти load (медіана)"
    )
    args = parser.parse_args()

    if args.synthetic is not None:

        def make_docs() -> Iterator[Doc]:
            return synthetic_docs(args.synthetic)

        print(f"Синтетичний корпус: {args.synthetic} документів (НЕ для README)")
    elif args.data_dir is not None:

        def make_docs() -> Iterator[Doc]:
            return real_docs(args.data_dir, args.limit)

        print(f"Реальний корпус: {args.data_dir}, limit={args.limit}")
    else:
        parser.error("Вкажіть data_dir або --synthetic N")

    rows = []
    for label, cls in VARIANTS:
        print(f"Вимірюю: {label} ...", flush=True)
        rows.append(measure(label, cls, make_docs, args.loads))

    print_report(rows)


if __name__ == "__main__":
    main()
