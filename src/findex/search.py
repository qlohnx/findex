import argparse
import logging
import time
import timeit
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from findex.index import Index, Posting
from findex.ranking import BM25 as BM25
from findex.ranking import SearchResult as SearchResult
from findex.ranking import TfIdf as TfIdf
from findex.ranking import search as search
from findex.ranking import timed as timed
from findex.store import load_index
from findex.tokenize import tokenize

log = logging.getLogger(__name__)


def intersect_postings(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """Реалізує оператор AND для двох відсортованих списків постінгів (Two-Pointer Merge)."""
    i, j = 0, 0
    result: list[Posting] = []

    while i < len(p1) and j < len(p2):
        if p1[i].doc_id == p2[j].doc_id:
            # Для AND частоту TF можна сумувати або брати з першого (для простого булевого пошуку зберігаємо doc_id)
            result.append(Posting(doc_id=p1[i].doc_id, tf=p1[i].tf + p2[j].tf))
            i += 1
            j += 1
        elif p1[i].doc_id < p2[j].doc_id:
            i += 1
        else:
            j += 1

    return result


def union_postings(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """Реалізує оператор OR для двох відсортованих списків постінгів."""
    i, j = 0, 0
    result: list[Posting] = []

    while i < len(p1) and j < len(p2):
        if p1[i].doc_id == p2[j].doc_id:
            result.append(Posting(doc_id=p1[i].doc_id, tf=p1[i].tf + p2[j].tf))
            i += 1
            j += 1
        elif p1[i].doc_id < p2[j].doc_id:
            result.append(p1[i])
            i += 1
        else:
            result.append(p2[j])
            j += 1

    # Додаємо залишки
    while i < len(p1):
        result.append(p1[i])
        i += 1
    while j < len(p2):
        result.append(p2[j])
        j += 1

    return result


def difference_postings(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """Реалізує оператор AND NOT (знаходити документи з p1, яких немає в p2)."""
    i, j = 0, 0
    result: list[Posting] = []

    while i < len(p1) and j < len(p2):
        if p1[i].doc_id == p2[j].doc_id:
            i += 1
            j += 1
        elif p1[i].doc_id < p2[j].doc_id:
            result.append(p1[i])
            i += 1
        else:
            j += 1

    while i < len(p1):
        result.append(p1[i])
        i += 1

    return result


class BooleanSearch:
    """Клас для виконання булевих запитів по оберненому індексу."""

    def __init__(self, index: Index) -> None:
        self.index = index

    def search_and(self, term1: str, term2: str) -> list[Posting]:
        p1 = self.index.get_postings(term1)
        p2 = self.index.get_postings(term2)
        return intersect_postings(p1, p2)

    def search_or(self, term1: str, term2: str) -> list[Posting]:
        p1 = self.index.get_postings(term1)
        p2 = self.index.get_postings(term2)
        return union_postings(p1, p2)

    def search_not(self, term1: str, term2: str) -> list[Posting]:
        """Шукає term1 AND NOT term2."""
        p1 = self.index.get_postings(term1)
        p2 = self.index.get_postings(term2)
        return difference_postings(p1, p2)


# ---------------------------------------------------------------------------
# Запити: розбір (зліва направо) та два рушії — merge і set
# ---------------------------------------------------------------------------
def and_ids(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def or_ids(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def not_ids(a: list[int], b: list[int]) -> list[int]:
    """a AND NOT b."""
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            j += 1
    out.extend(a[i:])
    return out


def and_set(a: list[int], b: list[int]) -> list[int]:
    return sorted(set(a) & set(b))


def or_set(a: list[int], b: list[int]) -> list[int]:
    return sorted(set(a) | set(b))


def not_set(a: list[int], b: list[int]) -> list[int]:
    return sorted(set(a) - set(b))


type Engine = Literal["merge", "set"]
ENGINES: dict[Engine, dict[str, Callable[[list[int], list[int]], list[int]]]] = {
    "merge": {"AND": and_ids, "OR": or_ids, "NOT": not_ids},
    "set": {"AND": and_set, "OR": or_set, "NOT": not_set},
}
OPERATORS = {"AND", "OR", "NOT"}


def term_ids(index: Index, term: str) -> list[int]:
    return [p.doc_id for p in index.get_postings(term)]


def evaluate(index: Index, query: str, engine: Engine = "merge") -> list[int]:
    """Обчислює булевий запит зліва направо (без пріоритетів і дужок).

    ``a b`` = ``a AND b``; ``a OR b``; ``a NOT b`` = ``a AND NOT b``.
    Операнди проходять той самий токенайзер, що й корпус (casefold тощо).
    Запит, що починається з NOT, означає «усі документи, крім ...».
    """
    ops = ENGINES[engine]
    result: list[int] | None = None
    op = "AND"
    for word in query.split():
        if word.upper() in OPERATORS:
            op = word.upper()
            continue
        for term in tokenize(word):
            ids = term_ids(index, term)
            if result is None:
                if op == "NOT":
                    result = ops["NOT"](sorted(index.doc_meta), ids)
                else:
                    result = ids
            else:
                result = ops[op](result, ids)
            op = "AND"
    return result or []


def benchmark(index: Index, repeat: int = 20) -> None:
    """merge vs set на двох найчастіших і двох найрідших термінах."""
    by_df = sorted(index.index, key=lambda t: len(index.index[t]))
    rare, common = by_df[:2], by_df[-2:]
    cases = [
        ("common AND common", common[0], common[1]),
        ("rare AND rare", rare[0], rare[1]),
        ("common AND rare", common[1], rare[0]),
    ]
    print(f"{'case':<20}{'df':>16}{'merge, ms':>12}{'set, ms':>10}")
    for name, t1, t2 in cases:
        a, b = term_ids(index, t1), term_ids(index, t2)
        t_merge = timeit.timeit(lambda a=a, b=b: and_ids(a, b), number=repeat) / repeat
        t_set = timeit.timeit(lambda a=a, b=b: and_set(a, b), number=repeat) / repeat
        df = f"{len(a)}/{len(b)}"
        print(f"{name:<20}{df:>16}{t_merge * 1000:>12.3f}{t_set * 1000:>10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Boolean search over a saved index.")
    parser.add_argument("index_path", type=Path)
    parser.add_argument(
        "query", nargs="?", default="", help='напр. "python code NOT java"'
    )
    parser.add_argument("--engine", choices=sorted(ENGINES), default="merge")
    parser.add_argument(
        "--limit", type=int, default=10, help="Скільки результатів показати"
    )
    parser.add_argument("--bench", action="store_true", help="merge vs set benchmark")
    args = parser.parse_args()

    tracemalloc.start()
    t0 = time.perf_counter()
    index = load_index(args.index_path)
    load_s = time.perf_counter() - t0

    if args.bench:
        benchmark(index)
    if args.query:
        t0 = time.perf_counter()
        ids = evaluate(index, args.query, engine=args.engine)
        search_s = time.perf_counter() - t0
        print(f"Знайдено документів: {len(ids)}")
        for doc_id in ids[: args.limit]:
            meta = index.doc_meta[doc_id]
            print(f"  [{doc_id}] {meta.title or meta.path}")
        log.debug("search time (%s): %.3f ms", args.engine, search_s * 1000)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    log.debug("load time: %.2f s; peak memory: %.1f MB", load_s, peak / 1024 / 1024)


if __name__ == "__main__":
    main()
