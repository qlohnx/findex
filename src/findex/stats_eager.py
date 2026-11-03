import argparse
import itertools
import logging
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import TypedDict

from findex.stats import iter_documents
from findex.tokenize import tokenize

log = logging.getLogger(__name__)


class EagerStats(TypedDict):
    doc_count: int
    total_tokens: int
    vocab_size: int


def collect_stats_eager(data_dir: Path, limit: int | None = None) -> EagerStats:
    bz2_files = list(data_dir.glob("*.xml.bz2"))
    if not bz2_files:
        log.warning("no .xml.bz2 files in %s", data_dir)
        return {"doc_count": 0, "total_tokens": 0, "vocab_size": 0}

    file_path = bz2_files[0]

    # 1. Жадібно читаємо всі документи генератора у звичайний список у пам'яті
    doc_generator = iter_documents(file_path)
    if limit is not None:
        doc_generator = itertools.islice(doc_generator, limit)

    all_docs = list(doc_generator)  # Всі сирі тексти завантажуються сюди одразу!
    doc_count = len(all_docs)

    # 2. Токенізуємо кожен документ у список списків
    all_tokens_lists = [list(tokenize(text)) for text in all_docs]

    # 3. Рахуємо статистику
    term_counts: Counter[str] = Counter()
    total_tokens = 0
    for tokens in all_tokens_lists:
        total_tokens += len(tokens)
        term_counts.update(tokens)

    return {
        "doc_count": doc_count,
        "total_tokens": total_tokens,
        "vocab_size": len(term_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute findex dataset statistics (Eager version)."
    )
    parser.add_argument("data_dir", type=Path, help="Path to the dataset directory")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of documents to process"
    )
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    stats = collect_stats_eager(args.data_dir, limit=args.limit)

    elapsed_time = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"EAGER VERSION (limit={args.limit}):")
    print(f"Documents processed: {stats['doc_count']}")
    print(f"Total tokens:        {stats['total_tokens']}")
    print(f"Vocabulary size:     {stats['vocab_size']}")
    print(f"Elapsed time:   {elapsed_time:.4f} seconds")
    print(f"Peak memory:    {peak_memory / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
