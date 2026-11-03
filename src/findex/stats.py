import argparse
import bz2
import itertools
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

from findex.tokenize import tokenize


def collect_stats(data_dir: Path, limit: int | None = None):
    doc_count = 0
    total_tokens = 0
    term_counts = Counter()

    # Шукаємо наш bz2 файл у папці data/
    bz2_files = list(data_dir.glob("*.xml.bz2"))
    if not bz2_files:
        print(f"Попередження: не знайдено .xml.bz2 файлів у папці {data_dir}")
        return {"doc_count": 0, "total_tokens": 0, "vocab_size": 0, "top_50": []}

    file_path = bz2_files[0]
    print(f"Знайдено файл для обробки: {file_path.name}")

    doc_generator = iter_documents(file_path)

    if limit is not None:
        doc_generator = itertools.islice(doc_generator, limit)

    for text in doc_generator:
        doc_count += 1
        if doc_count % 10 == 0:
            print(f"Оброблено документів: {doc_count}...")

        # Перетворюємо генератор токенів на список для цієї статті
        tokens = list(tokenize(text))
        total_tokens += len(tokens)
        term_counts.update(tokens)

    print(f"Всього успішно оброблено документів: {doc_count}")
    vocab_size = len(term_counts)
    top_50 = term_counts.most_common(50)

    return {
        "doc_count": doc_count,
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "top_50": top_50,
    }


def iter_pages(file_path: Path):
    """Генератор (title, text) для кожної статті з XML-bz2 файлу (потоково)."""
    with bz2.open(file_path, "rt", encoding="utf-8") as f:
        context = ElementTree.iterparse(f, events=("end",))
        for _, elem in context:
            if elem.tag.endswith("page"):
                title = ""
                text_content = None
                for child in elem.iter():
                    if child.tag.endswith("title") and not title:
                        title = child.text or ""
                    elif child.tag.endswith("text"):
                        text_content = child.text
                        break

                if text_content:
                    yield title, text_content

                elem.clear()


def iter_documents(file_path: Path):
    """Генератор, який читає XML-bz2 файл на льоту і повертає текст кожної статті (<text>)."""
    for _, text in iter_pages(file_path):
        yield text


def main():
    parser = argparse.ArgumentParser(description="Compute findex dataset statistics.")
    parser.add_argument("data_dir", type=Path, help="Path to the dataset directory")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of documents to process"
    )
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    stats = collect_stats(args.data_dir, limit=args.limit)

    elapsed_time = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Documents processed: {stats['doc_count']}")
    print(f"Total tokens:        {stats['total_tokens']}")
    print(f"Vocabulary size:     {stats['vocab_size']}")
    print("\nTop-50 terms:")
    for term, count in stats["top_50"]:
        print(f"  {term}: {count}")

    print("\n--- Performance ---")
    print(f"Elapsed time:   {elapsed_time:.4f} seconds")
    print(f"Peak memory:    {peak_memory / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
