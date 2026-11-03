"""M3: порівняння форматів збереження (pickle vs json): розмір, save, load.

Запуск:
    uv run python -m findex.format_study data/ --limit 5000
"""

import argparse
import itertools
import tempfile
import time
from pathlib import Path

from findex.index import build_index
from findex.stats import iter_pages
from findex.store import FORMATS, load_index, save_index


def main() -> None:
    parser = argparse.ArgumentParser(description="M3: pickle vs json.")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--positions", action="store_true")
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*.xml.bz2"))
    if not files:
        raise SystemExit(f"Не знайдено *.xml.bz2 у {args.data_dir}")
    pages = iter_pages(files[0])
    if args.limit is not None:
        pages = itertools.islice(pages, args.limit)
    index = build_index(pages, source=files[0].name, positions=args.positions)

    print("| Format | File size | Save time | Load time |")
    print("|---|---|---|---|")
    with tempfile.TemporaryDirectory() as tmp:
        for fmt in FORMATS:
            path = Path(tmp) / f"index.{fmt}"
            t0 = time.perf_counter()
            save_index(index, path, fmt=fmt)
            save_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            load_index(path, fmt=fmt)
            load_s = time.perf_counter() - t0
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"| {fmt} | {size_mb:.1f} MB | {save_s:.2f} s | {load_s:.2f} s |")


if __name__ == "__main__":
    main()
