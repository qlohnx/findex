"""Run the Lab 05 thread experiment under CPython 3.13t without installing findex.

Run with: uv run --no-project --python 3.13t --with psutil python scripts/benchmark_free_threaded.py data/simplewiki-latest-pages-articles.xml.bz2 --limit 1000
The project requires Python 3.14, so this isolated experiment imports only its
stdlib-based indexing modules from src/. It does not alter the project venv.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from findex.parallel import build_from_chunks, stage_pages  # noqa: E402
from findex.stats import iter_pages  # noqa: E402


class PeakParentRSS:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.wait(0.02):
            self.peak = max(self.peak, self.process.memory_info().rss)
        self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> PeakParentRSS:
        self.peak = self.process.memory_info().rss
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop.set()
        self.thread.join()


def main() -> None:
    if sys.version_info[:2] != (3, 13) or sys._is_gil_enabled():
        raise SystemExit("This experiment requires CPython 3.13t with GIL disabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--workers", type=int, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/lab05-3.13t.json"))
    args = parser.parse_args()
    if args.limit < 1 or args.chunk_size < 1 or args.runs < 3 or args.warmups < 0:
        parser.error("limit, chunk-size > 0; runs >= 3; warmups >= 0 required")
    workers = sorted(set(args.workers or [1, 2, 4, 8, os.cpu_count() or 1]))
    if min(workers) < 1:
        parser.error("workers must be positive")
    with tempfile.TemporaryDirectory(prefix="findex-free-threaded-") as temporary:
        start = time.perf_counter()
        chunks = stage_pages(
            itertools.islice(iter_pages(args.corpus), args.limit),
            str(args.corpus),
            Path(temporary),
            chunk_size=args.chunk_size,
        )
        stage_s = time.perf_counter() - start
        if not chunks:
            parser.error("corpus yielded no documents")
        print(
            f"GIL={sys._is_gil_enabled()} staged {len(chunks)} chunks in {stage_s:.2f}s",
            flush=True,
        )
        rows = []
        expected_shape = None
        for count in workers:
            for _ in range(args.warmups):
                build_from_chunks(chunks, executor="threads", workers=count)
            samples = []
            for repeat in range(args.runs):
                with PeakParentRSS() as memory:
                    index, timing = build_from_chunks(
                        chunks, executor="threads", workers=count
                    )
                shape = (index.num_docs, len(index))
                if expected_shape is None:
                    expected_shape = shape
                elif shape != expected_shape:
                    raise RuntimeError(
                        f"index shape differs: {shape} != {expected_shape}"
                    )
                samples.append(
                    {
                        "wall_s": timing.wall_seconds,
                        "cpu_s": timing.parent_cpu_seconds,
                        "parent_rss_mb": memory.peak / (1024 * 1024),
                        "merge_s": timing.merge_seconds,
                    }
                )
                print(
                    f"threads 3.13t {count:2d} run {repeat + 1}: "
                    f"wall={timing.wall_seconds:.2f}s cpu={timing.parent_cpu_seconds:.2f}s "
                    f"parent_rss={memory.peak / (1024 * 1024):.0f}MB",
                    flush=True,
                )
            row = {"executor": "threads-3.13t", "workers": count, "samples": samples}
            for metric in ("wall_s", "cpu_s", "parent_rss_mb", "merge_s"):
                row[metric] = round(statistics.median(s[metric] for s in samples), 3)
            rows.append(row)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(
                    {
                        "python": sys.version,
                        "gil_enabled": sys._is_gil_enabled(),
                        "platform": platform.platform(),
                        "cpu_model": platform.processor(),
                        "logical_cpus": os.cpu_count(),
                        "corpus": str(args.corpus),
                        "limit": args.limit,
                        "documents": expected_shape[0],
                        "terms": expected_shape[1],
                        "chunk_size": args.chunk_size,
                        "chunks": len(chunks),
                        "stage_s": stage_s,
                        "runs": args.runs,
                        "warmups": args.warmups,
                        "rows": rows,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
