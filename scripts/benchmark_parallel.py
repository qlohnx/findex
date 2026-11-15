"""Repeatable Lab 05 benchmark of the same staged corpus and three executors."""

import argparse
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

from findex.cli import _pages
from findex.parallel import ExecutorName, build_from_chunks, stage_pages


class PeakParentRSS:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            self.stop.wait(0.02)
        self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> PeakParentRSS:
        self.peak = self.process.memory_info().rss
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop.set()
        self.thread.join()


def plot_results(
    rows: list[dict[str, object]],
    output: Path,
    free_threaded_rows: list[dict[str, object]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    serial = next(float(row["wall_s"]) for row in rows if row["executor"] == "serial")
    counts = sorted({int(row["workers"]) for row in rows})
    ax.plot(counts, counts, "k--", label="ideal linear")
    ax.scatter([1], [1], color="black", label="serial 3.14")
    for executor in ("threads", "processes"):
        selected = sorted(
            (row for row in rows if row["executor"] == executor),
            key=lambda row: int(row["workers"]),
        )
        ax.plot(
            [int(row["workers"]) for row in selected],
            [serial / float(row["wall_s"]) for row in selected],
            marker="o",
            label=executor,
        )
    if free_threaded_rows:
        selected = sorted(free_threaded_rows, key=lambda row: int(row["workers"]))
        free_serial = float(selected[0]["wall_s"])
        ax.plot(
            [int(row["workers"]) for row in selected],
            [free_serial / float(row["wall_s"]) for row in selected],
            marker="o",
            label="threads 3.13t (GIL off)",
        )
    ax.set(
        xlabel="Workers",
        ylabel="Speedup vs 1 worker in each runtime (staging excluded)",
        title="findex indexing on the same 1,000-page slice",
    )
    ax.set_xticks(counts)
    ax.set_yscale("log", base=2)
    ax.set_ylim(0.125, max(counts) * 1.25)
    ax.grid(alpha=0.3)
    ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--workers", type=int, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/lab05.json"))
    parser.add_argument(
        "--plot", type=Path, default=Path("benchmarks/lab05-speedup.png")
    )
    args = parser.parse_args()
    if args.runs < 3 or args.warmups < 0 or args.chunk_size < 1:
        parser.error("requires --runs >= 3, --warmups >= 0, --chunk-size >= 1")
    counts = sorted(set(args.workers or [1, 2, 4, 8, os.cpu_count() or 1]))
    if min(counts) < 1:
        parser.error("workers must be positive")

    with tempfile.TemporaryDirectory(prefix="findex-benchmark-") as temporary:
        pages = _pages(args.corpus)
        if args.limit is not None:
            from itertools import islice

            pages = islice(pages, args.limit)
        start = time.perf_counter()
        chunks = stage_pages(
            pages, str(args.corpus), Path(temporary), chunk_size=args.chunk_size
        )
        stage_s = time.perf_counter() - start
        if not chunks:
            parser.error("corpus yielded no documents")
        print(f"staged {len(chunks)} chunks in {stage_s:.2f}s", flush=True)
        rows: list[dict[str, object]] = []
        configurations: list[tuple[ExecutorName, int]] = [("serial", 1)]
        configurations.extend(
            (executor, n) for executor in ("threads", "processes") for n in counts
        )
        expected_shape: tuple[int, int] | None = None
        for executor, workers in configurations:
            for _ in range(args.warmups):
                build_from_chunks(chunks, executor=executor, workers=workers)
            samples: list[dict[str, float]] = []
            for repeat in range(args.runs):
                with PeakParentRSS() as memory:
                    index, timing = build_from_chunks(
                        chunks, executor=executor, workers=workers
                    )
                shape = (index.num_docs, len(index))
                if expected_shape is None:
                    expected_shape = shape
                elif shape != expected_shape:
                    raise RuntimeError(
                        f"index shape differs: {shape} != {expected_shape}"
                    )
                cpu = timing.parent_cpu_seconds
                if executor == "processes":
                    cpu += timing.worker_cpu_seconds
                samples.append(
                    {
                        "wall_s": timing.wall_seconds,
                        "cpu_s": cpu,
                        "parent_rss_mb": memory.peak / (1024 * 1024),
                        "merge_s": timing.merge_seconds,
                    }
                )
                print(
                    f"{executor:9s} {workers:2d} run {repeat + 1}: "
                    f"wall={timing.wall_seconds:.2f}s cpu={cpu:.2f}s "
                    f"parent_rss={memory.peak / (1024 * 1024):.0f}MB",
                    flush=True,
                )
            row: dict[str, object] = {"executor": executor, "workers": workers}
            for metric in ("wall_s", "cpu_s", "parent_rss_mb", "merge_s"):
                row[metric] = round(
                    statistics.median(sample[metric] for sample in samples), 3
                )
            row["wall_spread_s"] = round(
                max(sample["wall_s"] for sample in samples)
                - min(sample["wall_s"] for sample in samples),
                3,
            )
            row["samples"] = samples
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
        plot_results(rows, args.plot)
        print(f"wrote {args.out} and {args.plot}", flush=True)


if __name__ == "__main__":
    main()
