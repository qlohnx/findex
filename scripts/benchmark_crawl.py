"""Deterministic, permission-safe 200-page local-mirror crawl benchmark."""

import argparse
import asyncio
import json
import logging
import os
import platform
import statistics
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

import httpx
import psutil

from findex.crawler import CrawlStats, Page, crawl

SEED = "https://mirror.test/"
USER_AGENT = "findex/0.6 (+https://github.com/qlohnx/findex/issues)"


class PeakRSS:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.wait(0.02):
            self.peak = max(self.peak, self.process.memory_info().rss)
        self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> PeakRSS:
        self.peak = self.process.memory_info().rss
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop.set()
        self.thread.join()


def _write_page(stream: TextIO, page: Page) -> None:
    stream.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
    stream.flush()


async def run_one(
    concurrency: int,
    output: Path,
    *,
    max_pages: int = 200,
    latency: float = 0.03,
    inject_429: bool = False,
) -> CrawlStats:
    attempts: dict[str, int] = {}

    async def mirror(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host != "mirror.test" and not host.endswith(".mirror.test"):
            raise AssertionError(f"out-of-scope request: {request.url}")
        await asyncio.sleep(latency)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /", request=request)
        if request.url.path == "/":
            links = "".join(
                f"<a href='https://h{i % 20}.mirror.test/page/{i}'>page {i}</a>"
                for i in range(1, 200)
            )
            return httpx.Response(
                200,
                text=f"<title>Local mirror</title><p>Root page</p>{links}",
                headers={"content-type": "text/html"},
                request=request,
            )
        url = str(request.url)
        attempts[url] = attempts.get(url, 0) + 1
        if inject_429 and request.url.path == "/page/17" and attempts[url] == 1:
            return httpx.Response(429, headers={"retry-after": "0.02"}, request=request)
        return httpx.Response(
            200,
            text=f"<title>Page {request.url.path}</title><p>Searchable local text</p>",
            headers={"content-type": "text/html"},
            request=request,
        )

    stats = CrawlStats()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(mirror), headers={"User-Agent": USER_AGENT}
    ) as client:
        stream = await asyncio.to_thread(output.open, "w", encoding="utf-8")
        try:
            async for page in crawl(
                [SEED],
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=2,
                per_host_delay=0.01,
                allowed_domains={"mirror.test"},
                client=client,
                stats=stats,
            ):
                await asyncio.to_thread(_write_page, stream, page)
        finally:
            await asyncio.to_thread(stream.close)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--latency", type=float, default=0.03)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/lab06.json"))
    parser.add_argument(
        "--evidence", type=Path, default=Path("benchmarks/lab06-crawl-evidence.txt")
    )
    args = parser.parse_args()
    if args.runs < 3 or args.warmups < 0 or args.max_pages < 200 or args.latency <= 0:
        parser.error("runs >= 3, warmups >= 0, max-pages >= 200, latency > 0 required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="findex-crawl-benchmark-") as temporary:
        output = Path(temporary) / "pages.jsonl"
        for concurrency in (1, 5, 20):
            for _ in range(args.warmups):
                asyncio.run(
                    run_one(
                        concurrency,
                        output,
                        max_pages=args.max_pages,
                        latency=args.latency,
                    ),
                    debug=True,
                )
            samples = []
            for repeat in range(args.runs):
                with PeakRSS() as memory:
                    started = time.perf_counter()
                    stats = asyncio.run(
                        run_one(
                            concurrency,
                            output,
                            max_pages=args.max_pages,
                            latency=args.latency,
                        ),
                        debug=True,
                    )
                    elapsed = time.perf_counter() - started
                if stats.pages != args.max_pages:
                    raise RuntimeError(
                        f"expected {args.max_pages} pages, got {stats.pages}"
                    )
                sample = {
                    "pages": stats.pages,
                    "wall_s": elapsed,
                    "pages_per_s": stats.pages / elapsed,
                    "errors": stats.errors,
                    "peak_rss_mb": memory.peak / (1024 * 1024),
                }
                samples.append(sample)
                print(
                    f"concurrency={concurrency:2d} run={repeat + 1} "
                    f"pages={stats.pages} wall={elapsed:.2f}s "
                    f"pages/s={sample['pages_per_s']:.1f} errors={stats.errors}",
                    flush=True,
                )
            row = {
                "concurrency": concurrency,
                "pages": args.max_pages,
                "samples": samples,
            }
            for metric in ("wall_s", "pages_per_s", "peak_rss_mb"):
                row[metric] = round(statistics.median(s[metric] for s in samples), 3)
            row["errors"] = sum(s["errors"] for s in samples)
            row["wall_spread_s"] = round(
                max(s["wall_s"] for s in samples) - min(s["wall_s"] for s in samples),
                3,
            )
            rows.append(row)

        logger = logging.getLogger("findex.crawler")
        previous_level = logger.level
        handler = logging.FileHandler(args.evidence, encoding="utf-8", mode="w")
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            asyncio.run(
                run_one(1, output, max_pages=20, latency=args.latency, inject_429=True),
                debug=True,
            )
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            handler.close()

    args.out.write_text(
        json.dumps(
            {
                "site": "generated MockTransport local mirror; no external requests",
                "seed": SEED,
                "max_pages": args.max_pages,
                "hosts": 20,
                "latency_s": args.latency,
                "per_host": 2,
                "per_host_delay_s": 0.01,
                "runs": args.runs,
                "warmups": args.warmups,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpu_model": platform.processor(),
                "logical_cpus": os.cpu_count(),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out} and {args.evidence}", flush=True)


if __name__ == "__main__":
    main()
