"""Bounded, robots-aware async URL frontier yielding pages as they arrive."""

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urlsplit

import httpx

from findex.crawler.fetch import fetch
from findex.crawler.robots import USER_AGENT, RobotsCache
from findex.crawler.urls import canonical_url, in_scope

log = logging.getLogger(__name__)
REDIRECTS = {301, 302, 303, 307, 308}


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str


@dataclass(slots=True)
class CrawlStats:
    pages: int = 0
    errors: int = 0
    skipped: int = 0
    in_flight: int = 0
    queue_depth: int = 0


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._title = False
        self._hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "template", "noscript"}:
            self._hidden += 1
        elif tag == "title":
            self._title = True
        elif tag == "a":
            for name, value in attrs:
                if name == "href" and value is not None:
                    self.links.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "template", "noscript"}:
            self._hidden = max(0, self._hidden - 1)
        elif tag == "title":
            self._title = False

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if not stripped or self._hidden:
            return
        if self._title:
            self.title_parts.append(stripped)
        else:
            self.text_parts.append(stripped)


def _parse_html(body: bytes) -> tuple[str, str, list[str]]:
    parser = _PageParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    return " ".join(parser.title_parts), " ".join(parser.text_parts), parser.links


async def crawl(
    seeds: Iterable[str],
    *,
    max_pages: int = 500,
    concurrency: int = 10,
    per_host: int = 2,
    allowed_domains: Iterable[str] | None = None,
    per_host_delay: float = 0.1,
    timeout: float = 10.0,
    user_agent: str = USER_AGENT,
    client: httpx.AsyncClient | None = None,
    stats: CrawlStats | None = None,
) -> AsyncIterator[Page]:
    """Crawl only allowed domains; close workers and client on stop or cancellation."""
    if max_pages < 1 or concurrency < 1 or per_host < 1 or timeout <= 0:
        raise ValueError(
            "max_pages, concurrency, per_host and timeout must be positive"
        )
    if per_host_delay < 0:
        raise ValueError("per_host_delay must be nonnegative")
    normalized_seeds = [canonical_url(seed) for seed in seeds]
    if not normalized_seeds or any(seed is None for seed in normalized_seeds):
        raise ValueError("seeds must be valid HTTP(S) URLs")
    safe_seeds = [seed for seed in normalized_seeds if seed is not None]
    domains = (
        {domain.lower().strip(".") for domain in allowed_domains}
        if allowed_domains is not None
        else {urlsplit(seed).hostname or "" for seed in safe_seeds}
    )
    if (
        not domains
        or "" in domains
        or any(not in_scope(seed, domains) for seed in safe_seeds)
    ):
        raise ValueError("each seed must be within allowed_domains")

    report = stats if stats is not None else CrawlStats()
    frontier: asyncio.Queue[str] = asyncio.Queue()
    results: asyncio.Queue[Page | None] = asyncio.Queue()
    seen: set[str] = set()
    limit_reached = asyncio.Event()
    global_limit = asyncio.Semaphore(concurrency)
    host_limits: dict[str, asyncio.Semaphore] = {}
    host_locks: dict[str, asyncio.Lock] = {}
    host_delays: dict[str, float] = {}
    last_start: dict[str, float] = {}
    next_start: dict[str, float] = {}

    async def pace(url: str) -> None:
        host = urlsplit(url).netloc
        lock = host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = next_start.get(host, 0.0) - asyncio.get_running_loop().time()
            if wait > 0:
                await asyncio.sleep(wait)
            started = asyncio.get_running_loop().time()
            last_start[host] = started
            next_start[host] = started + host_delays.get(host, per_host_delay)

    def enqueue(value: str, base: str | None = None) -> bool:
        url = canonical_url(value, base)
        if url is None or not in_scope(url, domains) or url in seen:
            return False
        seen.add(url)
        frontier.put_nowait(url)
        report.queue_depth = frontier.qsize()
        return True

    for seed in safe_seeds:
        enqueue(seed)

    async with AsyncExitStack() as stack:
        if client is None:
            client = await stack.enter_async_context(
                httpx.AsyncClient(headers={"User-Agent": user_agent})
            )
        else:
            client.headers["User-Agent"] = user_agent
        robots = RobotsCache(client, user_agent, before_attempt=pace)

        async def worker() -> None:
            while not limit_reached.is_set():
                url = await frontier.get()
                report.queue_depth = frontier.qsize()
                try:
                    if limit_reached.is_set():
                        return
                    if not await robots.allowed(url):
                        report.skipped += 1
                        log.info("url=%s status=robots-denied bytes=0 elapsed=0", url)
                        continue
                    host = urlsplit(url).netloc
                    host_limit = host_limits.setdefault(
                        host, asyncio.Semaphore(per_host)
                    )
                    host_lock = host_locks.setdefault(host, asyncio.Lock())
                    delay = max(per_host_delay, await robots.delay(url))
                    async with host_lock:
                        host_delays[host] = delay
                        next_start[host] = max(
                            next_start.get(host, 0.0), last_start.get(host, 0.0) + delay
                        )
                    async with global_limit, host_limit:
                        report.in_flight += 1
                        try:
                            result = await fetch(
                                client, url, timeout=timeout, before_attempt=pace
                            )
                        finally:
                            report.in_flight -= 1
                    log.info(
                        "url=%s status=%s bytes=%d elapsed=%.3fs",
                        url,
                        result.status if result.status is not None else "error",
                        result.byte_count,
                        result.elapsed,
                    )
                    if result.status in REDIRECTS and result.location:
                        if not enqueue(result.location, url):
                            report.skipped += 1
                            log.warning("url=%s status=redirect-rejected", url)
                        continue
                    if result.status != 200:
                        report.errors += 1
                        continue
                    if (
                        result.content_type
                        and not result.content_type.lower().startswith("text/html")
                    ):
                        report.skipped += 1
                        continue
                    title, text, links = await asyncio.to_thread(
                        _parse_html, result.body
                    )
                    if report.pages >= max_pages:
                        continue
                    report.pages += 1
                    await results.put(
                        Page(url, title or url, text, datetime.now(UTC).isoformat())
                    )
                    if report.pages == max_pages:
                        limit_reached.set()
                        return
                    for link in links:
                        enqueue(link, url)
                finally:
                    frontier.task_done()

        async def run_workers() -> None:
            try:
                async with asyncio.TaskGroup() as group:
                    workers = [group.create_task(worker()) for _ in range(concurrency)]
                    joined = group.create_task(frontier.join())
                    stopped = group.create_task(limit_reached.wait())
                    await asyncio.wait(
                        {joined, stopped}, return_when=asyncio.FIRST_COMPLETED
                    )
                    joined.cancel()
                    stopped.cancel()
                    for task in workers:
                        task.cancel()
            finally:
                while not frontier.empty():
                    frontier.get_nowait()
                    frontier.task_done()
                report.queue_depth = 0
                results.put_nowait(None)

        runner = asyncio.create_task(run_workers())
        try:
            while True:
                page = await results.get()
                if page is None:
                    await runner
                    return
                yield page
        finally:
            if not runner.done():
                runner.cancel()
                try:
                    await runner
                except asyncio.CancelledError:
                    pass
