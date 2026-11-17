import asyncio
import importlib
from urllib.parse import urlsplit

import httpx
import pytest

from findex.crawler import CrawlStats, crawl
from findex.crawler.urls import canonical_url


def test_url_canonicalization() -> None:
    assert (
        canonical_url("HTTPS://Example.TEST/a/#section?ignored")
        == "https://example.test/a"
    )
    assert canonical_url("/a/?z=2&a=1#fragment", "https://EXAMPLE.test/") == (
        "https://example.test/a?a=1&z=2"
    )
    assert canonical_url("javascript:alert(1)", "https://example.test/") is None
    assert canonical_url("https://user:password@example.test/") is None


def test_crawl_obeys_robots_scope_and_deduplicates() -> None:
    async def scenario() -> None:
        fetched: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            fetched.append(str(request.url))
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nDisallow: /private")
            if request.url.path == "/":
                return httpx.Response(
                    200,
                    text=(
                        "<title>Home</title><a href='/a/'>A</a>"
                        "<a href='/a#x'>dup</a><a href='/private'>no</a>"
                        "<a href='https://other.test/'>outside</a>"
                    ),
                    headers={"content-type": "text/html"},
                )
            return httpx.Response(
                200,
                text="<title>A</title><p>Python text</p>",
                headers={"content-type": "text/html"},
            )

        stats = CrawlStats()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            pages = [
                page
                async for page in crawl(
                    ["https://example.test/"],
                    client=client,
                    max_pages=10,
                    per_host_delay=0,
                    stats=stats,
                )
            ]
        assert {page.title for page in pages} == {"Home", "A"}
        assert "Python text" in pages[1].text
        assert fetched.count("https://example.test/a") == 1
        assert "https://example.test/private" not in fetched
        assert all(urlsplit(url).hostname == "example.test" for url in fetched)
        assert stats.pages == 2 and stats.skipped == 1 and stats.queue_depth == 0

    asyncio.run(scenario(), debug=True)


def test_global_limit_across_hosts_and_per_host_delay() -> None:
    async def scenario() -> None:
        active = 0
        peak = 0
        starts: dict[str, list[float]] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal active, peak
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nAllow: /")
            active += 1
            peak = max(peak, active)
            starts.setdefault(request.url.host, []).append(
                asyncio.get_running_loop().time()
            )
            try:
                await asyncio.sleep(0.02)
            finally:
                active -= 1
            if request.url.host == "example.test":
                links = "".join(
                    f"<a href='https://h{i % 3}.example.test/p{i}'>page</a>"
                    for i in range(9)
                )
                return httpx.Response(
                    200, text=links, headers={"content-type": "text/html"}
                )
            return httpx.Response(
                200,
                text="<title>Page</title>text",
                headers={"content-type": "text/html"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            pages = [
                page
                async for page in crawl(
                    ["https://example.test/"],
                    max_pages=10,
                    concurrency=2,
                    per_host=2,
                    per_host_delay=0.04,
                    allowed_domains={"example.test"},
                    client=client,
                )
            ]
        assert len(pages) == 10
        assert peak <= 2 and active == 0
        for host, times in starts.items():
            if host != "example.test":
                assert all(
                    right - left >= 0.035 for left, right in zip(times, times[1:])
                )

    asyncio.run(scenario(), debug=True)


def test_delay_also_separates_robots_from_first_page() -> None:
    async def scenario() -> None:
        starts: list[float] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            starts.append(asyncio.get_running_loop().time())
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nAllow: /")
            return httpx.Response(
                200,
                text="<title>Page</title>text",
                headers={"content-type": "text/html"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            pages = [
                page
                async for page in crawl(
                    ["https://example.test/"],
                    max_pages=1,
                    per_host_delay=0.04,
                    client=client,
                )
            ]
        assert len(pages) == 1
        assert len(starts) == 2 and starts[1] - starts[0] >= 0.035

    asyncio.run(scenario(), debug=True)


def test_worker_exception_propagates_and_cancels_group(monkeypatch) -> None:
    crawler_module = importlib.import_module("findex.crawler.crawl")

    def broken_parser(body: bytes):
        raise RuntimeError("broken HTML parser")

    monkeypatch.setattr(crawler_module, "_parse_html", broken_parser)

    async def scenario() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            return httpx.Response(
                200, text="<title>Broken</title>", headers={"content-type": "text/html"}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async for _ in crawl(
                ["https://example.test/"], client=client, per_host_delay=0
            ):
                pass

    with pytest.raises(ExceptionGroup) as failure:
        asyncio.run(scenario(), debug=True)
    assert any(
        isinstance(exc, RuntimeError) and str(exc) == "broken HTML parser"
        for exc in failure.value.exceptions
    )


def test_crawl_limits_inflight_and_stops_at_page_budget() -> None:
    async def scenario() -> None:
        active = 0
        peak = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal active, peak
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nAllow: /")
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.01)
            finally:
                active -= 1
            if request.url.path == "/":
                links = "".join(f"<a href='/p/{i}'>page</a>" for i in range(30))
                return httpx.Response(
                    200, text=links, headers={"content-type": "text/html"}
                )
            return httpx.Response(
                200,
                text="<title>Page</title>hello",
                headers={"content-type": "text/html"},
            )

        stats = CrawlStats()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            pages = [
                page
                async for page in crawl(
                    ["https://example.test/"],
                    client=client,
                    max_pages=5,
                    concurrency=10,
                    per_host=2,
                    per_host_delay=0,
                    stats=stats,
                )
            ]
        assert len(pages) == stats.pages == 5
        assert peak <= 2
        assert active == stats.in_flight == stats.queue_depth == 0

    asyncio.run(scenario(), debug=True)


def test_crawl_rejects_external_redirect() -> None:
    async def scenario() -> None:
        fetched: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            fetched.append(str(request.url))
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            return httpx.Response(302, headers={"location": "https://outside.test/"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            pages = [
                page
                async for page in crawl(
                    ["https://example.test/"], client=client, per_host_delay=0
                )
            ]
        assert pages == []
        assert fetched == ["https://example.test/robots.txt", "https://example.test/"]

    asyncio.run(scenario(), debug=True)


def test_redirect_loop_is_not_fetched_twice() -> None:
    async def scenario() -> None:
        fetched: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            fetched.append(str(request.url))
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            target = "/a" if request.url.path == "/" else "/"
            return httpx.Response(302, headers={"location": target})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            pages = [
                page
                async for page in crawl(
                    ["https://example.test/"], client=client, per_host_delay=0
                )
            ]
        assert pages == []
        assert fetched == [
            "https://example.test/robots.txt",
            "https://example.test/",
            "https://example.test/a",
        ]

    asyncio.run(scenario(), debug=True)


def test_crawl_early_close_cancels_workers() -> None:
    async def scenario() -> None:
        active = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal active
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            active += 1
            try:
                await asyncio.sleep(0.01)
            finally:
                active -= 1
            links = "".join(f"<a href='/p{i}'>page</a>" for i in range(20))
            return httpx.Response(
                200, text=links, headers={"content-type": "text/html"}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            pages = crawl(["https://example.test/"], client=client, per_host_delay=0)
            await anext(pages)
            await pages.aclose()
        assert active == 0

    asyncio.run(scenario(), debug=True)
