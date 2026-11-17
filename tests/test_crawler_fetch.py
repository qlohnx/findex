import asyncio
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx

from findex.crawler.fetch import fetch


def test_fetch_honors_retry_after_and_records_attempts() -> None:
    async def scenario() -> None:
        requests = 0
        delays: list[float] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            if requests == 1:
                return httpx.Response(429, headers={"Retry-After": "2"})
            return httpx.Response(
                200, text="done", headers={"content-type": "text/html"}
            )

        async def fake_sleep(delay: float) -> None:
            delays.append(delay)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch(
                client, "https://example.test/page", sleep=fake_sleep, jitter=lambda: 0
            )
        assert requests == result.attempts == 2
        assert delays == [2.0]
        assert result.status == 200
        assert result.body == b"done"
        assert result.byte_count == 4
        assert result.elapsed >= 0

    asyncio.run(scenario(), debug=True)


def test_fetch_retries_transient_error_but_not_404() -> None:
    async def scenario() -> None:
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(503 if calls == 1 else 200, text="ok")

        async def no_wait(_: float) -> None:
            return None

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch(
                client,
                "https://example.test/page",
                sleep=no_wait,
                jitter=lambda: 0,
            )
            assert result.status == 200 and result.attempts == 2
        calls = 0
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(404))
        ) as client:
            result = await fetch(client, "https://example.test/missing")
            assert result.status == 404 and result.attempts == 1

    asyncio.run(scenario(), debug=True)


def test_fetch_exhausts_transport_timeouts() -> None:
    async def scenario() -> None:
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("slow", request=request)

        async def no_wait(_: float) -> None:
            return None

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch(
                client,
                "https://example.test/slow",
                sleep=no_wait,
                jitter=lambda: 0,
            )
        assert calls == result.attempts == 3
        assert result.status is None and result.body == b""
        assert result.error is not None and "ReadTimeout" in result.error

    asyncio.run(scenario(), debug=True)


def test_fetch_enforces_async_timeout_and_http_date_retry_after() -> None:
    async def scenario() -> None:
        waits: list[float] = []

        async def no_wait(delay: float) -> None:
            waits.append(delay)

        async def slow(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.02)
            return httpx.Response(200)

        async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
            result = await fetch(
                client,
                "https://example.test/slow",
                timeout=0.001,
                sleep=no_wait,
                jitter=lambda: 0,
            )
        assert result.status is None and result.attempts == 3
        assert result.error is not None and "TimeoutError" in result.error
        assert len(waits) == 2

        after = format_datetime(datetime.now(UTC) + timedelta(seconds=5))
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(429, headers={"Retry-After": after})
            )
        ) as client:
            result = await fetch(
                client,
                "https://example.test/rate-limited",
                max_attempts=2,
                sleep=no_wait,
                jitter=lambda: 0,
            )
        assert result.status == 429
        assert waits[-1] >= 3

    asyncio.run(scenario(), debug=True)


def test_each_retry_invokes_per_host_pacing_hook() -> None:
    async def scenario() -> None:
        starts: list[str] = []
        calls = 0

        async def before_attempt(url: str) -> None:
            starts.append(url)

        async def no_wait(_: float) -> None:
            return None

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(429 if calls == 1 else 200)

        url = "https://example.test/page"
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch(
                client,
                url,
                jitter=lambda: 0,
                sleep=no_wait,
                before_attempt=before_attempt,
            )
        assert result.status == 200
        assert starts == [url, url]

    asyncio.run(scenario(), debug=True)
