import asyncio

import httpx

from findex.crawler.robots import USER_AGENT, RobotsCache


def test_robots_are_cached_and_obeyed() -> None:
    async def scenario() -> None:
        robots_requests = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal robots_requests
            assert request.headers["user-agent"] == USER_AGENT
            robots_requests += 1
            return httpx.Response(
                200,
                text="User-agent: *\nDisallow: /private\nCrawl-delay: 1\n",
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), headers={"User-Agent": USER_AGENT}
        ) as client:
            cache = RobotsCache(client)
            assert await cache.allowed("https://example.test/public")
            assert not await cache.allowed("https://example.test/private")
            assert await cache.delay("https://example.test/public") == 1
        assert robots_requests == 1

    asyncio.run(scenario(), debug=True)


def test_robots_404_allows_but_other_failure_denies() -> None:
    async def scenario() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404 if request.url.host == "open.test" else 403)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            cache = RobotsCache(client)
            assert await cache.allowed("https://open.test/page")
            assert not await cache.allowed("https://closed.test/page")

    asyncio.run(scenario(), debug=True)


def test_concurrent_requests_share_one_robots_fetch() -> None:
    async def scenario() -> None:
        requests = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            await asyncio.sleep(0.01)
            return httpx.Response(200, text="User-agent: *\nAllow: /")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            cache = RobotsCache(client)
            decisions = await asyncio.gather(
                *(cache.allowed(f"https://example.test/page/{i}") for i in range(10))
            )
        assert all(decisions)
        assert requests == 1

    asyncio.run(scenario(), debug=True)
