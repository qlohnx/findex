"""Per-host, fail-closed robots.txt cache."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import fetch
from findex.crawler.urls import origin

USER_AGENT = "findex/0.6 (+https://github.com/qlohnx/findex/issues)"
log = logging.getLogger(__name__)


class RobotsCache:
    """Fetch robots once per origin; disallow on errors other than 404."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        user_agent: str = USER_AGENT,
        before_attempt: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.client = client
        self.user_agent = user_agent
        self.before_attempt = before_attempt
        self._rules: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _parser(self, url: str) -> RobotFileParser:
        host = origin(url)
        if host in self._rules:
            return self._rules[host]
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            if host in self._rules:
                return self._rules[host]
            robots_url = f"{host}/robots.txt"
            result = await fetch(
                self.client,
                robots_url,
                max_attempts=2,
                before_attempt=self.before_attempt,
            )
            parser = RobotFileParser()
            parser.set_url(robots_url)
            if result.status == 200:
                await asyncio.to_thread(
                    parser.parse,
                    result.body.decode("utf-8", errors="replace").splitlines(),
                )
            elif result.status == 404:
                await asyncio.to_thread(parser.parse, ["User-agent: *", "Allow: /"])
            else:
                await asyncio.to_thread(parser.parse, ["User-agent: *", "Disallow: /"])
                log.warning("robots unavailable at %s; denying crawl", robots_url)
            self._rules[host] = parser
            return parser

    async def allowed(self, url: str) -> bool:
        parser = await self._parser(url)
        return parser.can_fetch(self.user_agent, url)

    async def delay(self, url: str) -> float:
        parser = await self._parser(url)
        return float(parser.crawl_delay(self.user_agent) or 0)
