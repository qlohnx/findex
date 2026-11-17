"""One bounded HTTP fetch, with transient retries and an auditable result."""

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    status: int | None
    body: bytes
    elapsed: float
    attempts: int
    error: str | None = None
    content_type: str = ""
    location: str | None = None

    @property
    def byte_count(self) -> int:
        return len(self.body)


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        date = parsedate_to_datetime(value)
    except TypeError, ValueError, IndexError:
        return None
    if date.tzinfo is None:
        date = date.replace(tzinfo=UTC)
    return max(0.0, (date - datetime.now(UTC)).total_seconds())


def _backoff(attempt: int, base_delay: float, jitter: Callable[[], float]) -> float:
    return min(8.0, base_delay * 2 ** (attempt - 1)) + max(0.0, jitter())


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_attempts: int = 3,
    base_delay: float = 0.25,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    jitter: Callable[[], float] = random.random,
    before_attempt: Callable[[str], Awaitable[None]] | None = None,
) -> FetchResult:
    """Fetch without following redirects outside the crawler's domain policy."""
    if timeout <= 0 or max_attempts < 1 or base_delay < 0:
        raise ValueError("timeout and max_attempts must be positive; base_delay >= 0")
    started = time.perf_counter()
    for attempt in range(1, max_attempts + 1):
        if before_attempt is not None:
            await before_attempt(url)
        try:
            async with asyncio.timeout(timeout):
                response = await client.get(url, follow_redirects=False)
        except (TimeoutError, httpx.TransportError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.warning("url=%s error=%s attempt=%d", url, error, attempt)
            if attempt == max_attempts:
                return FetchResult(
                    url, None, b"", time.perf_counter() - started, attempt, error
                )
            await sleep(_backoff(attempt, base_delay, jitter))
            continue

        status = response.status_code
        body = response.content
        log.info(
            "url=%s status=%d bytes=%d attempt=%d",
            url,
            status,
            len(body),
            attempt,
        )
        if (status == 429 or 500 <= status < 600) and attempt < max_attempts:
            delay = _backoff(attempt, base_delay, jitter)
            if status == 429:
                delay = max(
                    delay, _retry_after(response.headers.get("retry-after")) or 0.0
                )
            log.warning("url=%s status=%d retry_in=%.2fs", url, status, delay)
            await sleep(delay)
            continue
        error = f"HTTP {status}" if status >= 400 else None
        return FetchResult(
            url=url,
            status=status,
            body=body,
            elapsed=time.perf_counter() - started,
            attempts=attempt,
            error=error,
            content_type=response.headers.get("content-type", ""),
            location=response.headers.get("location"),
        )
    raise AssertionError("unreachable fetch attempt state")
