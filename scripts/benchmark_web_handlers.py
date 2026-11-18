"""Compare sync and blocking-async FastAPI routes with 20 in-process clients.

This isolates Starlette's thread-pool offload from network and server overhead.
It is M2 evidence, not a substitute for the lab's oha deployment load test.
"""

import asyncio
import json
import math
import platform
import time
from statistics import median

import httpx
from fastapi import FastAPI

from findex.index import build_index
from findex.web.app import search_index

CLIENTS = 20
REPETITIONS = 6
TRIALS = 3


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(p * len(ordered)) - 1]


async def measure(app: FastAPI, route: str) -> dict[str, float]:
    latencies: list[float] = []
    done = asyncio.Event()
    max_lag = 0.0

    async def ticker() -> None:
        nonlocal max_lag
        expected = time.perf_counter() + 0.01
        while not done.is_set():
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            max_lag = max(max_lag, now - expected)
            expected = now + 0.01

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://benchmark.test"
    ) as client:
        launched_at = time.perf_counter()

        async def request() -> None:
            response = await client.get(route)
            assert response.status_code == 200, response.text
            latencies.append((time.perf_counter() - launched_at) * 1000)

        ticker_task = asyncio.create_task(ticker())
        start = time.perf_counter()
        await asyncio.gather(*(request() for _ in range(CLIENTS)))
        wall_ms = (time.perf_counter() - start) * 1000
        done.set()
        await ticker_task

    return {
        "wall_ms": round(wall_ms, 2),
        "p50_ms": round(percentile(latencies, 0.50), 2),
        "p95_ms": round(percentile(latencies, 0.95), 2),
        "max_loop_lag_ms": round(max_lag * 1000, 2),
    }


async def main() -> None:
    index = build_index(
        (f"Document {n}", f"Python search engine document {n}") for n in range(1500)
    )
    app = FastAPI()

    def work() -> int:
        result = None
        for _ in range(REPETITIONS):
            result = search_index(
                "python OR search", index, k=10, scorer="bm25", page=1
            )
        assert result is not None
        return result.total

    @app.get("/sync")
    def sync_route() -> dict[str, int]:
        return {"total": work()}

    @app.get("/async")
    async def async_route() -> dict[str, int]:
        # Deliberately wrong for CPU work: this blocks the event loop.
        return {"total": work()}

    results: dict[str, list[dict[str, float]]] = {"sync": [], "async": []}
    for route in results:
        await measure(app, f"/{route}")  # warmup
        for _ in range(TRIALS):
            results[route].append(await measure(app, f"/{route}"))

    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "documents": index.num_docs,
                "clients": CLIENTS,
                "searches_per_request": REPETITIONS,
                "trials": TRIALS,
                "median": {
                    route: {
                        key: round(median(trial[key] for trial in measurements), 2)
                        for key in measurements[0]
                    }
                    for route, measurements in results.items()
                },
                "raw": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
