"""Benchmark-only ASGI app exposing a deliberately blocking async search route.

Run with ``uvicorn scripts.benchmark_web_server:app --workers 1`` or ``--workers 4``.
The real service remains ``findex.web:app`` and uses the sync /search route.
"""

from typing import Annotated, Literal

from fastapi import Depends, Query

from findex.index import Index
from findex.web import app
from findex.web.app import get_index, search_index
from findex.web.models import SearchResponse


@app.get("/search-async", response_model=SearchResponse, include_in_schema=False)
async def blocking_async_search(
    q: Annotated[str, Query(min_length=1, max_length=200)],
    k: Annotated[int, Query(ge=1, le=100)] = 10,
    scorer: Literal["bm25", "tfidf"] = "bm25",
    page: Annotated[int, Query(ge=1, le=1000)] = 1,
    index: Index = Depends(get_index),
) -> SearchResponse:
    """Intentionally block the event loop to measure the async anti-pattern."""
    return search_index(q, index, k=k, scorer=scorer, page=page)
