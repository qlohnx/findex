"""FastAPI routes backed by the existing in-memory search index."""

import logging
import re
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, cast
from urllib.parse import urlencode
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from findex.index import Index
from findex.query import parse
from findex.ranking import BM25, TfIdf
from findex.ranking import search as ranked_search
from findex.store import load_index
from findex.web.config import Settings
from findex.web.models import (
    DocumentResponse,
    SearchResponse,
    SearchResult,
    StatsResponse,
)

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).parent


def highlight_snippet(snippet: str) -> Markup:
    """Escape document text before adding trusted HTML highlight tags."""
    parts = re.split(r"\*\*(.*?)\*\*", snippet)
    return Markup("").join(
        Markup("<mark>") + escape(part) + Markup("</mark>")
        if position % 2
        else escape(part)
        for position, part in enumerate(parts)
    )


def search_index(
    query_text: str,
    index: Index,
    *,
    k: int,
    scorer: Literal["bm25", "tfidf"],
    page: int,
) -> SearchResponse:
    started = time.perf_counter()
    try:
        query = parse(query_text)
        total = len(query.evaluate(index))
        ranker = BM25() if scorer == "bm25" else TfIdf()
        top = ranked_search(query, index, ranker, k=min(page * k, total))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    results = [
        SearchResult(
            doc_id=row.doc_id,
            title=row.title,
            score=row.score,
            snippet=row.snippet,
        )
        for row in top[(page - 1) * k :]
    ]
    return SearchResponse(
        query=query_text,
        total=total,
        took_ms=(time.perf_counter() - started) * 1000,
        results=results,
        page=page,
        k=k,
    )


def get_index(request: Request) -> Index:
    index = getattr(request.app.state, "index", None)
    if not isinstance(index, Index):
        raise HTTPException(status_code=503, detail="Index is not loaded")
    return index


def get_settings(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if not isinstance(settings, Settings):
        raise HTTPException(status_code=503, detail="Settings are not loaded")
    return settings


def create_app(
    *,
    settings: Settings | None = None,
    index_loader: Callable[[Path], Index] = load_index,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
        # BaseSettings supplies the required index_path from INDEX_PATH or .env.
        configured = settings if settings is not None else Settings()  # pyright: ignore[reportCallIssue]
        index = index_loader(configured.index_path)
        application.state.settings = configured
        application.state.index = index
        application.state.started_at = time.monotonic()
        log.info("loaded %d documents from %s", index.num_docs, configured.index_path)
        try:
            yield
        finally:
            application.state.index = None
            index.close()

    application = FastAPI(title="Findex search", version="0.7.0", lifespan=lifespan)
    application.mount(
        "/static", StaticFiles(directory=WEB_DIR / "static"), name="static"
    )
    templates = Jinja2Templates(directory=WEB_DIR / "templates")
    templates.env.filters["highlight_snippet"] = highlight_snippet

    @application.middleware("http")
    async def request_id_middleware(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = str(uuid4())
        try:
            response = await call_next(request)
        except Exception:
            log.exception(
                "request %s failed: %s %s", request_id, request.method, request.url.path
            )
            response = JSONResponse(
                status_code=500, content={"detail": "Internal server error"}
            )
        response.headers["X-Request-ID"] = request_id
        return response

    @application.get("/health")
    def health(request: Request) -> dict[str, str]:
        if not isinstance(getattr(request.app.state, "index", None), Index):
            raise HTTPException(status_code=503, detail="Index is not loaded")
        return {"status": "ok"}

    @application.get("/search", response_model=SearchResponse)
    def search(
        q: Annotated[str, Query(min_length=1, max_length=200)],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        scorer: Literal["bm25", "tfidf"] = "bm25",
        page: Annotated[int, Query(ge=1, le=1000)] = 1,
        index: Index = Depends(get_index),
    ) -> SearchResponse:
        return search_index(q, index, k=k, scorer=scorer, page=page)

    @application.get("/", include_in_schema=False)
    def home(
        request: Request,
        q: Annotated[str, Query(max_length=200)] = "",
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        scorer: Literal["bm25", "tfidf"] = "bm25",
        page: Annotated[int, Query(ge=1, le=1000)] = 1,
        index: Index = Depends(get_index),
    ) -> Response:
        found: SearchResponse | None = None
        error: str | None = None
        if q.strip():
            try:
                found = search_index(q, index, k=k, scorer=scorer, page=page)
            except HTTPException as exc:
                error = str(exc.detail)

        def page_url(number: int) -> str:
            return (
                str(request.url_for("home"))
                + "?"
                + urlencode({"q": q, "k": k, "scorer": scorer, "page": number})
            )

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "query": q,
                "k": k,
                "scorer": scorer,
                "found": found,
                "error": error,
                "previous_url": page_url(page - 1) if page > 1 else None,
                "next_url": page_url(page + 1)
                if found is not None and page * k < found.total
                else None,
            },
        )

    @application.get("/docs/{doc_id}", response_model=DocumentResponse)
    def document(doc_id: int, index: Index = Depends(get_index)) -> DocumentResponse:
        metadata = index.doc_meta.get(doc_id)
        if metadata is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return DocumentResponse(
            doc_id=doc_id,
            title=metadata.title,
            path=metadata.path,
            text=index.doc_texts.get(doc_id, ""),
        )

    @application.get("/doc/{doc_id}", include_in_schema=False)
    def document_page(
        request: Request, doc_id: int, index: Index = Depends(get_index)
    ) -> Response:
        record = document(doc_id, index)
        return templates.TemplateResponse(request, "document.html", {"doc": record})

    @application.get("/stats", response_model=StatsResponse)
    def stats(
        request: Request,
        index: Index = Depends(get_index),
        configured: Settings = Depends(get_settings),
    ) -> StatsResponse:
        path = configured.index_path
        size = path.stat().st_size if path.is_file() else None
        started_at = cast(float, request.app.state.started_at)
        return StatsResponse(
            documents=index.num_docs,
            vocabulary=len(index),
            postings=sum(map(len, index.values())),
            index_bytes=size,
            uptime_seconds=time.monotonic() - started_at,
        )

    return application


app = create_app()
