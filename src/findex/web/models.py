"""Public API response models; the index's internal dataclasses stay unchanged."""

from pydantic import BaseModel


class SearchResult(BaseModel):
    doc_id: int
    title: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    total: int
    took_ms: float
    results: list[SearchResult]
    page: int
    k: int


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    path: str
    text: str


class StatsResponse(BaseModel):
    documents: int
    vocabulary: int
    postings: int
    index_bytes: int | None
    uptime_seconds: float
