"""TF-IDF/BM25 ranking and result presentation."""

from __future__ import annotations

import functools
import heapq
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Protocol, overload

from findex.index import Index, Posting
from findex.query import Query, parse

log = logging.getLogger(__name__)


def timed[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            log.debug(
                "%s took %.2f ms", fn.__name__, (time.perf_counter() - started) * 1000
            )

    return wrapper


class Scorer(Protocol):
    def __call__(self, term: str, posting: Posting, index: Index) -> float: ...


class TfIdf:
    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        tf = 1.0 + math.log(posting.tf)
        idf = math.log((index.num_docs + 1) / (index.df(term) + 1)) + 1.0
        return tf * idf


@dataclass(frozen=True)
class BM25:
    k1: float = 1.5
    b: float = 0.75

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        idf = math.log(1.0 + (index.num_docs - df + 0.5) / (df + 0.5))
        length = index.doc_length(posting.doc_id)
        norm = 1.0 - self.b + self.b * length / (index.avg_doc_length or 1.0)
        tf_weight = posting.tf * (self.k1 + 1) / (posting.tf + self.k1 * norm)
        return idf * tf_weight


@dataclass(order=True, frozen=True)
class SearchResult:
    _sort_key: float = field(init=False, repr=False, compare=True)
    doc_id: int = field(compare=False)
    score: float = field(compare=False)
    title: str = field(default="", compare=False)
    snippet: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_sort_key", -self.score)

    def __str__(self) -> str:
        label = self.title or str(self.doc_id)
        return f"{label} ({self.score:.4f})" + (
            f": {self.snippet}" if self.snippet else ""
        )


def _snippet(text: str, terms: set[str], width: int = 80) -> str:
    folded = text.casefold()
    starts = [folded.find(term) for term in terms if folded.find(term) >= 0]
    start = max(0, (min(starts) if starts else 0) - width)
    result = text[start : start + 2 * width]
    for term in sorted(terms, key=len, reverse=True):
        result = re.sub(
            re.escape(term), lambda m: f"**{m.group(0)}**", result, flags=re.IGNORECASE
        )
    return result


@overload
def search(
    query: str | Query, index: Index, scorer: Scorer | None = None, k: int = 10
) -> list[SearchResult]: ...


@overload
def search(
    query: Index, index: str | Query, scorer: Scorer | None = None, k: int = 10
) -> list[SearchResult]: ...


@timed
def search(
    query: str | Query | Index,
    index: Index | str | Query,
    scorer: Scorer | None = None,
    k: int = 10,
) -> list[SearchResult]:
    if isinstance(query, Index):
        query, index = index, query
    if not isinstance(index, Index) or isinstance(query, Index):
        raise TypeError("search requires a query and an Index")
    if k < 0:
        raise ValueError("k must be nonnegative")
    node = parse(query) if isinstance(query, str) else query
    scorer = scorer or BM25()
    # Cache immutable query trees, never matches from a mutable index.
    matched = node.evaluate(index)
    log.debug("query cache: %s", parse.cache_info())
    scores: dict[int, float] = {doc_id: 0.0 for doc_id in matched}
    for term in node.terms():
        for posting in index.get_postings(term):
            if posting.doc_id in scores:
                scores[posting.doc_id] += scorer(term, posting, index)
    top = heapq.nlargest(k, scores.items(), key=itemgetter(1))
    terms = set(node.terms())
    return [
        SearchResult(
            doc_id,
            score,
            index.doc_meta[doc_id].title or index.doc_meta[doc_id].path,
            _snippet(index.doc_texts.get(doc_id, ""), terms) if index.doc_texts else "",
        )
        for doc_id, score in top
    ]
