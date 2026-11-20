"""Lazy NumPy posting arrays for warm-query ranking.

The persisted Index and its positional postings remain unchanged. Arrays are
created only for queried terms and discarded when a document is added.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from findex.index import Index


class VectorView:
    def __init__(self, index: Index) -> None:
        limit = np.iinfo(np.int32).max
        if index.doc_lengths and (
            next(reversed(index.doc_lengths)) > limit
            or max(index.doc_lengths.values()) > limit
        ):
            raise OverflowError("document IDs or lengths exceed int32")
        self.doc_ids: NDArray[np.int32] = np.fromiter(
            index.doc_lengths, dtype=np.int32, count=index.num_docs
        )
        self.doc_lengths: NDArray[np.int32] = np.fromiter(
            index.doc_lengths.values(), dtype=np.int32, count=index.num_docs
        )
        self.posting_arrays: dict[str, tuple[NDArray[np.int32], NDArray[np.int32]]] = {}

    def postings(
        self, term: str, index: Index
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        if term not in self.posting_arrays:
            postings = index.get_postings(term)
            self.posting_arrays[term] = (
                np.fromiter(
                    (p.doc_id for p in postings), dtype=np.int32, count=len(postings)
                ),
                np.fromiter(
                    (p.tf for p in postings), dtype=np.int32, count=len(postings)
                ),
            )
        return self.posting_arrays[term]


def view_for(index: Index) -> VectorView:
    view: VectorView | None = index.__dict__.get("_vector_view")
    if view is None:
        view = VectorView(index)
        index.__dict__["_vector_view"] = view
    return view


def score_matches(
    index: Index,
    matched_ids: NDArray[np.int32],
    terms: Iterable[str],
    *,
    bm25: bool,
    k1: float = 1.5,
    b: float = 0.75,
) -> NDArray[np.float64]:
    """Return one score per matched ID, in the caller's original match order."""
    view = view_for(index)
    scores = np.zeros(len(matched_ids), dtype=np.float64)
    if not len(matched_ids):
        return scores
    order = np.argsort(matched_ids)
    sorted_ids = matched_ids[order]
    lengths = view.doc_lengths[np.searchsorted(view.doc_ids, matched_ids)]
    avg_length = index.avg_doc_length or 1.0
    for term in terms:
        doc_ids, tfs = view.postings(term, index)
        if not len(doc_ids):
            continue
        positions = np.searchsorted(sorted_ids, doc_ids)
        valid = positions < len(sorted_ids)
        valid[valid] &= sorted_ids[positions[valid]] == doc_ids[valid]
        indices = order[positions[valid]]
        selected_tfs = tfs[valid]
        df = len(doc_ids)
        if bm25:
            idf = math.log(1.0 + (index.num_docs - df + 0.5) / (df + 0.5))
            norm = 1.0 - b + b * lengths[indices] / avg_length
            weighted = selected_tfs * (k1 + 1.0) / (selected_tfs + k1 * norm)
            scores[indices] += idf * weighted
        else:
            idf = math.log((index.num_docs + 1) / (df + 1)) + 1.0
            scores[indices] += (1.0 + np.log(selected_tfs)) * idf
    return scores


def top_k(
    scores: NDArray[np.float32] | NDArray[np.float64], k: int
) -> NDArray[np.intp]:
    """Descending top-k, keeping original match order for equal scores."""
    if k == 0 or not len(scores):
        return np.empty(0, dtype=np.intp)
    if k >= len(scores):
        candidates = np.arange(len(scores))
    else:
        partition = np.argpartition(scores, -k)[-k:]
        # argpartition does not preserve ties at the boundary.
        candidates = np.flatnonzero(scores >= scores[partition].min())
    return candidates[np.lexsort((candidates, -scores[candidates]))][:k]
