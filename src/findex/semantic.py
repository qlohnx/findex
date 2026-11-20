"""Optional chunk embeddings and reciprocal-rank hybrid retrieval."""

from __future__ import annotations

import importlib
import itertools
import json
import math
import mmap
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from rich.progress import track

from findex.index import Index
from findex.ranking import SearchResult
from findex.vectorized import top_k


class EmbeddingModel(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> NDArray[np.float32]: ...

    def get_sentence_embedding_dimension(self) -> int | None: ...


@lru_cache(maxsize=2)
def load_model(name: str) -> EmbeddingModel:
    try:
        package = importlib.import_module("sentence_transformers")
    except ImportError as exc:
        raise ValueError(
            "Semantic search requires the optional dependency: uv sync --extra semantic"
        ) from exc
    constructor = cast(Callable[[str], EmbeddingModel], package.SentenceTransformer)
    return constructor(name)


def chunk_documents(
    index: Index, *, chunk_words: int = 256, overlap: int = 32
) -> Iterator[tuple[int, str]]:
    if chunk_words < 1 or not 0 <= overlap < chunk_words:
        raise ValueError("chunk_words must be positive and overlap smaller than it")
    step = chunk_words - overlap
    for doc_id in index.doc_meta:
        words = index.doc_texts.get(doc_id, "").split()
        for start in range(0, len(words), step):
            chunk = words[start : start + chunk_words]
            if chunk:
                yield doc_id, " ".join(chunk)
            if start + chunk_words >= len(words):
                break


def embedding_paths(prefix: Path) -> tuple[Path, Path, Path]:
    return (
        Path(f"{prefix}.embeddings.npy"),
        Path(f"{prefix}.doc_ids.npy"),
        Path(f"{prefix}.meta.json"),
    )


def default_prefix(index_path: Path) -> Path:
    return index_path.with_suffix("")


def build_embeddings(
    index: Index,
    prefix: Path,
    *,
    model_name: str = "all-MiniLM-L6-v2",
    chunk_words: int = 256,
    overlap: int = 32,
    batch_size: int = 32,
    model: EmbeddingModel | None = None,
    progress: bool = True,
) -> int:
    """Write normalized float32 vectors and aligned int32 document IDs as .npy."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    count = sum(
        1 for _ in chunk_documents(index, chunk_words=chunk_words, overlap=overlap)
    )
    if not count:
        raise ValueError("index has no stored document text to embed")
    vectors_path, ids_path, meta_path = embedding_paths(prefix)
    if any(path.exists() for path in (vectors_path, ids_path, meta_path)):
        raise FileExistsError(f"embedding output already exists for {prefix}")
    encoder = model or load_model(model_name)
    dimension = encoder.get_sentence_embedding_dimension()
    if dimension is None or dimension < 1:
        raise ValueError("model has no sentence embedding dimension")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Publish metadata last so an interrupted build cannot appear complete.
    with tempfile.TemporaryDirectory(prefix="findex-embed-", dir=prefix.parent) as tmp:
        staged_vectors = Path(tmp) / vectors_path.name
        staged_ids = Path(tmp) / ids_path.name
        vectors = np.lib.format.open_memmap(
            staged_vectors, mode="w+", dtype=np.float32, shape=(count, dimension)
        )
        doc_ids = np.lib.format.open_memmap(
            staged_ids, mode="w+", dtype=np.int32, shape=(count,)
        )
        rows = itertools.batched(
            chunk_documents(index, chunk_words=chunk_words, overlap=overlap), batch_size
        )
        batches = (
            track(rows, total=math.ceil(count / batch_size), description="Embedding")
            if progress
            else rows
        )
        offset = 0
        try:
            for batch in batches:
                encoded = np.asarray(
                    encoder.encode(
                        [text for _, text in batch],
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    ),
                    dtype=np.float32,
                )
                if encoded.shape != (len(batch), dimension):
                    raise ValueError("model returned an unexpected embedding shape")
                norms = np.linalg.norm(encoded, axis=1, keepdims=True)
                vectors[offset : offset + len(batch)] = encoded / np.maximum(
                    norms, 1e-12
                )
                doc_ids[offset : offset + len(batch)] = [doc_id for doc_id, _ in batch]
                offset += len(batch)
        except BaseException:
            # A traceback can retain the memmap objects. Close their mappings
            # explicitly before TemporaryDirectory removes files on Windows.
            cast(mmap.mmap, getattr(vectors, "_mmap")).close()
            cast(mmap.mmap, getattr(doc_ids, "_mmap")).close()
            raise
        vectors.flush()
        doc_ids.flush()
        del vectors, doc_ids
        if any(path.exists() for path in (vectors_path, ids_path, meta_path)):
            raise FileExistsError(f"embedding output already exists for {prefix}")
        staged_vectors.replace(vectors_path)
        staged_ids.replace(ids_path)
        meta_path.write_text(
            json.dumps(
                {
                    "model": model_name,
                    "dimension": dimension,
                    "chunks": count,
                    "documents": index.num_docs,
                    "chunk_words": chunk_words,
                    "overlap": overlap,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return count


@dataclass(frozen=True)
class SemanticStore:
    vectors: NDArray[np.float32]
    doc_ids: NDArray[np.int32]
    model_name: str

    @classmethod
    def load(cls, prefix: Path, index: Index) -> SemanticStore:
        vectors_path, ids_path, meta_path = embedding_paths(prefix)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        vectors = np.load(vectors_path, mmap_mode="r")
        doc_ids = np.load(ids_path, mmap_mode="r")
        if (
            meta["documents"] != index.num_docs
            or vectors.ndim != 2
            or doc_ids.ndim != 1
            or vectors.shape != (meta["chunks"], meta["dimension"])
            or doc_ids.shape != (meta["chunks"],)
            or vectors.dtype != np.float32
            or doc_ids.dtype != np.int32
            or any(int(doc_id) not in index.doc_meta for doc_id in np.unique(doc_ids))
        ):
            raise ValueError("embedding files do not match this index")
        return cls(vectors, doc_ids, str(meta["model"]))

    @property
    def total_documents(self) -> int:
        return len(np.unique(self.doc_ids))


def semantic_search(
    query: str,
    index: Index,
    store: SemanticStore,
    *,
    k: int = 10,
    model: EmbeddingModel | None = None,
) -> list[SearchResult]:
    if k < 0:
        raise ValueError("k must be nonnegative")
    if not query.strip() or not len(store.doc_ids) or k == 0:
        return []
    encoder = model or load_model(store.model_name)
    query_vector = np.asarray(
        encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0],
        dtype=np.float32,
    )
    if query_vector.shape != (store.vectors.shape[1],):
        raise ValueError("query model dimension does not match stored embeddings")
    query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
    similarities = store.vectors @ query_vector
    unique_ids, inverse = np.unique(store.doc_ids, return_inverse=True)
    best = np.full(len(unique_ids), -np.inf, dtype=np.float32)
    np.maximum.at(best, inverse, similarities)
    return [
        SearchResult(
            doc_id=int(unique_ids[pos]),
            score=float(best[pos]),
            title=index.doc_meta[int(unique_ids[pos])].title
            or index.doc_meta[int(unique_ids[pos])].path,
            snippet=index.doc_texts.get(int(unique_ids[pos]), "")[:160],
        )
        for pos in top_k(best, k)
    ]


def rrf_fuse(
    rankings: Sequence[Sequence[SearchResult]], *, k: int = 10, rank_constant: int = 60
) -> list[SearchResult]:
    """Fuse result ranks without mixing incompatible score scales."""
    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")
    scores: dict[int, float] = {}
    rows: dict[int, SearchResult] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            scores[row.doc_id] = scores.get(row.doc_id, 0.0) + 1.0 / (
                rank_constant + rank
            )
            rows.setdefault(row.doc_id, row)
    return [
        SearchResult(doc_id, score, rows[doc_id].title, rows[doc_id].snippet)
        for doc_id, score in sorted(scores.items(), key=lambda pair: -pair[1])[:k]
    ]
