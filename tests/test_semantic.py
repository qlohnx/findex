"""Semantic and hybrid retrieval without downloading a model in CI."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from findex.index import build_index
from findex.ranking import search
from findex.semantic import (
    SemanticStore,
    build_embeddings,
    chunk_documents,
    embedding_paths,
    rrf_fuse,
    semantic_search,
)
from findex.web import create_app
from findex.web.config import Settings


class TopicModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, sentences, **kwargs):
        rows = []
        for sentence in sentences:
            lower = sentence.casefold()
            rows.append(
                [
                    int("cat" in lower or "feline" in lower),
                    int("market" in lower or "stocks" in lower),
                    int("dog" in lower or "canine" in lower),
                ]
            )
        return np.array(rows, dtype=np.float32)


@pytest.fixture
def semantic_fixture(tmp_path: Path):
    index = build_index(
        [
            ("Cat", "The cat rested on the mat. The cat slept."),
            ("Stocks", "The stock market fell and then recovered."),
            ("Dog", "The dog ran through the park."),
        ]
    )
    prefix = tmp_path / "index"
    count = build_embeddings(
        index,
        prefix,
        chunk_words=4,
        overlap=1,
        batch_size=2,
        model=TopicModel(),
        progress=False,
    )
    store = SemanticStore.load(prefix, index)
    assert count > index.num_docs
    return index, store, prefix


def test_chunk_overlap_and_validation(semantic_fixture):
    index, _, _ = semantic_fixture
    chunks = list(chunk_documents(index, chunk_words=4, overlap=1))
    assert chunks[0][0] == chunks[1][0] == 1
    assert chunks[0][1].split()[-1] == chunks[1][1].split()[0]
    with pytest.raises(ValueError):
        list(chunk_documents(index, chunk_words=4, overlap=4))


def test_semantic_finds_paraphrase_and_rrf(semantic_fixture):
    index, store, _ = semantic_fixture
    assert search("feline", index) == []
    semantic = semantic_search("feline", index, store, model=TopicModel())
    assert semantic[0].doc_id == 1
    assert semantic[0].score == pytest.approx(1.0)
    keyword = search("cat", index)
    fused = rrf_fuse([keyword, semantic])
    assert fused[0].doc_id == 1
    assert fused[0].score == pytest.approx(2 / 61)


def test_store_rejects_mismatched_index(semantic_fixture):
    _, _, prefix = semantic_fixture
    other = build_index([("Only", "cat")])
    with pytest.raises(ValueError, match="do not match"):
        SemanticStore.load(prefix, other)


def test_failed_embedding_build_does_not_publish_partial_store(tmp_path: Path):
    class ExplodingModel(TopicModel):
        calls = 0

        def encode(self, sentences, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("model interrupted")
            return super().encode(sentences, **kwargs)

    index = build_index([("Cat", "cat cat cat cat"), ("Dog", "dog dog dog dog")])
    prefix = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="interrupted"):
        build_embeddings(
            index,
            prefix,
            chunk_words=2,
            overlap=0,
            batch_size=1,
            model=ExplodingModel(),
            progress=False,
        )
    assert not any(path.exists() for path in embedding_paths(prefix))
    assert not list(tmp_path.glob("findex-embed-*"))


def test_api_and_ui_modes(semantic_fixture, tmp_path: Path, monkeypatch):
    index, _, prefix = semantic_fixture
    import findex.semantic as semantic_module

    monkeypatch.setattr(semantic_module, "load_model", lambda _: TopicModel())
    app = create_app(
        settings=Settings(index_path=prefix.with_suffix(".json")),
        index_loader=lambda _: index,
    )
    with TestClient(app) as client:
        semantic = client.get("/search", params={"q": "feline", "mode": "semantic"})
        hybrid = client.get("/search", params={"q": "feline", "mode": "hybrid"})
        second = client.get(
            "/search", params={"q": "feline", "mode": "hybrid", "page": 2, "k": 1}
        )
        ui = client.get("/", params={"q": "feline", "mode": "semantic"})
        assert semantic.status_code == hybrid.status_code == ui.status_code == 200
        assert semantic.json()["results"][0]["doc_id"] == 1
        assert hybrid.json()["results"][0]["doc_id"] == 1
        assert second.json()["total"] == hybrid.json()["total"] == 3
        assert second.json()["results"][0]["doc_id"] != 1
        assert 'option value="semantic" selected' in ui.text
        assert (
            client.get("/search", params={"q": "cat", "mode": "invalid"}).status_code
            == 422
        )


def test_api_requires_embeddings(tmp_path: Path):
    index = build_index([("Cat", "The cat sleeps.")])
    app = create_app(
        settings=Settings(index_path=tmp_path / "index.json"),
        index_loader=lambda _: index,
    )
    with TestClient(app) as client:
        response = client.get("/search", params={"q": "cat", "mode": "semantic"})
        assert response.status_code == 503
