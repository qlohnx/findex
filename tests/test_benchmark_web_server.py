"""Verify that the benchmark-only async route uses the same search semantics."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.index import InvertedIndex
from findex.store import save_index
from findex.web.benchmark import app


def test_sync_and_async_benchmark_routes_match(
    tiny_index: InvertedIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "index.json"
    save_index(tiny_index, path)
    monkeypatch.setenv("INDEX_PATH", str(path))
    with TestClient(app) as client:
        sync = client.get("/search", params={"q": "python", "k": 1})
        async_ = client.get("/search-async", params={"q": "python", "k": 1})
        assert sync.status_code == async_.status_code == 200
        assert sync.json()["total"] == async_.json()["total"]
        assert sync.json()["results"] == async_.json()["results"]
        assert (
            client.get("/search-async", params={"q": "python", "k": 0}).status_code
            == 422
        )
