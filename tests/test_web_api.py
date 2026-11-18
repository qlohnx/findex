from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient
from pydantic import ValidationError
from typer.testing import CliRunner

from findex.cli import app as cli_app
from findex.index import Index, InvertedIndex, build_index
from findex.store import save_index
from findex.web import create_app
from findex.web.app import get_index
from findex.web.config import Settings


@pytest.fixture
def client(tiny_index: InvertedIndex, tmp_path: Path) -> Iterator[TestClient]:
    calls: list[Path] = []
    path = tmp_path / "index.json"

    def load(requested: Path) -> Index:
        calls.append(requested)
        return tiny_index

    app = create_app(settings=Settings(index_path=path), index_loader=load)
    app.dependency_overrides[get_index] = lambda: tiny_index
    with TestClient(app) as test_client:
        assert calls == [path]
        yield test_client
    assert calls == [path]
    assert tiny_index._closed


def test_search_pagination_and_models(client: TestClient) -> None:
    first = client.get("/search", params={"q": "python", "k": 1, "page": 1})
    second = client.get("/search", params={"q": "python", "k": 1, "page": 2})
    assert first.status_code == second.status_code == 200
    assert first.json()["total"] == second.json()["total"] == 2
    assert first.json()["results"][0]["doc_id"] != second.json()["results"][0]["doc_id"]
    assert first.json()["results"][0]["score"] > 0
    assert first.json()["results"][0]["snippet"]
    assert first.json()["took_ms"] >= 0
    assert (
        client.get("/search", params={"q": "python", "page": 3, "k": 1}).json()[
            "results"
        ]
        == []
    )
    assert (
        client.get("/search", params={"q": "python", "scorer": "tfidf"}).status_code
        == 200
    )


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "python", "k": "abc"},
        {"q": "python", "k": 0},
        {"q": "python", "k": 101},
        {"q": "python", "page": 0},
        {"q": "python", "scorer": "wrong"},
        {"q": "("},
    ],
)
def test_search_validation(client: TestClient, params: dict[str, object]) -> None:
    response = client.get("/search", params=params)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_document_and_openapi_docs(client: TestClient) -> None:
    response = client.get("/docs/1")
    assert response.status_code == 200
    assert response.json()["title"] == "Python"
    assert "search engine" in response.json()["text"]
    assert client.get("/docs/999").status_code == 404
    assert client.get("/docs/999").json() == {"detail": "Document not found"}
    assert client.get("/docs").status_code == 200
    assert "/search" in client.get("/openapi.json").json()["paths"]


def test_stats_and_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    stats = client.get("/stats")
    assert stats.status_code == 200
    assert stats.json()["documents"] == 4
    assert stats.json()["vocabulary"] > 0
    assert stats.json()["postings"] > 0
    assert stats.json()["index_bytes"] is None
    assert stats.json()["uptime_seconds"] >= 0
    assert "X-Request-ID" in stats.headers


def test_web_ui_search_pagination_and_document(client: TestClient) -> None:
    home = client.get("/")
    assert home.status_code == 200
    assert '<form action="/" method="get"' in home.text
    results = client.get("/", params={"q": "python", "k": 1})
    assert results.status_code == 200
    assert "<mark>Python</mark>" in results.text
    assert "Next →" in results.text
    assert "page=2" in results.text
    second_page = client.get("/", params={"q": "python", "k": 1, "page": 2})
    assert "Previous" in second_page.text
    assert client.get("/doc/1").status_code == 200
    assert "Python search engine" in client.get("/doc/1").text
    assert client.get("/doc/999").status_code == 404
    css = client.get("/static/style.css")
    assert css.status_code == 200
    assert ".search-form" in css.text


def test_web_ui_escapes_document_text(tmp_path: Path) -> None:
    index = build_index([("Unsafe", "<script>alert(1)</script> python")])
    app = create_app(
        settings=Settings(index_path=tmp_path / "unused.json"),
        index_loader=lambda _: index,
    )
    with TestClient(app) as client:
        search_page = client.get("/", params={"q": "python"})
        document_page = client.get("/doc/1")
    assert "<script>" not in search_page.text
    assert "&lt;script&gt;" in search_page.text
    assert "<script>" not in document_page.text
    assert "&lt;script&gt;" in document_page.text


def test_health_is_unavailable_before_lifespan(tmp_path: Path) -> None:
    app = create_app(settings=Settings(index_path=tmp_path / "missing.json"))
    response = TestClient(app).get("/health")
    assert response.status_code == 503
    assert response.json() == {"detail": "Index is not loaded"}


def test_missing_index_fails_at_startup(tmp_path: Path) -> None:
    app = create_app(settings=Settings(index_path=tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError), TestClient(app):
        pass


def test_real_saved_index_loads_on_startup(
    tiny_index: InvertedIndex, tmp_path: Path
) -> None:
    path = tmp_path / "index.json"
    save_index(tiny_index, path)
    app = create_app(settings=Settings(index_path=path))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/search", params={"q": "python"}).json()["total"] == 2
        assert client.get("/stats").json()["index_bytes"] == path.stat().st_size


def test_unexpected_error_is_clean_json(client: TestClient) -> None:
    def fail() -> Index:
        raise RuntimeError("private details")

    client.app.dependency_overrides[get_index] = fail
    response = client.get("/search", params={"q": "python"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "X-Request-ID" in response.headers


def test_serve_command_uses_environment_and_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "index.json"
    monkeypatch.setenv("INDEX_PATH", str(path))
    monkeypatch.setenv("PORT", "8173")
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_run(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(uvicorn, "run", fake_run)
    runner = CliRunner()
    result = runner.invoke(cli_app, ["serve", "--workers", "2"])
    assert result.exit_code == 0, result.output
    assert calls == [
        (
            ("findex.web:app",),
            {"host": "127.0.0.1", "port": 8173, "workers": 2, "log_level": "info"},
        )
    ]
    calls.clear()
    result = runner.invoke(cli_app, ["serve", "--port", "9000", "--host", "0.0.0.0"])
    assert result.exit_code == 0, result.output
    assert calls[0][1]["port"] == 9000
    assert calls[0][1]["host"] == "0.0.0.0"


def test_settings_are_validated_from_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "index.json"))
    monkeypatch.setenv("PORT", "8123")
    assert Settings().port == 8123
    monkeypatch.setenv("PORT", "not-a-port")
    with pytest.raises(ValidationError):
        Settings()
    monkeypatch.setenv("PORT", "8123")
    monkeypatch.setenv("LOG_LEVEL", "verbose")
    with pytest.raises(ValidationError):
        Settings()
