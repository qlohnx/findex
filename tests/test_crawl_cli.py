import asyncio
import importlib
import json

import httpx
from typer.testing import CliRunner

from findex.cli import _crawl_to_jsonl, app
from findex.crawler import crawl as real_crawl
from findex.crawler.robots import USER_AGENT
from findex.store import load_index


def test_crawl_help_and_invalid_seed(tmp_path) -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["crawl", "--help"])
    assert help_result.exit_code == 0
    assert "--max-pages" in help_result.stdout
    result = runner.invoke(
        app,
        [
            "crawl",
            "file:///etc/passwd",
            "--out",
            str(tmp_path / "crawl.jsonl"),
            "--log",
            str(tmp_path / "crawl.log"),
        ],
    )
    assert result.exit_code == 1
    assert result.stderr.startswith("Error:")
    assert not (tmp_path / "crawl.jsonl").exists()


def test_crawl_writes_jsonl_before_next_fetch_and_index_reads_it(tmp_path) -> None:
    async def scenario() -> None:
        output = tmp_path / "crawl.jsonl"
        second_started = asyncio.Event()
        release_second = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nAllow: /")
            if request.url.path == "/next":
                second_started.set()
                await release_second.wait()
                return httpx.Response(
                    200,
                    text="<title>Next</title><p>Async index</p>",
                    headers={"content-type": "text/html"},
                )
            return httpx.Response(
                200,
                text="<title>Root</title><p>Python search</p><a href='/next'>Next</a>",
                headers={"content-type": "text/html"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            task = asyncio.create_task(
                _crawl_to_jsonl(
                    "https://example.test/",
                    output,
                    max_pages=2,
                    concurrency=1,
                    per_host=1,
                    delay=0,
                    allowed_domains=None,
                    client=client,
                )
            )
            async with asyncio.timeout(2):
                await second_started.wait()
                while not output.exists() or not await asyncio.to_thread(
                    output.read_text, encoding="utf-8"
                ):
                    await asyncio.sleep(0.01)
            assert len(output.read_text(encoding="utf-8").splitlines()) == 1
            release_second.set()
            stats = await task
        assert stats.pages == 2
        rows = [
            json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
        ]
        assert [row["title"] for row in rows] == ["Root", "Next"]
        assert set(rows[0]) == {"url", "title", "text", "fetched_at"}
        index_file = tmp_path / "index.json"
        result = CliRunner().invoke(
            app, ["index", str(output), "--out", str(index_file)]
        )
        assert result.exit_code == 0, result.output
        index = load_index(index_file)
        assert index.num_docs == 2
        assert [index.doc_meta[i].title for i in (1, 2)] == ["Root", "Next"]

    asyncio.run(scenario(), debug=True)


def test_crawl_command_writes_log_and_uses_honest_agent(tmp_path, monkeypatch) -> None:
    cli_module = importlib.import_module("findex.cli")
    requests: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"] == USER_AGENT
        requests.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /")
        if request.url.path == "/":
            return httpx.Response(
                200,
                text="<title>Root</title><a href='https://extra.test/next'>Next</a>",
                headers={"content-type": "text/html"},
            )
        return httpx.Response(
            200,
            text="<title>Next</title>Async content",
            headers={"content-type": "text/html"},
        )

    async def mock_crawl(*args, **kwargs):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            kwargs["client"] = client
            async for page in real_crawl(*args, **kwargs):
                yield page

    monkeypatch.setattr(cli_module, "crawl", mock_crawl)
    output = tmp_path / "crawl.jsonl"
    log = tmp_path / "crawl.log"
    result = CliRunner().invoke(
        app,
        [
            "crawl",
            "https://example.test/",
            "--max-pages",
            "2",
            "--concurrency",
            "2",
            "--delay",
            "0",
            "--allowed-domain",
            "extra.test",
            "--out",
            str(output),
            "--log",
            str(log),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Saved 2 pages" in result.stdout
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    assert "status=200" in log.read_text(encoding="utf-8")
    assert "elapsed=" in log.read_text(encoding="utf-8")
    assert requests == [
        "https://example.test/robots.txt",
        "https://example.test/",
        "https://extra.test/robots.txt",
        "https://extra.test/next",
    ]
