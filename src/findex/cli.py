"""Installed search CLI with streaming input and parallel indexing options."""

import asyncio
import itertools
import json
import logging
import pickle
import re
import tempfile
import time
from collections import Counter
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal, TextIO, TypedDict
from urllib.parse import urlsplit
from xml.etree.ElementTree import ParseError

import httpx
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.live import Live
from rich.progress import track
from rich.table import Table
from rich.text import Text

from findex.corpus import iter_local_pages
from findex.crawler import CrawlStats, Page, crawl
from findex.crawler.urls import canonical_url
from findex.parallel import build_from_chunks, stage_pages
from findex.ranking import BM25, Scorer, TfIdf
from findex.ranking import search as ranked_search
from findex.stats import iter_pages
from findex.store import open_index, save_index

app = typer.Typer(
    no_args_is_help=True, help="Index a corpus and search with BM25 or TF-IDF."
)
log = logging.getLogger(__name__)


class ResultRow(TypedDict):
    doc_id: int
    score: float
    title: str
    snippet: str


@app.callback()
def configure(
    ctx: typer.Context,
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", count=True, help="-v: INFO; -vv: DEBUG.")
    ] = 0,
) -> None:
    """Configure diagnostics once, leaving library imports free of side effects."""
    logger = logging.getLogger("findex")
    level = (
        logging.DEBUG if verbose >= 2 else logging.INFO if verbose else logging.WARNING
    )
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    old_level, old_propagate = logger.level, logger.propagate
    logger.setLevel(level)
    logger.propagate = False
    logger.addHandler(handler)

    def restore() -> None:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(old_level)
        logger.propagate = old_propagate

    # Also makes repeated CliRunner invocations independent of each other.
    ctx.call_on_close(restore)


@contextmanager
def _expected_errors() -> Generator[None]:
    try:
        yield
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        EOFError,
        pickle.UnpicklingError,
        ParseError,
        ValidationError,
    ) as exc:
        message = " ".join(str(exc).splitlines())
        typer.echo(f"Error: {message}", err=True)
        raise typer.Exit(code=1) from None


def _pages(corpus: Path) -> Iterator[tuple[str, str]]:
    if not corpus.exists():
        raise FileNotFoundError(f"corpus does not exist: {corpus}")
    if corpus.is_file():
        if corpus.name.endswith(".xml.bz2"):
            yield from iter_pages(corpus)
        elif corpus.suffix in (".txt", ".jsonl"):
            yield from iter_local_pages(corpus)
        else:
            raise ValueError(
                "corpus must be a .txt file, .jsonl crawl, .xml.bz2 dump, or directory"
            )
        return
    dumps = sorted(corpus.glob("*.xml.bz2"))
    if dumps:
        for dump in dumps:
            yield from iter_pages(dump)
    else:
        found = False
        for title, text in iter_local_pages(corpus):
            found = True
            yield title, text
        if not found:
            raise ValueError(f"no .txt, .jsonl, or .xml.bz2 corpus files in {corpus}")


@app.command("index")
def index_command(
    corpus: Annotated[
        Path, typer.Argument(help="Text corpus directory or Wikipedia XML-bz2 dump.")
    ],
    out: Annotated[Path, typer.Option("--out", help="Index file (.json or pickle).")],
    positions: Annotated[
        bool, typer.Option("--positions", help="Store positions for phrase queries.")
    ] = False,
    limit: Annotated[
        int | None, typer.Option("--limit", min=1, help="Maximum documents to index.")
    ] = None,
    workers: Annotated[
        int, typer.Option("--workers", min=1, help="Number of indexing workers.")
    ] = 1,
    executor: Annotated[
        Literal["serial", "threads", "processes"],
        typer.Option("--executor", help="serial, threads, or processes."),
    ] = "serial",
) -> None:
    """Stage a streaming corpus and build its index with a chosen executor."""
    with _expected_errors():
        pages = _pages(corpus)
        if limit is not None:
            pages = itertools.islice(pages, limit)
        console = Console(stderr=True)
        progress = track(
            pages,
            description="Staging corpus",
            total=limit,
            console=console,
            disable=not console.is_terminal,
        )
        with tempfile.TemporaryDirectory(prefix="findex-chunks-") as temporary:
            stage_start = time.perf_counter()
            chunks = stage_pages(progress, str(corpus), Path(temporary))
            log.info(
                "staged %d chunks in %.3fs",
                len(chunks),
                time.perf_counter() - stage_start,
            )
            index, timing = build_from_chunks(
                chunks, positions=positions, executor=executor, workers=workers
            )
        log.info(
            "index build %.3fs (merge %.3fs), executor=%s workers=%d",
            timing.wall_seconds,
            timing.merge_seconds,
            executor,
            workers,
        )
        save_index(index, out)
        typer.echo(
            f"Indexed {index.num_docs} documents, {len(index)} terms. Saved to {out}"
        )


def _crawl_display(stats: CrawlStats, elapsed: float) -> Table:
    table = Table(title="Crawl progress")
    table.add_column("Pages")
    table.add_column("Pages/s")
    table.add_column("In flight")
    table.add_column("Errors")
    table.add_column("Queue")
    table.add_row(
        str(stats.pages),
        f"{stats.pages / elapsed:.1f}" if elapsed > 0 else "0.0",
        str(stats.in_flight),
        str(stats.errors),
        str(stats.queue_depth),
    )
    return table


class _CrawlProgress:
    """Render current counters on every Rich refresh, even during slow requests."""

    def __init__(self, stats: CrawlStats, started: float) -> None:
        self.stats = stats
        self.started = started

    def __rich__(self) -> Table:
        return _crawl_display(self.stats, time.perf_counter() - self.started)


def _write_page(stream: TextIO, page: Page) -> None:
    stream.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
    stream.flush()


async def _crawl_to_jsonl(
    seed_url: str,
    out: Path,
    *,
    max_pages: int,
    concurrency: int,
    per_host: int,
    delay: float,
    allowed_domains: list[str] | None,
    client: httpx.AsyncClient | None = None,
) -> CrawlStats:
    stats = CrawlStats()
    started = time.perf_counter()
    console = Console(stderr=True)
    stream = await asyncio.to_thread(out.open, "w", encoding="utf-8")
    try:
        with Live(
            _CrawlProgress(stats, started),
            console=console,
            refresh_per_second=4,
            transient=not console.is_terminal,
        ) as display:
            async for page in crawl(
                [seed_url],
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                per_host_delay=delay,
                allowed_domains=allowed_domains,
                client=client,
                stats=stats,
            ):
                await asyncio.to_thread(_write_page, stream, page)
                display.refresh()
            display.refresh()
    finally:
        await asyncio.to_thread(stream.close)
    return stats


@app.command("crawl")
def crawl_command(
    seed_url: Annotated[str, typer.Argument(help="HTTP(S) seed URL to crawl.")],
    max_pages: Annotated[
        int, typer.Option("--max-pages", min=1, help="Maximum saved pages.")
    ] = 500,
    concurrency: Annotated[
        int, typer.Option("--concurrency", min=1, help="Global worker count.")
    ] = 10,
    per_host: Annotated[
        int, typer.Option("--per-host", min=1, help="Concurrent requests per host.")
    ] = 2,
    out: Annotated[
        Path, typer.Option("--out", help="Streaming JSON Lines output file.")
    ] = Path("data/crawl.jsonl"),
    log_path: Annotated[
        Path, typer.Option("--log", help="Per-URL crawl status log.")
    ] = Path("crawl.log"),
    delay: Annotated[
        float,
        typer.Option("--delay", min=0, help="Minimum seconds between starts per host."),
    ] = 0.1,
    allowed_domain: Annotated[
        list[str],
        typer.Option("--allowed-domain", help="Additional allowed domain; repeatable."),
    ] = [],
) -> None:
    """Polite async crawl; save pages as an indexable JSONL corpus."""
    with _expected_errors():
        normalized_seed = canonical_url(seed_url)
        if normalized_seed is None:
            raise ValueError("seed must be a valid HTTP(S) URL")
        seed_host = urlsplit(normalized_seed).hostname
        assert seed_host is not None
        out.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        crawler_logger = logging.getLogger("findex.crawler")
        previous_level = crawler_logger.level
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        crawler_logger.setLevel(logging.INFO)
        crawler_logger.addHandler(handler)
        try:
            stats = asyncio.run(
                _crawl_to_jsonl(
                    seed_url,
                    out,
                    max_pages=max_pages,
                    concurrency=concurrency,
                    per_host=per_host,
                    delay=delay,
                    allowed_domains=[seed_host, *allowed_domain]
                    if allowed_domain
                    else None,
                ),
                debug=True,
            )
        finally:
            crawler_logger.removeHandler(handler)
            handler.close()
            crawler_logger.setLevel(previous_level)
        typer.echo(
            f"Saved {stats.pages} pages to {out}; {stats.errors} errors. Log: {log_path}"
        )


def _highlight(snippet: str) -> Text:
    text = Text()
    for i, part in enumerate(re.split(r"\*\*(.*?)\*\*", snippet)):
        text.append(part, style="bold yellow" if i % 2 else "")
    return text


@app.command("search")
def search_command(
    index_path: Annotated[Path, typer.Argument(help="Saved index file.")],
    query: Annotated[
        str,
        typer.Argument(
            help="Terms, Boolean operators, parentheses, or quoted phrases."
        ),
    ],
    k: Annotated[int, typer.Option("--k", min=1, help="Maximum results.")] = 10,
    scorer: Annotated[
        Literal["bm25", "tfidf"], typer.Option("--scorer", help="Ranking algorithm.")
    ] = "bm25",
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit one JSON object per result.")
    ] = False,
) -> None:
    """Search a saved index and show ranked results with highlighted snippets."""
    with _expected_errors():
        ranking: Scorer = BM25() if scorer == "bm25" else TfIdf()
        with open_index(index_path) as index:
            results = ranked_search(query, index, ranking, k=k)
        log.info("returned %d results using %s", len(results), scorer)
        if json_output:
            for result in results:
                row: ResultRow = {
                    "doc_id": result.doc_id,
                    "score": result.score,
                    "title": result.title,
                    "snippet": result.snippet,
                }
                typer.echo(json.dumps(row, ensure_ascii=False))
            return
        table = Table(title=f"Search results ({scorer})")
        for column in ("Rank", "Document", "Score", "Title", "Snippet"):
            table.add_column(column)
        for rank, result in enumerate(results, start=1):
            table.add_row(
                str(rank),
                str(result.doc_id),
                f"{result.score:.4f}",
                Text(result.title),
                _highlight(result.snippet),
            )
        Console().print(table)


@app.command("stats")
def stats_command(
    index_path: Annotated[Path, typer.Argument(help="Saved index file.")],
) -> None:
    """Show document, token, vocabulary, and top-50 term statistics."""
    with _expected_errors():
        with open_index(index_path) as index:
            table = Table(title="Index statistics")
            table.add_column("Metric")
            table.add_column("Value", justify="right")
            table.add_row("Documents", str(index.num_docs))
            table.add_row("Total tokens", str(sum(index.doc_lengths.values())))
            table.add_row("Vocabulary size", str(len(index)))
            table.add_row("Average document length", f"{index.avg_doc_length:.2f}")
            table.add_row("Positions", str(index.positions))
            counts = Counter({term: sum(p.tf for p in index[term]) for term in index})
            terms = Table(title="Top-50 terms")
            terms.add_column("Term")
            terms.add_column("Count", justify="right")
            for term, count in counts.most_common(50):
                terms.add_row(Text(term), str(count))
        console = Console()
        console.print(table)
        console.print(terms)


@app.command("serve")
def serve_command(
    port: Annotated[
        int | None,
        typer.Option("--port", min=1, max=65535, help="HTTP port (defaults to PORT)."),
    ] = None,
    workers: Annotated[
        int, typer.Option("--workers", min=1, help="Uvicorn worker processes.")
    ] = 1,
    host: Annotated[
        str, typer.Option("--host", help="Bind address; use 0.0.0.0 in containers.")
    ] = "127.0.0.1",
) -> None:
    """Serve the search API and web UI using INDEX_PATH from the environment."""
    import uvicorn

    from findex.web.config import Settings

    with _expected_errors():
        settings = Settings()  # pyright: ignore[reportCallIssue]
        uvicorn.run(
            "findex.web:app",
            host=host,
            port=port if port is not None else settings.port,
            workers=workers,
            log_level=settings.log_level,
        )


if __name__ == "__main__":
    app()
