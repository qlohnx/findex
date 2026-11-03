"""Installed CLI for the search engine from Labs 1–3."""

import itertools
import json
import logging
import pickle
import re
from collections import Counter
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal, TypedDict
from xml.etree.ElementTree import ParseError

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich.text import Text

from findex.corpus import iter_documents
from findex.index import build_index
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
        elif corpus.suffix == ".txt":
            yield corpus.stem, corpus.read_text(encoding="utf-8")
        else:
            raise ValueError("corpus must be a .txt file, .xml.bz2 dump, or directory")
        return
    dumps = sorted(corpus.glob("*.xml.bz2"))
    if dumps:
        for dump in dumps:
            yield from iter_pages(dump)
    else:
        found = False
        for _, path, text in iter_documents(corpus):
            found = True
            yield Path(path).stem, text
        if not found:
            raise ValueError(f"no .txt or .xml.bz2 corpus files in {corpus}")


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
) -> None:
    """Build an index in one streaming pass, with progress on stderr."""
    with _expected_errors():
        pages = _pages(corpus)
        if limit is not None:
            pages = itertools.islice(pages, limit)
        console = Console(stderr=True)
        progress = track(
            pages,
            description="Indexing",
            total=limit,
            console=console,
            disable=not console.is_terminal,
        )
        index = build_index(progress, source=str(corpus), positions=positions)
        save_index(index, out)
        typer.echo(
            f"Indexed {index.num_docs} documents, {len(index)} terms. Saved to {out}"
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


if __name__ == "__main__":
    app()
