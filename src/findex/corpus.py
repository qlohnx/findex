import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import cast

log = logging.getLogger(__name__)


def iter_crawled_pages(path: Path) -> Iterator[tuple[str, str, str]]:
    """Stream (url, title, text) from the crawler's JSON Lines format."""
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = cast(object, json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid crawl JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"invalid crawl record at {path}:{line_number}")
            row = cast(dict[str, object], value)
            url, title, text = row.get("url"), row.get("title"), row.get("text")
            if (
                not isinstance(url, str)
                or not isinstance(title, str)
                or not isinstance(text, str)
            ):
                raise ValueError(f"invalid crawl record at {path}:{line_number}")
            yield url, title, text


def _local_records(root: Path) -> Iterator[tuple[str, str, str]]:
    files = (
        [root]
        if root.is_file()
        else sorted([*root.rglob("*.txt"), *root.rglob("*.jsonl")])
    )
    for file_path in files:
        if file_path.suffix == ".jsonl":
            yield from iter_crawled_pages(file_path)
            continue
        if file_path.suffix != ".txt":
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            if "\ufffd" in text:
                log.warning("replacement characters in UTF-8 document %s", file_path)
            yield str(file_path), file_path.stem, text
        except OSError as e:
            log.warning("cannot read %s: %s", file_path, e)


def iter_documents(root: Path) -> Iterator[tuple[int, str, str]]:
    """Lazily read text files or crawled JSONL, numbering documents from one."""
    for doc_id, (path, _, text) in enumerate(_local_records(root), start=1):
        yield doc_id, path, text


def iter_local_pages(root: Path) -> Iterator[tuple[str, str]]:
    """Lazily produce (title, text) for the indexer's local input format."""
    for _, title, text in _local_records(root):
        yield title, text
