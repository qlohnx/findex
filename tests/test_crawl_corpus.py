import json

import pytest

from findex.corpus import iter_crawled_pages, iter_documents


def test_crawl_jsonl_is_streamed_into_corpus(tmp_path) -> None:
    path = tmp_path / "crawl.jsonl"
    rows = [
        {"url": "https://example.test/a", "title": "Alpha", "text": "Python search"},
        {"url": "https://example.test/b", "title": "Beta", "text": "Async crawler"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert list(iter_crawled_pages(path)) == [
        ("https://example.test/a", "Alpha", "Python search"),
        ("https://example.test/b", "Beta", "Async crawler"),
    ]
    assert list(iter_documents(path)) == [
        (1, "https://example.test/a", "Python search"),
        (2, "https://example.test/b", "Async crawler"),
    ]


@pytest.mark.parametrize("line", ["not json", "[]", '{"url": "x", "text": "y"}'])
def test_invalid_crawl_jsonl_is_rejected(tmp_path, line) -> None:
    path = tmp_path / "crawl.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid crawl"):
        list(iter_documents(path))
