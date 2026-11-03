import inspect
import logging
from pathlib import Path

import pytest

from findex.corpus import iter_documents as iter_text_documents
from findex.stats import collect_stats, iter_documents, iter_pages
from findex.stats_eager import collect_stats_eager
from findex.tokenize import tokenize


def test_streaming_pipeline_uses_generators():
    assert inspect.isgeneratorfunction(iter_documents)
    assert inspect.isgeneratorfunction(iter_pages)
    assert inspect.isgeneratorfunction(iter_text_documents)
    assert inspect.isgeneratorfunction(tokenize)


@pytest.mark.parametrize(
    "limit, documents, tokens", [(None, 2, 4), (1, 1, 2), (0, 0, 0)]
)
def test_eager_and_lazy_stats_agree(wiki_dump, limit, documents, tokens):
    lazy = collect_stats(wiki_dump.parent, limit=limit)
    eager = collect_stats_eager(wiki_dump.parent, limit=limit)
    assert lazy["doc_count"] == eager["doc_count"] == documents
    assert lazy["total_tokens"] == eager["total_tokens"] == tokens
    assert lazy["vocab_size"] == eager["vocab_size"]
    assert sum(count for _, count in lazy["top_50"]) == tokens


def test_missing_dump_logs_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert collect_stats(tmp_path)["doc_count"] == 0
        assert collect_stats_eager(tmp_path)["doc_count"] == 0
    assert "no .xml.bz2 files" in caplog.text


def test_text_intake_logs_decoding_problem(tmp_path, caplog):
    (tmp_path / "bad.txt").write_bytes(b"hello \xff world")
    with caplog.at_level(logging.WARNING):
        documents = list(iter_text_documents(tmp_path))
    assert len(documents) == 1
    assert "replacement characters" in caplog.text


def test_text_intake_skips_unreadable_file(tmp_path, monkeypatch, caplog):
    (tmp_path / "a.txt").write_text("text", encoding="utf-8")

    def unreadable(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", unreadable)
    with caplog.at_level(logging.WARNING):
        assert list(iter_text_documents(tmp_path)) == []
    assert "cannot read" in caplog.text
