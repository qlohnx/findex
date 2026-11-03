import bz2
from pathlib import Path

import pytest

from findex.index import InvertedIndex, build_index


@pytest.fixture
def tiny_corpus() -> list[tuple[str, str]]:
    return [
        ("Python", "Python search engine. Rare python!"),
        ("Java", "Java search engine."),
        ("Unicode", "Привіт світ. Python Straße café."),
        ("Empty", ""),
    ]


@pytest.fixture
def tiny_index(tiny_corpus: list[tuple[str, str]]) -> InvertedIndex:
    # A fresh mutable index per test prevents order-dependent results.
    return build_index(iter(tiny_corpus), positions=True)


@pytest.fixture
def text_corpus(tmp_path: Path, tiny_corpus: list[tuple[str, str]]) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i, (title, text) in enumerate(tiny_corpus):
        (corpus / f"{i}-{title}.txt").write_text(text, encoding="utf-8")
    return corpus


@pytest.fixture
def wiki_dump(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.xml.bz2"
    xml = '<mediawiki xmlns="urn:wiki"><page><title>Python</title><revision><text>Python search</text></revision></page><page><title>Java</title><revision><text>Java search</text></revision></page></mediawiki>'
    path.write_bytes(bz2.compress(xml.encode("utf-8")))
    return path
