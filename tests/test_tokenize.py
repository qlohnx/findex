import pytest

from findex.tokenize import tokenize


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Straße STRASSE", ["strasse", "strasse"]),
        ("Σ σ ς", ["σ", "σ", "σ"]),
        ("中文 日本語", ["中文", "日本語"]),
        ("п’ять don't", ["п’ять", "don't"]),
        ("🙂 — ...", []),
        ("CAFÉ cafe\u0301", ["café", "café"]),
    ],
)
def test_tokenize_unicode_table(text, expected):
    assert list(tokenize(text)) == expected


def test_tokenize_empty_string():
    assert list(tokenize("")) == []


def test_tokenize_mixed_case():
    assert list(tokenize("Hello World PYTHON")) == ["hello", "world", "python"]


def test_tokenize_cyrillic():
    assert list(tokenize("Привіт Світ 2026")) == ["привіт", "світ", "2026"]


def test_tokenize_combining_mark():
    # 'e' + комбінований гострий наголос (U+0301) нормалізується через NFC в 'é' (U+00E9)
    text = "e\u0301lan"
    tokens = list(tokenize(text))
    assert len(tokens) == 1
    assert tokens[0] == "\u00e9lan"


def test_tokenize_punctuation_and_apostrophes():
    assert list(tokenize("Hello, world! It's test... ---")) == [
        "hello",
        "world",
        "it's",
        "test",
    ]


def test_tokenize_hyphens_and_digits():
    assert list(tokenize("state-of-the-art v2.0")) == [
        "state",
        "of",
        "the",
        "art",
        "v2",
        "0",
    ]
