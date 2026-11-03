import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.index import build_index
from findex.query import And, Not, Or, Phrase, Query, Term, parse
from findex.search import and_ids, not_ids, or_ids
from findex.store import load_index, save_index
from findex.tokenize import tokenize


@given(st.text())
def test_tokenizer_never_emits_empty_tokens(text):
    assert all(token for token in tokenize(text))


@given(st.text())
def test_tokenizer_is_idempotent(text):
    tokens = list(tokenize(text))
    assert list(tokenize(" ".join(tokens))) == tokens


@given(
    st.sets(st.integers(min_value=0, max_value=10000)),
    st.sets(st.integers(min_value=0, max_value=10000)),
)
def test_merges_agree_with_set_oracle(a, b):
    left, right = sorted(a), sorted(b)
    assert and_ids(left, right) == sorted(a & b)
    assert or_ids(left, right) == sorted(a | b)
    assert not_ids(left, right) == sorted(a - b)


@pytest.mark.slow
@given(
    st.lists(st.text(max_size=100), max_size=8),
    st.booleans(),
    st.sampled_from(["json", "pickle"]),
)
def test_index_roundtrip(texts, positions, fmt):
    original = build_index(
        ((str(i), text) for i, text in enumerate(texts)), positions=positions
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "index"
        save_index(original, path, fmt=fmt)
        restored = load_index(path, fmt=fmt)
    assert dict(restored) == dict(original)
    assert restored.doc_meta == original.doc_meta
    assert restored.doc_lengths == original.doc_lengths
    assert restored.doc_texts == original.doc_texts
    assert restored.positions == original.positions


def render(query: Query) -> str:
    if isinstance(query, Term):
        return query.value
    if isinstance(query, Phrase):
        return '"' + " ".join(query.terms()) + '"'
    if isinstance(query, Not):
        return f"NOT ({render(query.child)})"
    if isinstance(query, (And, Or)):
        operator = "AND" if isinstance(query, And) else "OR"
        return f"({render(query.left)} {operator} {render(query.right)})"
    raise AssertionError(f"unexpected query: {query!r}")


trees = st.recursive(
    st.one_of(
        st.sampled_from(["python", "java", "café", "привіт"]).map(Term),
        st.just(Phrase("search engine")),
    ),
    lambda children: st.one_of(
        st.builds(And, children, children),
        st.builds(Or, children, children),
        st.builds(Not, children),
    ),
    max_leaves=12,
)


@given(trees)
def test_query_tree_render_parse_roundtrip(query):
    assert parse(render(query)) == query
