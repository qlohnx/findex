from findex.index import InvertedIndex
from findex.query import And, Not, Or, Term, parse


def test_parser_builds_trees():
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))
    assert parse("a AND NOT b") == And(Term("a"), Not(Term("b")))


def test_phrase_evaluation_uses_positions():
    index = InvertedIndex(positions=True)
    index.add_document(1, "a.txt", ["phrase", "term"])
    index.add_document(2, "b.txt", ["term", "phrase"])
    assert parse('"phrase term"').evaluate(index) == {1}
