from findex.index import InvertedIndex
from findex.query import And, Not, Or, Term, parse


def test_parser_builds_trees():
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))
    assert parse("a AND NOT b") == And(Term("a"), Not(Term("b")))


def test_parentheses_without_spaces():
    assert parse("a AND(b OR c)") == And(Term("a"), Or(Term("b"), Term("c")))
    assert parse("((a))") == Term("a")


def test_query_operators(tiny_index):
    assert ((Term("python") | Term("java")) & ~Term("rare")).evaluate(tiny_index) == {
        2,
        3,
    }


def test_phrase_evaluation_uses_positions():
    index = InvertedIndex(positions=True)
    index.add_document(1, "a.txt", ["phrase", "term"])
    index.add_document(2, "b.txt", ["term", "phrase"])
    assert parse('"phrase term"').evaluate(index) == {1}
