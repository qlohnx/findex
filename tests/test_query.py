import pytest

from findex.index import InvertedIndex
from findex.search import ENGINES, and_ids, evaluate, not_ids, or_ids


@pytest.fixture
def index() -> InvertedIndex:
    idx = InvertedIndex()
    idx.add_document(1, "d1", ["python", "code", "fast"], title="One")
    idx.add_document(2, "d2", ["python", "search"], title="Two")
    idx.add_document(3, "d3", ["java", "code"], title="Three")
    return idx


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_query_operators(index, engine):
    assert evaluate(index, "python code", engine) == [1]  # неявний AND
    assert evaluate(index, "python OR java", engine) == [1, 2, 3]
    assert evaluate(index, "python NOT fast", engine) == [2]
    assert evaluate(index, "NOT python", engine) == [3]
    assert evaluate(index, "PYTHON", engine) == [1, 2]  # casefold
    assert evaluate(index, "missing", engine) == []


def test_engines_agree_on_random_lists():
    import random

    rng = random.Random(1)
    for _ in range(50):
        a = sorted(rng.sample(range(200), rng.randint(0, 60)))
        b = sorted(rng.sample(range(200), rng.randint(0, 60)))
        for op in ("AND", "OR", "NOT"):
            assert ENGINES["merge"][op](a, b) == ENGINES["set"][op](a, b)


def test_merge_basic():
    assert and_ids([1, 3, 5], [3, 4, 5]) == [3, 5]
    assert or_ids([1, 3], [2, 3, 9]) == [1, 2, 3, 9]
    assert not_ids([1, 2, 3], [2]) == [1, 3]
