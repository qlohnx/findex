from pathlib import Path

from findex.memory_study import VARIANTS, measure, postings_as_pairs, synthetic_docs
from findex.store import load_index, save_index


def _build(index_cls):
    index = index_cls()
    index.add_document(1, "d1.txt", ["python", "search", "python"])
    index.add_document(2, "d2.txt", ["python", "data"])
    index.add_document(3, "d3.txt", ["search", "data", "data"])
    return index


def test_all_variants_give_same_postings():
    expected = {
        "python": [(1, 2), (2, 1)],
        "search": [(1, 1), (3, 1)],
        "data": [(2, 1), (3, 2)],
    }
    for _, cls in VARIANTS:
        index = _build(cls)
        for term, pairs in expected.items():
            assert postings_as_pairs(index, term) == pairs
        assert index.doc_lengths == {1: 3, 2: 2, 3: 3}


def test_all_variants_survive_pickle_roundtrip(tmp_path: Path):
    for i, (_, cls) in enumerate(VARIANTS):
        path = tmp_path / f"index_{i}.pkl"
        save_index(_build(cls), path)
        loaded = load_index(path)
        assert type(loaded) is cls
        assert postings_as_pairs(loaded, "data") == [(2, 1), (3, 2)]


def test_measure_runs_and_variants_agree():
    def make_docs():
        return synthetic_docs(30, vocab=500)

    rows = [measure(label, cls, make_docs, loads=1) for label, cls in VARIANTS]
    assert rows[0].fingerprint == rows[1].fingerprint == rows[2].fingerprint
    assert all(r.peak_mb > 0 and r.file_mb > 0 for r in rows)
