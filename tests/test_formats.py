import pickle

import pytest

from findex.index import DocMeta, InvertedIndex, Posting, build_index
from findex.store import load_index, save_index


def make(positions: bool) -> InvertedIndex:
    pages = [("A", "red green red"), ("B", "green blue")]
    return build_index(pages, source="t", positions=positions)


@pytest.mark.parametrize("suffix", [".pkl", ".json"])
@pytest.mark.parametrize("positions", [False, True])
def test_roundtrip(tmp_path, suffix, positions):
    idx = make(positions)
    path = tmp_path / f"index{suffix}"
    save_index(idx, path)
    loaded = load_index(path)
    assert loaded.positions == positions
    assert loaded.doc_lengths == idx.doc_lengths
    assert loaded.doc_meta == idx.doc_meta
    assert dict(loaded.index) == dict(idx.index)


def test_positions_recorded():
    idx = make(True)
    assert idx.get_postings("red")[0] == Posting(1, 2, (0, 2))


def test_hashable_records():
    assert len({Posting(1, 2), Posting(1, 2), DocMeta("p")}) == 2


def test_postings_sorted_by_doc_id():
    idx = make(False)
    for postings in idx.index.values():
        ids = [p.doc_id for p in postings]
        assert ids == sorted(ids)


def test_load_rejects_wrong_type(tmp_path):
    path = tmp_path / "x.pkl"
    path.write_bytes(pickle.dumps({"not": "an index"}))
    with pytest.raises(TypeError):
        load_index(path)
