"""The same document stream must produce identical bytes with all executors."""

import pickle
from pathlib import Path

import pytest

from findex.index import build_index
from findex.parallel import (
    WorkerBuildError,
    build_from_chunks,
    build_partial,
    merge,
    stage_pages,
)
from findex.store import load_index, save_index


@pytest.mark.parametrize("positions", [False, True])
@pytest.mark.parametrize("suffix", ["json", "pkl"])
def test_parallel_bytes_match_legacy(
    tmp_path: Path, tiny_corpus: list[tuple[str, str]], positions: bool, suffix: str
) -> None:
    pages = tiny_corpus * 3
    chunks = stage_pages(pages, "corpus", tmp_path / "chunks", chunk_size=2)
    legacy = build_index(pages, positions=positions)
    expected = tmp_path / f"legacy.{suffix}"
    save_index(legacy, expected)
    assert (
        merge([build_partial(chunks, positions)], positions).doc_meta == legacy.doc_meta
    )
    serial_bytes = b""
    for executor in ("serial", "threads", "processes"):
        actual = tmp_path / f"{executor}.{suffix}"
        index, timing = build_from_chunks(
            chunks, positions=positions, executor=executor, workers=2
        )
        save_index(index, actual)
        assert load_index(actual).index == legacy.index
        assert load_index(actual).doc_meta == legacy.doc_meta
        if suffix == "json":
            assert actual.read_bytes() == expected.read_bytes()
        if executor == "serial":
            serial_bytes = actual.read_bytes()
        else:
            assert actual.read_bytes() == serial_bytes
        assert timing.wall_seconds >= timing.merge_seconds >= 0


def test_partial_boundaries_and_errors(tmp_path: Path) -> None:
    chunks = stage_pages(
        [("one", "python"), ("two", "java"), ("three", "python")],
        "source",
        tmp_path,
        chunk_size=2,
    )
    assert [
        p.index.num_docs for p in map(lambda path: build_partial([path]), chunks)
    ] == [
        2,
        1,
    ]
    with pytest.raises(ValueError, match="increase"):
        merge([build_partial([chunks[1]]), build_partial([chunks[0]])])
    with pytest.raises(ValueError, match="positions"):
        merge([build_partial(chunks, positions=True)])
    with pytest.raises(ValueError, match="workers"):
        build_from_chunks(chunks, workers=0)
    with pytest.raises(ValueError, match="chunk_size"):
        stage_pages([], "source", tmp_path, chunk_size=0)


def test_process_worker_failure_surfaces(tmp_path: Path) -> None:
    damaged = tmp_path / "damaged.pkl"
    damaged.write_bytes(b"not a pickle")
    with pytest.raises(WorkerBuildError, match="worker failed") as failure:
        build_from_chunks([damaged], executor="processes", workers=2)
    assert isinstance(failure.value.__cause__, pickle.UnpicklingError)
    assert failure.value.__cause__.__cause__ is not None
