import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.index import InvertedIndex, build_index
from findex.store import load_index, save_index

runner = CliRunner()


@pytest.fixture
def saved_index(tmp_path: Path, tiny_index: InvertedIndex) -> Path:
    path = tmp_path / "index.json"
    save_index(tiny_index, path)
    return path


@pytest.mark.parametrize("command", [[], ["index"], ["search"], ["stats"]])
def test_help(command):
    result = runner.invoke(app, [*command, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.stdout


@pytest.mark.parametrize("suffix", ["json", "pkl"])
def test_index_text_corpus(text_corpus, tmp_path, suffix):
    path = tmp_path / f"index.{suffix}"
    result = runner.invoke(
        app,
        ["index", str(text_corpus), "--out", str(path), "--positions", "--limit", "2"],
    )
    assert result.exit_code == 0, result.output
    index = load_index(path)
    assert index.num_docs == 2
    assert index.positions
    assert index.df("search") == 2
    assert "Indexed 2 documents" in result.stdout


def test_index_xml_dump(wiki_dump, tmp_path):
    path = tmp_path / "index.json"
    result = runner.invoke(app, ["index", str(wiki_dump), "--out", str(path)])
    assert result.exit_code == 0, result.output
    assert load_index(path).num_docs == 2


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
def test_search_json_is_clean(saved_index, scorer):
    result = runner.invoke(
        app,
        [
            "-vv",
            "search",
            str(saved_index),
            "python",
            "--scorer",
            scorer,
            "--k",
            "1",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 1
    assert set(rows[0]) == {"doc_id", "score", "title", "snippet"}
    assert rows[0]["score"] > 0
    assert "**" in rows[0]["snippet"]
    assert "DEBUG:" in result.stderr
    assert "INFO:" in result.stderr


def test_search_rich_table(saved_index):
    result = runner.invoke(app, ["search", str(saved_index), '"search engine"'])
    assert result.exit_code == 0, result.output
    assert "Search results" in result.stdout
    assert "Python" in result.stdout
    assert "Java" in result.stdout
    assert "**" not in result.stdout
    assert not result.stderr


def test_stats(saved_index):
    result = runner.invoke(app, ["stats", str(saved_index)])
    assert result.exit_code == 0, result.output
    assert "Documents" in result.stdout
    assert "Total tokens" in result.stdout
    assert "Top-50" in result.stdout
    assert "python" in result.stdout


@pytest.mark.parametrize(
    "verbosity, info, debug",
    [([], False, False), (["-v"], True, False), (["-vv"], True, True)],
)
def test_logging_levels(saved_index, verbosity, info, debug):
    result = runner.invoke(
        app, [*verbosity, "search", str(saved_index), "python", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert ("INFO:" in result.stderr) is info
    assert ("DEBUG:" in result.stderr) is debug
    for line in result.stdout.splitlines():
        assert isinstance(json.loads(line), dict)


@pytest.mark.parametrize(
    "query",
    ["", "(", "python OR", "python)", '"unclosed', "AND python", "python AND OR java"],
)
def test_bad_query_is_one_line_error(saved_index, query):
    result = runner.invoke(app, ["search", str(saved_index), query, "--json"])
    assert result.exit_code == 1
    assert result.stderr.startswith("Error:")
    assert len(result.stderr.splitlines()) == 1
    assert not result.stdout
    assert "Traceback" not in result.output


@pytest.mark.parametrize("command", ["search", "stats"])
def test_missing_index(tmp_path, command):
    args = [command, str(tmp_path / "missing.json")]
    if command == "search":
        args.append("python")
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert len(result.stderr.splitlines()) == 1
    assert not result.stdout


@pytest.mark.parametrize("contents", ["{", "[]", "{}", '{"positions": "wrong"}'])
def test_corrupt_json_index(tmp_path, contents):
    path = tmp_path / "bad.json"
    path.write_text(contents, encoding="utf-8")
    result = runner.invoke(app, ["stats", str(path)])
    assert result.exit_code == 1
    assert result.stderr.startswith("Error:")
    assert not result.stdout


@pytest.mark.parametrize(
    "problem",
    [
        "zero_tf",
        "negative_tf",
        "duplicate_posting",
        "bad_positions",
        "negative_length",
        "missing_metadata",
    ],
)
def test_invalid_index_values_are_rejected(saved_index, problem):
    data = json.loads(saved_index.read_text(encoding="utf-8"))
    posting = data["index"]["python"][0]
    if problem == "zero_tf":
        posting[1] = 0
    elif problem == "negative_tf":
        posting[1] = -1
    elif problem == "duplicate_posting":
        data["index"]["python"].insert(0, posting.copy())
    elif problem == "bad_positions":
        posting[2] = [999] * posting[1]
    elif problem == "negative_length":
        data["doc_lengths"]["1"] = -1
    elif problem == "missing_metadata":
        del data["doc_meta"]["1"]
    saved_index.write_text(json.dumps(data), encoding="utf-8")
    result = runner.invoke(app, ["stats", str(saved_index)])
    assert result.exit_code == 1, result.output
    assert result.stderr.startswith("Error:")
    assert len(result.stderr.splitlines()) == 1
    assert not result.stdout


def test_invalid_tf_does_not_escape_as_zero_division(tmp_path):
    path = tmp_path / "index.json"
    save_index(build_index([("A", "python " * 5), ("B", "python " * 13)]), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["index"]["python"][0][1] = -1
    path.write_text(json.dumps(data), encoding="utf-8")
    result = runner.invoke(app, ["search", str(path), "python", "--json"])
    assert result.exit_code == 1
    assert result.stderr.startswith("Error:")
    assert "Traceback" not in result.output
    assert not result.stdout


def test_no_matches_emits_no_json(saved_index):
    result = runner.invoke(app, ["search", str(saved_index), "nonexistent", "--json"])
    assert result.exit_code == 0
    assert not result.stdout
    assert not result.stderr


@pytest.mark.parametrize("option, value", [("--k", "0"), ("--scorer", "invalid")])
def test_invalid_search_options(saved_index, option, value):
    result = runner.invoke(app, ["search", str(saved_index), "python", option, value])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_phrase_requires_positions(text_corpus, tmp_path):
    path = tmp_path / "index.json"
    assert (
        runner.invoke(app, ["index", str(text_corpus), "--out", str(path)]).exit_code
        == 0
    )
    result = runner.invoke(app, ["search", str(path), '"search engine"'])
    assert result.exit_code == 1
    assert "--positions" in result.stderr


@pytest.mark.parametrize("kind", ["missing", "empty", "unsupported", "broken_xml"])
def test_bad_corpus(tmp_path, kind):
    corpus = tmp_path / "corpus"
    if kind == "empty":
        corpus.mkdir()
    elif kind == "unsupported":
        corpus.write_text("text", encoding="utf-8")
    elif kind == "broken_xml":
        import bz2

        corpus = tmp_path / "bad.xml.bz2"
        corpus.write_bytes(bz2.compress(b"<broken"))
    result = runner.invoke(
        app, ["index", str(corpus), "--out", str(tmp_path / "index.json")]
    )
    assert result.exit_code == 1
    assert result.stderr.startswith("Error:")
    assert not (tmp_path / "index.json").exists()
