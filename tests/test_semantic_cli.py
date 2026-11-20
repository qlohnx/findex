"""CLI wiring for embedding creation and semantic/hybrid modes."""

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

import findex.semantic as semantic_module
from findex.cli import app
from findex.index import build_index
from findex.store import save_index


class SmallModel:
    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, sentences, **kwargs):
        return np.array(
            [
                [1.0, 0.0] if "cat" in text or "feline" in text else [0.0, 1.0]
                for text in sentences
            ],
            dtype=np.float32,
        )


def test_embed_then_semantic_and_hybrid_cli(tmp_path: Path, monkeypatch):
    index_path = tmp_path / "index.json"
    index = build_index([("Cat", "The cat sleeps."), ("Dog", "The dog runs.")])
    save_index(index, index_path)
    monkeypatch.setattr(semantic_module, "load_model", lambda _: SmallModel())
    runner = CliRunner()
    embedded = runner.invoke(app, ["embed", str(index_path), "--batch-size", "1"])
    assert embedded.exit_code == 0, embedded.output
    for mode in ("semantic", "hybrid"):
        result = runner.invoke(
            app, ["search", str(index_path), "feline", "--mode", mode, "--json"]
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output.splitlines()[0])["doc_id"] == 1


def test_semantic_cli_needs_embeddings(tmp_path: Path):
    index_path = tmp_path / "index.json"
    save_index(build_index([("Cat", "The cat sleeps.")]), index_path)
    result = CliRunner().invoke(
        app, ["search", str(index_path), "cat", "--mode", "semantic"]
    )
    assert result.exit_code == 1
    assert "Error:" in result.output
