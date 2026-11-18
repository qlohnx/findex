"""Install a wheel in isolated uv tool directories and exercise its CLI.

Usage: uv run python scripts/check_wheel.py dist/findex-0.7.0-py3-none-any.whl
The user's tools and PATH are left untouched.
"""

import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from rich.text import Text


def main() -> None:
    wheel = Path(sys.argv[1]).resolve()
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "findex/py.typed" in names
        assert "findex/web/templates/index.html" in names
        assert "findex/web/templates/document.html" in names
        assert "findex/web/static/style.css" in names
        entry_points_file = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        entry_points = archive.read(entry_points_file).decode()
        assert "findex = findex.cli:app" in entry_points

    with tempfile.TemporaryDirectory(prefix="findex-wheel-") as directory:
        root = Path(directory)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["UV_TOOL_DIR"] = str(root / "tools")
        env["UV_TOOL_BIN_DIR"] = str(root / "bin")
        subprocess.run(
            ["uv", "tool", "install", "--python", sys.executable, str(wheel)],
            cwd=root,
            env=env,
            check=True,
        )
        executable = root / "bin" / ("findex.exe" if os.name == "nt" else "findex")

        def run(*args: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [str(executable), *args],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )

        assert "Usage" in run("--help").stdout
        assert "--max-pages" in Text.from_ansi(run("crawl", "--help").stdout).plain
        assert "--workers" in Text.from_ansi(run("serve", "--help").stdout).plain
        corpus = root / "corpus"
        corpus.mkdir()
        (corpus / "one.txt").write_text("Python search engine", encoding="utf-8")
        (corpus / "two.txt").write_text("Java search engine", encoding="utf-8")
        index = root / "index.json"
        run("index", str(corpus), "--out", str(index), "--positions")
        parallel = root / "parallel.json"
        run(
            "index",
            str(corpus),
            "--out",
            str(parallel),
            "--positions",
            "--executor",
            "processes",
            "--workers",
            "2",
        )
        assert parallel.read_bytes() == index.read_bytes()
        assert "Documents" in run("stats", str(index)).stdout
        for scorer in ("bm25", "tfidf"):
            rows = [
                json.loads(line)
                for line in run(
                    "search", str(index), "python", "--scorer", scorer, "--json"
                ).stdout.splitlines()
            ]
            assert len(rows) == 1 and rows[0]["doc_id"] == 1
        print(
            "Wheel installed in an isolated tool directory: web assets, serve/crawl help, serial/process indexing, stats, and both scorers passed."
        )


if __name__ == "__main__":
    main()
