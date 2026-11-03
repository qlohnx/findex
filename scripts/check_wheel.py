"""Install a wheel in isolated uv tool directories and exercise its CLI.

Usage: uv run python scripts/check_wheel.py dist/findex-0.4.0-py3-none-any.whl
The user's tools and PATH are left untouched.
"""

import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    wheel = Path(sys.argv[1]).resolve()
    with zipfile.ZipFile(wheel) as archive:
        assert "findex/py.typed" in archive.namelist()
        entry_points = archive.read("findex-0.4.0.dist-info/entry_points.txt").decode()
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
        corpus = root / "corpus"
        corpus.mkdir()
        (corpus / "one.txt").write_text("Python search engine", encoding="utf-8")
        (corpus / "two.txt").write_text("Java search engine", encoding="utf-8")
        index = root / "index.json"
        run("index", str(corpus), "--out", str(index), "--positions")
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
            "Wheel installed outside the repository: help, index, stats, and both scorers passed."
        )


if __name__ == "__main__":
    main()
