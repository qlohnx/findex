"""Regenerate the Lab 05 comparison plot from saved measurement JSON files."""

import json
from pathlib import Path

from benchmark_parallel import plot_results


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    standard = json.loads((root / "lab05.json").read_text(encoding="utf-8"))
    free = json.loads((root / "lab05-3.13t.json").read_text(encoding="utf-8"))
    if standard["documents"] != free["documents"] or standard["terms"] != free["terms"]:
        raise ValueError("benchmark datasets differ")
    plot_results(standard["rows"], root / "lab05-speedup.png", free["rows"])


if __name__ == "__main__":
    main()
