"""Profile repeated search requests after loading an index once.

Example: uv run python scripts/profile_ranking.py data/index.json
    "the AND of AND in" --repeats 100 --profile-out .cache/profiles/search.prof
"""

import argparse
import cProfile
import time
from pathlib import Path

from findex.ranking import BM25, search
from findex.store import open_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("query")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--profile-out", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    with open_index(args.index) as index:
        profiler = cProfile.Profile() if args.profile_out else None
        if profiler:
            profiler.enable()
        started = time.perf_counter()
        for _ in range(args.repeats):
            results = search(args.query, index, BM25(), k=10)
        elapsed = time.perf_counter() - started
        if profiler:
            profiler.disable()
            args.profile_out.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(args.profile_out))
        print(
            f"documents={index.num_docs} matches_returned={len(results)} "
            f"repeats={args.repeats} elapsed_s={elapsed:.3f} "
            f"ms_per_search={elapsed * 1000 / args.repeats:.3f}"
        )


if __name__ == "__main__":
    main()
