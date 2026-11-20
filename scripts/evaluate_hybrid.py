"""Print keyword, semantic, and hybrid precision@5 for 15 judged queries.

First run: uv run --extra semantic findex embed data/index.json
Then:      uv run --extra semantic python scripts/evaluate_hybrid.py data/index.json
"""

import argparse
from pathlib import Path

from evaluate import JUDGMENTS, precision_at_5

from findex.ranking import BM25, search
from findex.semantic import (
    SemanticStore,
    default_prefix,
    load_model,
    rrf_fuse,
    semantic_search,
)
from findex.store import open_index

PARAPHRASE_JUDGMENTS: dict[str, set[int]] = {
    "study of computers and computation": {71},
    "machines that mimic human thinking": {2621},
    "algorithms that learn from data": {2621},
    "global armed conflict": {1514},
    "reducing file size without losing information": {1318},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    args = parser.parse_args()
    judgments = {**JUDGMENTS, **PARAPHRASE_JUDGMENTS}
    print("| Query | Relevant IDs | Keyword P@5 | Semantic P@5 | Hybrid P@5 |")
    print("| --- | --- | ---: | ---: | ---: |")
    means = [0.0, 0.0, 0.0]
    with open_index(args.index) as index:
        store = SemanticStore.load(default_prefix(args.index), index)
        model = load_model(store.model_name)
        for query, relevant in judgments.items():
            keyword = search(query, index, BM25(), k=100)
            semantic = semantic_search(query, index, store, k=100, model=model)
            hybrid = rrf_fuse([keyword, semantic], k=5)
            scores = [
                precision_at_5([r.doc_id for r in ranking], relevant)
                for ranking in (keyword, semantic, hybrid)
            ]
            means = [a + b for a, b in zip(means, scores, strict=True)]
            print(
                f"| `{query}` | {', '.join(map(str, sorted(relevant)))} | "
                + " | ".join(f"{score:.2f}" for score in scores)
                + " |"
            )
    print("| **Mean** | | " + " | ".join(f"{total / 15:.3f}" for total in means) + " |")


if __name__ == "__main__":
    main()
