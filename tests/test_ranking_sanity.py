import runpy


def test_bm25_sanity_checks():
    module = runpy.run_path("scripts/ranking_sanity.py")
    short, long, tf1, tf19, tf20 = module["controlled_bm25"]()
    assert short > long
    assert tf20 - tf19 < tf19 - tf1
