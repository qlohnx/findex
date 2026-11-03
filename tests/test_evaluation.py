import runpy


def test_precision_at_5():
    module = runpy.run_path("scripts/evaluate.py")
    precision_at_5 = module["precision_at_5"]
    assert precision_at_5([1, 2, 3, 4, 5], {2, 8}) == 0.2
    assert precision_at_5([1, 2, 3, 4, 5], {1, 2, 3, 4, 5}) == 1.0
