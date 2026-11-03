import logging

import pytest

from findex.ranking import timed


def test_timed_preserves_signature_and_logs_on_failure(caplog):
    @timed
    def divide(numerator: float, denominator: float) -> float:
        """A typed function with both success and failure paths."""
        return numerator / denominator

    assert divide.__name__ == "divide"
    assert "typed function" in divide.__doc__
    with caplog.at_level(logging.DEBUG, logger="findex.ranking"):
        assert divide(6, denominator=2) == 3
        with pytest.raises(ZeroDivisionError):
            divide(1, 0)
    records = [record for record in caplog.records if "divide took" in record.message]
    assert len(records) == 2
    assert all(record.levelno == logging.DEBUG for record in records)
