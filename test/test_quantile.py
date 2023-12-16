
import pytest

from relistats.quantile import (
    confidence_in_quantile,
    assurance_in_quantile,
    index_at_quantile,
)

def test_confidence_in_quantile() -> None:
    assert confidence_in_quantile(14, 20, 0.5) == pytest.approx(0.98, abs=0.01)

def test_assurance_in_quantile() -> None:
    assert assurance_in_quantile(14, 20) == pytest.approx(0.67, abs=0.01)


def test_index_at_quantile() -> None:
    assert index_at_quantile(20, 0.5) == 10
    assert index_at_quantile(20, 0.5, 0.95) == 14
    