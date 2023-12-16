
import pytest

from relistats.quantile import (
    assurance_in_quantile,
    confidence_in_quantile,
    index_at_quantile,
    median_index
)

def test_confidence_in_quantile() -> None:
    assert confidence_in_quantile(14, 20, 0.5) == pytest.approx(0.98, abs=0.01)

def test_assurance_in_quantile() -> None:
    assert assurance_in_quantile(14, 20) == pytest.approx(0.67, abs=0.01)


def test_index_at_quantile() -> None:
    assert index_at_quantile(20, 0.5) == 10
    assert index_at_quantile(20, 0.5, 0.95) == 14

def test_median_index() -> None:
    assert median_index(20) == 14
    assert median_index(20, 0.99) == 15
    assert median_index(20, 0.9) == 13


