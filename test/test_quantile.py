import pytest

from relistats.quantile import (
    assurance_in_quantile,
    confidence_in_quantile_at_index,
    index_at_quantile,
    interval_at_quantile,
    median_index,
    median_interval,
    median_interval_with_confidence,
    median_with_confidence,
    quantile_interval,
    quantile_interval_indices,
    quantile_with_confidence,
)


def test_confidence_in_quantile_at_index() -> None:
    assert confidence_in_quantile_at_index(14, 20, 0.5) == pytest.approx(0.98, abs=0.01)
    assert confidence_in_quantile_at_index(19, 20, 0.95) == pytest.approx(
        0.64, abs=0.01
    )
    assert confidence_in_quantile_at_index(19, 20, 0.9) == pytest.approx(0.88, abs=0.01)
    assert confidence_in_quantile_at_index(19, 20, 0.85) == pytest.approx(
        0.96, abs=0.01
    )
    assert confidence_in_quantile_at_index(1, 20, 0.05) == pytest.approx(0.74, abs=0.01)


def test_confidence_interval_indices_in_quantile() -> None:
    # For debugging
    # for k in range(41, 53):
    #     print(f"{k} : {confidence_in_quantile_at_index(k, n=60, q=0.8)}")

    assert quantile_interval_indices(n=60, q=0.8, c=0.5) == (42, 48)
    assert quantile_interval_indices(n=60, q=0.8, c=0.6) == (42, 49)
    assert quantile_interval_indices(n=60, q=0.8, c=0.7) == (43, 50)
    assert quantile_interval_indices(n=60, q=0.8, c=0.8) == (42, 51)
    assert quantile_interval_indices(n=60, q=0.8, c=0.9) == (41, 52)

    assert quantile_interval_indices(n=60, q=0.9, c=0.7) == (48, 55)
    assert quantile_interval_indices(n=60, q=0.9, c=0.8) == (49, 56)
    assert quantile_interval_indices(n=60, q=0.9, c=0.9) == (49, 57)
    assert quantile_interval_indices(n=60, q=0.9, c=0.95) == (49, 58)
    assert quantile_interval_indices(n=60, q=0.95, c=0.95) is None

    assert quantile_interval_indices(n=60, q=0.8, c=0.2) == (46, 48)

    assert quantile_interval_indices(n=60, q=0.5, c=0.5) == (23, 30)
    assert quantile_interval_indices(n=60, q=0.5, c=0.7) == (22, 32)
    assert quantile_interval_indices(n=60, q=0.5, c=0.8) == (21, 33)
    assert quantile_interval_indices(n=60, q=0.5, c=0.9) == (21, 35)
    assert quantile_interval_indices(n=60, q=0.5, c=0.95) == (19, 36)
    assert quantile_interval_indices(n=60, q=0.5, c=0.99) == (19, 39)


def test_assurance_in_quantile() -> None:
    assert assurance_in_quantile(14, 20) == pytest.approx(0.67, abs=0.01)


def test_index_at_quantile() -> None:
    assert index_at_quantile(20, 0.5) == 10
    assert index_at_quantile(20, 0.5, 0.95) == 14


def test_interval_at_qunatile() -> None:
    assert interval_at_quantile(60, 0.75) == (10, 49)
    assert interval_at_quantile(60, 0.75, 0.99) == (6, 53)
    assert interval_at_quantile(60, 0.75, 0.9) == (9, 50)


def test_median_index() -> None:
    assert median_index(20) == 14
    assert median_index(20, 0.99) == 15
    assert median_index(20, 0.9) == 13


def test_median_interval() -> None:
    assert median_interval(20) == (7, 12)
    assert median_interval(20, 0.99) == (3, 16)
    assert median_interval(20, 0.9) == (5, 14)


def test_median_with_confidence() -> None:
    arr = range(10, 30)
    assert median_with_confidence(0.95, arr) == 24
    arr_float = [k * 0.1 for k in range(10, 30)]
    assert median_with_confidence(0.95, arr_float) == pytest.approx(2.4, abs=0.01)


def test_quantile_with_confidence() -> None:
    arr = range(10, 30)
    assert quantile_with_confidence(0.75, 0.75, arr) == 26
    assert quantile_with_confidence(0.5, 0.95, arr) == 24


def test_median_interval_with_confidence() -> None:
    arr = range(10, 30)
    assert median_interval_with_confidence(0.95, arr) == (15, 24)
    arr_float = [k * 0.1 for k in range(10, 30)]
    assert median_interval_with_confidence(0.95, arr_float) == (
        pytest.approx(1.5, abs=0.01),
        pytest.approx(2.4, abs=0.01),
    )


def test_quantile_interval() -> None:
    arr = range(10, 30)
    assert quantile_interval(0.75, 0.75, arr) == (20, 26)
    assert quantile_interval(0.75, 0.9, arr) == (19, 27)
    assert quantile_interval(0.5, 0.95, arr) == (15, 24)

    arr_float = [k * 0.1 for k in range(10, 70)]
    assert quantile_interval(0.8, 0.8, arr_float) == (
        pytest.approx(5.2, abs=0.01),
        pytest.approx(6.1, abs=0.01),
    )
