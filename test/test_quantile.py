import pytest

from relistats.quantile import (
    assurance_in_quantile,
    confidence_in_quantile_at_index,
    confidence_interval_of_median,
    confidence_interval_of_quantile,
    quantile_interval_indices,
    tolerance_interval_indices,
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


def test_tolerance_interval_indices() -> None:
    # For debugging
    # for k in range(41, 53):
    #     print(f"{k} : {confidence_in_quantile_at_index(k, n=60, q=0.8)}")

    assert tolerance_interval_indices(n=60, t=0.8, c=0.5) == (2, 54)
    assert tolerance_interval_indices(n=60, t=0.8, c=0.7) == (1, 55)
    assert tolerance_interval_indices(n=60, t=0.8, c=0.8) == (2, 56)

    assert tolerance_interval_indices(n=80, t=0.9, c=0.8) == (1, 78)
    assert tolerance_interval_indices(n=120, t=0.9, c=0.9) == (1, 117)


def test_assurance_in_quantile() -> None:
    assert assurance_in_quantile(14, 20) == pytest.approx(0.67, abs=0.01)


def test_median_interval() -> None:
    arr = range(10, 30)
    assert confidence_interval_of_median(0.95, arr) == (15, 24)
    arr_float = [k * 0.1 for k in range(10, 30)]
    assert confidence_interval_of_median(0.95, arr_float) == (
        pytest.approx(1.5, abs=0.01),
        pytest.approx(2.4, abs=0.01),
    )


def test_quantile_interval() -> None:
    arr = range(10, 30)
    assert confidence_interval_of_quantile(0.75, 0.75, arr) == (20, 26)
    assert confidence_interval_of_quantile(0.75, 0.9, arr) == (19, 27)
    assert confidence_interval_of_quantile(0.5, 0.95, arr) == (15, 24)

    arr_float = [k * 0.1 for k in range(10, 70)]
    assert confidence_interval_of_quantile(0.8, 0.8, arr_float) == (
        pytest.approx(5.2, abs=0.01),
        pytest.approx(6.1, abs=0.01),
    )
