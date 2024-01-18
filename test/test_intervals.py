import pytest

from relistats.intervals import (
    assurance_in_interval,
    confidence_interval_of_median,
    confidence_interval_of_percentile,
    percentile_interval_locs,
    tolerance_interval,
    tolerance_interval_locs,
)


def test_confidence_interval_indices_in_quantile() -> None:
    # Same numbers from the table in textbook mentioned above.
    # The confidence numbers are set to two decimal places to see if the
    # same interval as in the textbook is returned for a few samples
    # from the table
    assert percentile_interval_locs(n=5, p=0.5, c=0.93) == (1, 5)
    assert percentile_interval_locs(n=8, p=0.5, c=0.92) == (2, 7)
    assert percentile_interval_locs(n=11, p=0.5, c=0.93) == (3, 9)
    assert percentile_interval_locs(n=14, p=0.5, c=0.94) == (4, 11)
    assert percentile_interval_locs(n=17, p=0.5, c=0.95) == (5, 13)
    assert percentile_interval_locs(n=20, p=0.5, c=0.95) == (6, 15)

    # The rest are regression tests
    assert percentile_interval_locs(n=60, p=0.8, c=0.5) == (45, 50)
    assert percentile_interval_locs(n=60, p=0.8, c=0.6) == (45, 51)
    assert percentile_interval_locs(n=60, p=0.8, c=0.7) == (44, 51)
    assert percentile_interval_locs(n=60, p=0.8, c=0.8) == (45, 53)
    assert percentile_interval_locs(n=60, p=0.8, c=0.9) == (42, 53)

    assert percentile_interval_locs(n=60, p=0.9, c=0.7) == (52, 57)
    assert percentile_interval_locs(n=60, p=0.9, c=0.8) == (52, 58)
    assert percentile_interval_locs(n=60, p=0.9, c=0.9) == (50, 58)
    assert percentile_interval_locs(n=60, p=0.9, c=0.95) == (50, 59)
    assert percentile_interval_locs(n=60, p=0.95, c=0.95) == (52, 60)

    assert percentile_interval_locs(n=60, p=0.8, c=0.2) == (46, 48)

    assert percentile_interval_locs(n=60, p=0.5, c=0.5) == (26, 32)
    assert percentile_interval_locs(n=60, p=0.5, c=0.7) == (25, 34)
    assert percentile_interval_locs(n=60, p=0.5, c=0.8) == (24, 35)
    assert percentile_interval_locs(n=60, p=0.5, c=0.9) == (24, 37)
    assert percentile_interval_locs(n=60, p=0.5, c=0.95) == (22, 38)
    assert percentile_interval_locs(n=60, p=0.5, c=0.99) == (20, 40)


def test_tolerance_interval_indices() -> None:
    assert tolerance_interval_locs(n=60, t=0.8, c=0.5) == (5, 55)
    assert tolerance_interval_locs(n=60, t=0.8, c=0.7) == (4, 56)
    assert tolerance_interval_locs(n=60, t=0.8, c=0.8) == (5, 57)
    assert tolerance_interval_locs(n=60, t=0.8, c=0.9) == (4, 58)
    assert tolerance_interval_locs(n=60, t=0.85, c=0.85) == (4, 59)
    assert tolerance_interval_locs(n=60, t=0.9, c=0.9) == (3, 60)
    assert tolerance_interval_locs(n=120, t=0.95, c=0.95) == (2, 120)


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
    assert confidence_interval_of_percentile(0.75, 0.75, arr) == (22, 27)
    assert confidence_interval_of_percentile(0.75, 0.9, arr) == (21, 28)
    assert confidence_interval_of_percentile(0.5, 0.95, arr) == (15, 24)

    arr_float = [k * 0.1 for k in range(10, 70)]
    assert confidence_interval_of_percentile(0.8, 0.8, arr_float) == (
        pytest.approx(5.4, abs=0.01),
        pytest.approx(6.2, abs=0.01),
    )


def test_tolerance_interval() -> None:
    arr = range(10, 30)
    assert tolerance_interval(0.75, 0.75, arr) == (12, 29)
    assert tolerance_interval(0.75, 0.9, arr) == (11, 29)
    assert tolerance_interval(0.5, 0.95, arr) == (13, 28)
    assert tolerance_interval(0.8, 0.8, arr) == (11, 29)
    assert tolerance_interval(0.9, 0.9, arr) is None

    arr_float = [k * 0.1 for k in range(10, 70)]
    assert tolerance_interval(0.8, 0.8, arr_float) == (
        pytest.approx(1.4, abs=0.01),
        pytest.approx(6.6, abs=0.01),
    )


def test_assurance_interval() -> None:
    assert assurance_in_interval(1, 15, 16) == pytest.approx(0.818, 0.001)
    assert assurance_in_interval(1, 37, 38) == pytest.approx(0.901, 0.001)
    assert assurance_in_interval(9, 28, 38) == pytest.approx(0.686, 0.001)
    assert assurance_in_interval(7, 30, 38) == pytest.approx(0.731, 0.001)
    assert assurance_in_interval(5, 32, 38) == pytest.approx(0.777, 0.001)
    assert assurance_in_interval(3, 34, 38) == pytest.approx(0.824, 0.001)
    assert assurance_in_interval(2, 36, 38) == pytest.approx(0.874, 0.001)
    assert assurance_in_interval(1, 93, 94) == pytest.approx(0.951, 0.001)
