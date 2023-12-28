import pytest

from relistats.quantile import (
    assurance_in_quantile,
    confidence_in_quantile_at_index,
    confidence_interval_of_median,
    confidence_interval_of_quantile,
    quantile_interval_indices,
    tolerance_interval_indices,
)


def confidence_from_textbook_example(n, i, j, p) -> float:
    return confidence_in_quantile_at_index(j, n, p) - confidence_in_quantile_at_index(
        i, n, p
    )


# Textbook Reference: https://online.stat.psu.edu/stat415/lesson/19/19.1
# Penn State Department of Statistics, "STAT 415: Introduction to Mathematical Statistics", Lesson 19.1
# See Table of confidence numbers
def test_textbook_examples() -> None:
    assert confidence_from_textbook_example(5, 1, 5, 0.5) == pytest.approx(
        0.9376, abs=0.001
    )
    assert confidence_from_textbook_example(6, 1, 6, 0.5) == pytest.approx(
        0.9688, abs=0.001
    )
    assert confidence_from_textbook_example(7, 1, 7, 0.5) == pytest.approx(
        0.9844, abs=0.001
    )
    assert confidence_from_textbook_example(8, 2, 7, 0.5) == pytest.approx(
        0.9296, abs=0.001
    )
    assert confidence_from_textbook_example(9, 2, 8, 0.5) == pytest.approx(
        0.9610, abs=0.001
    )
    assert confidence_from_textbook_example(10, 2, 9, 0.5) == pytest.approx(
        0.9786, abs=0.001
    )
    assert confidence_from_textbook_example(11, 3, 9, 0.5) == pytest.approx(
        0.9346, abs=0.001
    )
    assert confidence_from_textbook_example(12, 3, 10, 0.5) == pytest.approx(
        0.9614, abs=0.001
    )

    assert confidence_from_textbook_example(13, 3, 11, 0.5) == pytest.approx(
        0.9776, abs=0.001
    )
    assert confidence_from_textbook_example(14, 4, 11, 0.5) == pytest.approx(
        0.9426, abs=0.001
    )
    assert confidence_from_textbook_example(15, 4, 12, 0.5) == pytest.approx(
        0.9648, abs=0.001
    )
    assert confidence_from_textbook_example(16, 5, 12, 0.5) == pytest.approx(
        0.9232, abs=0.001
    )
    assert confidence_from_textbook_example(17, 5, 13, 0.5) == pytest.approx(
        0.9510, abs=0.001
    )
    assert confidence_from_textbook_example(18, 5, 14, 0.5) == pytest.approx(
        0.9692, abs=0.001
    )
    assert confidence_from_textbook_example(19, 6, 14, 0.5) == pytest.approx(
        0.9364, abs=0.001
    )
    assert confidence_from_textbook_example(20, 6, 15, 0.5) == pytest.approx(
        0.9586, abs=0.001
    )


def test_confidence_in_quantile_at_index() -> None:
    assert confidence_in_quantile_at_index(14, 20, 0.5) == pytest.approx(
        0.942, abs=0.001
    )
    assert confidence_in_quantile_at_index(19, 20, 0.95) == pytest.approx(
        0.264, abs=0.001
    )
    assert confidence_in_quantile_at_index(19, 20, 0.9) == pytest.approx(
        0.608, abs=0.001
    )
    assert confidence_in_quantile_at_index(19, 20, 0.85) == pytest.approx(
        0.824, abs=0.001
    )
    assert confidence_in_quantile_at_index(1, 20, 0.05) == pytest.approx(
        0.358, abs=0.01
    )


def test_confidence_interval_indices_in_quantile() -> None:
    # Same numbers from the table in textbook mentioned above.
    # The confidence numbers are set to two decimal places to see if the
    # same interval as in the textbook is returned for a few samples
    # from the table
    assert quantile_interval_indices(n=5, pp=0.5, c=0.93) == (1, 5)
    assert quantile_interval_indices(n=8, pp=0.5, c=0.92) == (2, 7)
    assert quantile_interval_indices(n=11, pp=0.5, c=0.93) == (3, 9)
    assert quantile_interval_indices(n=14, pp=0.5, c=0.94) == (4, 11)
    assert quantile_interval_indices(n=17, pp=0.5, c=0.95) == (5, 13)
    assert quantile_interval_indices(n=20, pp=0.5, c=0.95) == (6, 15)

    # The rest are regression tests
    assert quantile_interval_indices(n=60, pp=0.8, c=0.5) == (43, 49)
    assert quantile_interval_indices(n=60, pp=0.8, c=0.6) == (43, 50)
    assert quantile_interval_indices(n=60, pp=0.8, c=0.7) == (44, 51)
    assert quantile_interval_indices(n=60, pp=0.8, c=0.8) == (43, 52)
    assert quantile_interval_indices(n=60, pp=0.8, c=0.9) == (42, 53)

    assert quantile_interval_indices(n=60, pp=0.9, c=0.7) == (49, 56)
    assert quantile_interval_indices(n=60, pp=0.9, c=0.8) == (50, 57)
    assert quantile_interval_indices(n=60, pp=0.9, c=0.9) == (50, 58)
    assert quantile_interval_indices(n=60, pp=0.9, c=0.95) == (50, 59)
    assert quantile_interval_indices(n=60, pp=0.95, c=0.95) == (52, 60)

    assert quantile_interval_indices(n=60, pp=0.8, c=0.2) == (46, 48)

    assert quantile_interval_indices(n=60, pp=0.5, c=0.5) == (24, 31)
    assert quantile_interval_indices(n=60, pp=0.5, c=0.7) == (23, 33)
    assert quantile_interval_indices(n=60, pp=0.5, c=0.8) == (22, 34)
    assert quantile_interval_indices(n=60, pp=0.5, c=0.9) == (22, 36)
    assert quantile_interval_indices(n=60, pp=0.5, c=0.95) == (20, 37)
    assert quantile_interval_indices(n=60, pp=0.5, c=0.99) == (20, 40)


def test_tolerance_interval_indices() -> None:
    assert tolerance_interval_indices(n=60, t=0.8, c=0.5) == (7, 55)
    assert tolerance_interval_indices(n=60, t=0.8, c=0.7) == (6, 56)
    assert tolerance_interval_indices(n=60, t=0.8, c=0.8) == (5, 57)
    assert tolerance_interval_indices(n=60, t=0.8, c=0.9) == (4, 58)
    assert tolerance_interval_indices(n=60, t=0.85, c=0.85) == (3, 59)
    assert tolerance_interval_indices(n=60, t=0.9, c=0.9) == (2, 60)
    assert tolerance_interval_indices(n=120, t=0.95, c=0.95) == (2, 120)


def test_assurance_in_quantile() -> None:
    assert assurance_in_quantile(14, 20) == pytest.approx(0.635, abs=0.001)


def test_median_interval() -> None:
    arr = range(10, 30)
    assert confidence_interval_of_median(0.95, arr) == (16, 25)
    arr_float = [k * 0.1 for k in range(10, 30)]
    assert confidence_interval_of_median(0.95, arr_float) == (
        pytest.approx(1.6, abs=0.01),
        pytest.approx(2.5, abs=0.01),
    )


def test_quantile_interval() -> None:
    arr = range(10, 30)
    assert confidence_interval_of_quantile(0.75, 0.75, arr) == (21, 27)
    assert confidence_interval_of_quantile(0.75, 0.9, arr) == (20, 28)
    assert confidence_interval_of_quantile(0.5, 0.95, arr) == (16, 25)

    arr_float = [k * 0.1 for k in range(10, 70)]
    assert confidence_interval_of_quantile(0.8, 0.8, arr_float) == (
        pytest.approx(5.3, abs=0.01),
        pytest.approx(6.2, abs=0.01),
    )
