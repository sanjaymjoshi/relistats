import pytest

from relistats.percentile import assurance_in_percentile, confidence_in_percentile


def confidence_from_textbook_example(n, i, j, p) -> float:
    return confidence_in_percentile(j, n, p) - confidence_in_percentile(i, n, p)


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
    assert confidence_in_percentile(14, 20, 0.5) == pytest.approx(0.942, abs=0.001)
    assert confidence_in_percentile(19, 20, 0.95) == pytest.approx(0.264, abs=0.001)
    assert confidence_in_percentile(19, 20, 0.9) == pytest.approx(0.608, abs=0.001)
    assert confidence_in_percentile(19, 20, 0.85) == pytest.approx(0.824, abs=0.001)
    assert confidence_in_percentile(1, 20, 0.05) == pytest.approx(0.358, abs=0.01)


def test_assurance_in_quantile() -> None:
    assert assurance_in_percentile(14, 20) == pytest.approx(0.635, abs=0.001)


def test_invalids() -> None:
    assert assurance_in_percentile(0, 0) is None
    assert assurance_in_percentile(-1, 0) is None
    assert assurance_in_percentile(10, 9) is None
