import pytest

from relistats.binom_fin import (
    conf_fin,
    reli_fin,
    assur_fin
)

def test_conf_fin() -> None:
    ABS_TOL_CONFIDENCE = 0.001

    assert conf_fin(4, 0, 0, 2) == (1, 1)
    assert conf_fin(4, 1, 0, 2) == (1, 0.75)
    assert conf_fin(4, 2, 0, 2) == (1, 0.5)
    assert conf_fin(4, 3, 0, 2) == (1, 0.25)
    assert conf_fin(4, 4, 0, 2) == (1, 0)

    assert conf_fin(4, 0, 4, 2) == pytest.approx((0.938, 0.75), abs=ABS_TOL_CONFIDENCE)
    assert conf_fin(4, 1, 4, 2) == pytest.approx((0.688, 0.625), abs=ABS_TOL_CONFIDENCE)
    assert conf_fin(4, 2, 4, 2) == pytest.approx((0.313, 0.5), abs=ABS_TOL_CONFIDENCE)
    assert conf_fin(4, 3, 4, 2) == pytest.approx((0.063, 0.375), abs=ABS_TOL_CONFIDENCE)
    assert conf_fin(4, 3, 4, 3) == pytest.approx((0.316, 0.25), abs=ABS_TOL_CONFIDENCE)
    assert conf_fin(4, 3, 4, 4) == pytest.approx((1, 0.125), abs=ABS_TOL_CONFIDENCE)

    assert conf_fin(10, 0, 10, 0) == pytest.approx((0.614, 0.952), abs=ABS_TOL_CONFIDENCE)
    assert conf_fin(10, 0, 10, 1) == pytest.approx((0.651, 0.95), abs=ABS_TOL_CONFIDENCE)
    assert conf_fin(10, 0, 10, 2) == pytest.approx((0.893, 0.9), abs=ABS_TOL_CONFIDENCE)
    assert conf_fin(10, 0, 10, 5) == pytest.approx((0.999, 0.75), abs=ABS_TOL_CONFIDENCE)
    assert conf_fin(10, 1, 10, 5) == pytest.approx((0.989, 0.7), abs=ABS_TOL_CONFIDENCE)
    assert conf_fin(10, 2, 10, 5) == pytest.approx((0.945, 0.65), abs=ABS_TOL_CONFIDENCE)
    assert conf_fin(10, 3, 10, 5) == pytest.approx((0.828, 0.6), abs=ABS_TOL_CONFIDENCE)
    assert conf_fin(10, 4, 10, 5) == pytest.approx((0.623, 0.55), abs=ABS_TOL_CONFIDENCE)
    assert conf_fin(10, 5, 10, 5) == pytest.approx((0.377, 0.5), abs=ABS_TOL_CONFIDENCE)
    assert conf_fin(10, 6, 10, 5) == pytest.approx((0.172, 0.45), abs=ABS_TOL_CONFIDENCE)

    assert conf_fin(2, 0, 2, 5) == (None, None)
    assert conf_fin(2, -2, 2, 0) == (None, None)
    assert conf_fin(-2, 0, 2, 0) == (None, None)
    assert conf_fin(2, 0, 2, -2) == (None, None)
    assert conf_fin(2, 0, -2, -2) == (None, None)

def test_reli_fin() -> None:
    ABS_TOL_RELIABILITY = 0.001
    assert reli_fin(4, 0, 0.5, 0) == (1, 1)
    assert reli_fin(4, 1, 0.5, 0) == (0.75, 1)
    assert reli_fin(4, 2, 0.5, 0) == (0.5, 1)
    assert reli_fin(4, 3, 0.5, 0) == (0.25, 1)
    assert reli_fin(4, 4, 0.5, 0) == (0, 1)

    assert reli_fin(4, 0, 0.5, 4) == pytest.approx((0.889, 0.590), abs=ABS_TOL_RELIABILITY)
    assert reli_fin(4, 1, 0.94, 4) == pytest.approx((0.5, 0.949), abs=ABS_TOL_RELIABILITY)
    assert reli_fin(4, 2, 0.31, 4) == pytest.approx((0.5, 0.313), abs=ABS_TOL_RELIABILITY)
    assert reli_fin(4, 3, 0.0039, 4) == pytest.approx((0.5, 0.004), abs=ABS_TOL_RELIABILITY)
    assert reli_fin(4, 3, 0.315, 4) == pytest.approx((0.25, 0.316), abs=ABS_TOL_RELIABILITY)

    assert reli_fin(2, 0, 2, 2) == (None, 2)
    assert reli_fin(2, -2, 0.5, 2) == (None, 0.5)
    assert reli_fin(-2, 0, 0.5, 2) == (None, 0.5)
    assert reli_fin(2, 0, -0.5, 2) == (None, -0.5)

def test_assurance() -> None:
    assert assur_fin(4, 1, 0) == pytest.approx((0.75, 0.75, 1), abs=0.001)
    assert assur_fin(4, 2, 0) == pytest.approx((0.5, 0.5, 1), abs=0.001)
    assert assur_fin(4, 3, 0) == pytest.approx((0.25, 0.25, 1), abs=0.001)
    assert assur_fin(4, 0, 4) == pytest.approx((0.75, 0.75, 0.938), abs=0.001)
    assert assur_fin(4, 1, 4) == pytest.approx((0.625, 0.625, 0.688), abs=0.001)
    assert assur_fin(4, 1, 8) == pytest.approx((0.583, 0.583, 0.688), abs=0.001)

    assert assur_fin(2, -2, 2) == (None, 0, 0)
    assert assur_fin(-2, 0, 2) == (None, 0, 0)








