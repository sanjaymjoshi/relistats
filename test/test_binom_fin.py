from dataclasses import asdict, dataclass

import pytest

from relistats.binom_fin import (
    #assurance,
    conf_fin,
    reli_fin,
)


@dataclass
class NFRC:
    n: int
    f: int
    r: float
    c: float


test_nfrc = (
    NFRC(1, 0, 0.5, 0.5),
    NFRC(2, 0, 0.5, 0.75),
    NFRC(2, 1, 0.5, 0.25),
    NFRC(8, 0, 0.7, 0.942),
    NFRC(8, 0, 0.9, 0.570),
    NFRC(10, 9, 0.1, 0.349),
    NFRC(100, 10, 0.9, 0.417),
    NFRC(1000, 100, 0.9, 0.473),
    NFRC(10000, 1000, 0.9, 0.492),
)


def test_conf_fin() -> None:

    # Confidence computation is exact, so set the tolerance tight
    ABS_TOL_CONFIDENCE = 0.001
#    for x in test_nfrc:
#        print(f"Testing confidence: {asdict(x)}")
#        assert confidence(x.n, x.f, x.r) == pytest.approx(x.c, abs=ABS_TOL_CONFIDENCE)

    assert conf_fin(4, 0, 0.5, 0) == (1, 0.5)
    assert conf_fin(4, 1, 0.5, 0) == (1, 0.5)
    assert conf_fin(4, 2, 0.5, 0) == (1, 0.5)
    assert conf_fin(4, 3, 0.5, 0) == (0, 0.5)
    assert conf_fin(4, 4, 0.5, 0) == (0, 0.5)

    assert conf_fin(4, 0, 0.5, 4) == (1, 0.5)
    assert conf_fin(4, 1, 0.5, 4) == pytest.approx((0.949, 0.5), abs=ABS_TOL_CONFIDENCE)
    assert conf_fin(4, 2, 0.5, 4) == pytest.approx((0.313, 0.5), abs=ABS_TOL_CONFIDENCE)
    assert conf_fin(4, 3, 0.5, 4) == pytest.approx((0.004, 0.5), abs=ABS_TOL_CONFIDENCE)
    assert conf_fin(4, 3, 0.25, 4) == pytest.approx((0.316, 0.25), abs=ABS_TOL_CONFIDENCE)

    assert conf_fin(2, 0, 2, 2) == (None, 2)
    assert conf_fin(2, -2, 0.5, 2) == (None, 0.5)
    assert conf_fin(-2, 0, 0.5, 2) == (None, 0.5)
    assert conf_fin(2, 0, -0.5, 2) == (None, -0.5)



def test_reli_fin() -> None:
#     # Reliability computation should be accurate, so set the tolerance tight
    ABS_TOL_RELIABILITY = 0.001
#     for x in test_nfrc:
#         print(f"Testing reliability: {asdict(x)}")
#         c1 = confidence(x.n, x.f, x.r)
#         if c1 is None:
#             c1 = 0
#         assert reliability(x.n, x.f, c1) == pytest.approx(x.r, abs=ABS_TOL_RELIABILITY)
    assert reli_fin(4, 0, 0.5, 0) == (1, 1)
    assert reli_fin(4, 1, 0.5, 0) == (0.75, 1)
    assert reli_fin(4, 2, 0.5, 0) == (0.5, 1)
    assert reli_fin(4, 3, 0.5, 0) == (0.25, 1)
    assert reli_fin(4, 4, 0.5, 0) == (0, 1)

    assert reli_fin(4, 0, 0.5, 4) == pytest.approx((1, 0.590), abs=ABS_TOL_RELIABILITY)
    assert reli_fin(4, 1, 0.94, 4) == pytest.approx((0.5, 0.949), abs=ABS_TOL_RELIABILITY)
    assert reli_fin(4, 2, 0.31, 4) == pytest.approx((0.5, 0.313), abs=ABS_TOL_RELIABILITY)
    assert reli_fin(4, 3, 0.0039, 4) == pytest.approx((0.5, 0.004), abs=ABS_TOL_RELIABILITY)
    assert reli_fin(4, 3, 0.315, 4) == pytest.approx((0.25, 0.316), abs=ABS_TOL_RELIABILITY)

    assert reli_fin(2, 0, 2, 2) == (None, 2)
    assert reli_fin(2, -2, 0.5, 2) == (None, 0.5)
    assert reli_fin(-2, 0, 0.5, 2) == (None, 0.5)
    assert reli_fin(2, 0, -0.5, 2) == (None, -0.5)

# def test_assurance() -> None:
#     assert assurance(22, 0) == pytest.approx(0.9, abs=0.001)
#     assert assurance(59, 0) == pytest.approx(0.95, abs=0.001)
#     assert assurance(22, 2) == pytest.approx(0.812, abs=0.001)
#     assert assurance(59, 6) == pytest.approx(0.842, abs=0.001)
#     assert assurance(59, 10, 0.0001) == pytest.approx(0.7798, abs=0.0001)

#     assert assurance(2, -2) is None
#     assert assurance(-2, 0) is None
