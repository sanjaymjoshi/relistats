from dataclasses import asdict, dataclass
from typing import Optional

import pytest

from relistats.binomial import (
    assurance,
    confidence,
    reliability,
    reliability_closed,
    reliability_optim,
)


@dataclass
class NFRMC:
    n: int
    f: int
    r: float
    m: Optional[int]
    c: float


test_nfrmc = (
    # infinite population
    NFRMC(1, 0, 0.5, None, 0.5),
    NFRMC(2, 0, 0.5, None, 0.75),
    NFRMC(2, 1, 0.5, None, 0.25),
    NFRMC(8, 0, 0.7, None, 0.942),
    NFRMC(8, 0, 0.9, None, 0.570),
    NFRMC(10, 9, 0.1, None, 0.349),
    NFRMC(100, 10, 0.9, None, 0.417),
    NFRMC(1000, 100, 0.9, None, 0.473),
    NFRMC(10000, 1000, 0.9, None, 0.492),

    # finite population
    NFRMC(8, 3, 0.5, 4, 0.973),
    NFRMC(8, 4, 0.5, 4, 0.363),
    NFRMC(8, 5, 0.5, 4, 0.004),
    NFRMC(8, 6, 0.5, 4, 0),
)


def test_confidence() -> None:

    # Confidence computation is exact, so set the tolerance tight
    ABS_TOL_CONFIDENCE = 0.001
    for x in test_nfrmc:
        print(f"Testing confidence: {asdict(x)}")
        assert confidence(x.n, x.f, x.r, x.m) == pytest.approx(x.c, abs=ABS_TOL_CONFIDENCE)

    # Infinite samples corner cases
    assert confidence(2, 2, 0.5) == 0
    assert confidence(2, 3, 0.5) == 0
    assert confidence(20, 0, 0) == 1
    assert confidence(20, 0, 1) == 0

    assert confidence(2, 0, 2) is None
    assert confidence(2, -2, 0.5) is None
    assert confidence(-2, 0, 0.5) is None
    assert confidence(2, 0, -0.5) is None

    # Finite samples corner cases
    assert confidence(2, 2, 0.5, -1 ) is None
    assert confidence(10, 0, 0.9, 0 ) == 1
    assert confidence(10, 0, 0.9, 1 ) == 1

    assert confidence(8, 0, 0.5, 1 ) == 1
    assert confidence(8, 0, 0.5, 2 ) == 1
    assert confidence(8, 0, 0.5, 4 ) == 1
    assert confidence(8, 1, 0.5, 4 ) == 1
    assert confidence(8, 2, 0.5, 4 ) == 1
    assert confidence(8, 7, 0.5, 4 ) == 0

def test_reliability_closed() -> None:
    # Reliability closed form computation is approximate, so set the tolerance loose
    ABS_TOL_RELIABILITY_CLOSED = 0.03

    for x in test_nfrmc:
        print(f"Testing reliability: {asdict(x)}")
        c1 = confidence(x.n, x.f, x.r)
        if c1 is None:
            c1 = 0
        assert reliability_closed(x.n, x.f, c1) == pytest.approx(
            x.r, abs=ABS_TOL_RELIABILITY_CLOSED
        )

    assert reliability_closed(2, 0, 2) is None
    assert reliability_closed(2, -2, 0.5) is None
    assert reliability_closed(-2, 0, 0.5) is None
    assert reliability_closed(2, 0, -0.5) is None


def test_reliability_optim() -> None:
    # Reliability computation via optimization is more accurate, so set the tolerance tight
    ABS_TOL_RELIABILITY_OPTIM = 0.001
    for x in test_nfrmc:
        print(f"Testing reliability: {asdict(x)}")
        c1 = confidence(x.n, x.f, x.r)
        if c1 is None:
            c1 = 0
        assert reliability_optim(x.n, x.f, c1) == pytest.approx(
            x.r, abs=ABS_TOL_RELIABILITY_OPTIM
        )

    assert reliability_optim(20, 0, 0.9, 0.0001) == pytest.approx(0.8912, abs=0.0001)

    assert reliability_optim(2, 0, 2) is None
    assert reliability_optim(2, -2, 0.5) is None
    assert reliability_optim(-2, 0, 0.5) is None
    assert reliability_optim(2, 0, -0.5) is None


def test_reliability() -> None:
    # Reliability computation should be accurate, so set the tolerance tight
    ABS_TOL_RELIABILITY = 0.001
    for x in test_nfrmc:
        print(f"Testing reliability: {asdict(x)}")
        c1 = confidence(x.n, x.f, x.r)
        if c1 is None:
            c1 = 0
        assert reliability(x.n, x.f, c1) == pytest.approx(x.r, abs=ABS_TOL_RELIABILITY)


def test_assurance() -> None:
    assert assurance(22, 0) == pytest.approx(0.9, abs=0.001)
    assert assurance(59, 0) == pytest.approx(0.95, abs=0.001)
    assert assurance(22, 2) == pytest.approx(0.812, abs=0.001)
    assert assurance(59, 6) == pytest.approx(0.842, abs=0.001)
    assert assurance(59, 10, 0.0001) == pytest.approx(0.7798, abs=0.0001)

    assert assurance(2, -2) is None
    assert assurance(-2, 0) is None
