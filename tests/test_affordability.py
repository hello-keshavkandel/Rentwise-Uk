"""
test_affordability.py  —  I unit-test the affordability engine's reusable logic.

The dashboard and the reporting both rely on these two methods, so I lock their
behaviour down: the ratio is annual rent over annual income, and an area is
affordable only at or below the 30% benchmark.
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.affordability import AffordabilityEngine


@pytest.fixture(scope="module")
def engine():
    return AffordabilityEngine()


def test_ratio_is_rent_over_income(engine):
    # £1,000/month on £40,000/year = 12,000 / 40,000 = 0.30.
    assert engine.ratio(12000, 40000) == pytest.approx(0.30)


def test_threshold_boundary_is_affordable(engine):
    # Exactly 30% is affordable; just above is not.
    assert engine.is_affordable(0.30) is True
    assert engine.is_affordable(0.3001) is False


def test_clearly_affordable_and_unaffordable(engine):
    assert engine.is_affordable(engine.ratio(9000, 40000)) is True   # 22.5%
    assert engine.is_affordable(engine.ratio(18000, 40000)) is False  # 45%
