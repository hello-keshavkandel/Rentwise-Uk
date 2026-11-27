"""Real-world validation: RentWise reproduces the figures ONS itself published.

Golden values from ONS, "Private rent and house prices, UK: November 2026" (released 18 Nov 2026,
data for October 2026) and ASHE 2025 Table 8.7a. If the data are refreshed to a newer month
these golden checks skip; update them from the new bulletin.
https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/privaterentandhousepricesuk/november2026
"""
import pandas as pd
import pytest

import config

P = config.DATA_PROCESSED
o = pd.read_csv(P / "official_rents.csv") if (P / "official_rents.csv").exists() else None
pytestmark = pytest.mark.skipif(o is None or o.period.max() != "2026-10",
                                reason="golden values are for the Oct 2026 PIPR release")

# (area, measure) -> (average monthly rent £, annual change %) as printed in the bulletin
BULLETIN = {
    ("United Kingdom", "All"): (1411, 3.8), ("England", "All"): (1472, 4.0), ("Wales", "All"): (849, 3.9),
    ("London", "All"): (2350, None), ("North East", "All"): (793, None),
    ("North West", "All"): (None, 5.6), ("South West", "All"): (None, 3.3),
    ("United Kingdom", "Detached"): (1603, None), ("United Kingdom", "Flat"): (1377, None),
    ("United Kingdom", "4+ bed"): (2097, None), ("United Kingdom", "1 bed"): (1145, None),
    ("Kensington and Chelsea", "All"): (3765, None), ("Oxford", "All"): (1956, None),
}


@pytest.mark.parametrize("key", list(BULLETIN))
def test_matches_ons_bulletin(key):
    area, measure = key
    rent, change = BULLETIN[key]
    r = o[(o.area_name == area) & (o.measure == measure)].iloc[0]
    if rent is not None:
        assert int(r.rent) == rent
    if change is not None:
        assert round(float(r.annual_change_pct), 1) == change


def test_highest_and_lowest_local_rents_match_bulletin():
    la = o[(o.level == "la") & (o.measure == "All")]
    assert la.sort_values("rent").iloc[-1].area_name == "Kensington and Chelsea"     # highest in the UK
    assert la[la.region != "London"].sort_values("rent").iloc[-1].area_name == "Oxford"  # highest outside London


def test_local_pay_is_fulltime_median_the_ons_affordability_basis():
    e = pd.read_csv(P / "earnings.csv").set_index("la_name")
    # ASHE 2025 Table 8.7a, residence-based medians (full-time vs all employee jobs)
    assert e.loc["Leeds", "median_pay_fulltime"] == 36716 and e.loc["Leeds", "median_pay_all"] == 31679
    assert (e.median_pay_fulltime > e.median_pay_all).all()              # full-time pay is always higher
    from src.models.area_data import AreaData
    d = AreaData()
    assert d.area_income(d.code_for("Leeds")) == 36716
