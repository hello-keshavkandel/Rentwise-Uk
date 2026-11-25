"""Phase A correctness rules for the official Module 2 dataset."""
import openpyxl
import pandas as pd
import pytest

from src.data.clean_data import DataCleaner
from src.data.rent_dataset import crime_counts, crime_rate, read_population, read_pipr


def test_crime_is_a_rate_per_resident_and_annualised():
    counts = pd.DataFrame({"crimes": [600.0, 600.0], "months_covered": [12.0, 6.0]}, index=["A", "B"])
    pop = pd.Series({"A": 100_000.0, "B": 10_000.0})
    r = crime_rate(counts, pop)
    assert r["A"] == pytest.approx(6.0)          # 600 / 100k * 1000
    assert r["B"] == pytest.approx(120.0)        # 600 over 6 months -> 1200 a year / 10k * 1000


def test_implausibly_low_rate_means_force_did_not_publish():
    # 30 crimes a year in a 500k city = 0.06 per 1,000: the home force did not publish
    counts = pd.DataFrame({"crimes": [30.0, 2500.0], "months_covered": [12.0, 12.0]}, index=["GM", "RUT"])
    pop = pd.Series({"GM": 500_000.0, "RUT": 41_000.0})
    r = crime_rate(counts, pop)
    assert pd.isna(r["GM"]) and r["RUT"] == pytest.approx(61.0, abs=0.1)


def test_areas_without_police_data_are_missing_not_zero():
    counts = pd.DataFrame({"crimes": [50.0], "months_covered": [2.0]}, index=["A"])
    pop = pd.Series({"A": 1000.0, "GMP_AREA": 500_000.0})
    r = crime_rate(counts, pop)
    assert pd.isna(r["GMP_AREA"])                # absent force -> NaN, never 0
    assert pd.isna(r["A"])                       # too few months (2 < 3) -> NaN


def _street(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Crime ID,Reported by,Falls within,LSOA code\n"
                    + "".join(f"{i},{r},{f},{l}\n" for i, (r, f, l) in enumerate(rows)))


def test_crime_counts_use_home_force_coverage(tmp_path):
    P = tmp_path / "police"
    # Kent publishes both months; L1/L2 are Kent areas.
    _street(P / "2026-01" / "2026-01-kent-street.csv",
            [("Kent Police", "Kent Police", "L1"), ("Kent Police", "Kent Police", "L1"),
             ("Kent Police", "Kent Police", "L2"),
             # a crime in a Greater Manchester LSOA logged by Kent: GM's own force published nothing
             ("Kent Police", "Greater Manchester Police", "LGM")])
    _street(P / "2026-02" / "2026-02-kent-street.csv", [("Kent Police", "Kent Police", "L1")])
    c = crime_counts(P, {"L1": "E07000001", "L2": "E07000002", "LGM": "E08000003"})
    assert c.loc["E07000001", "crimes"] == 3 and c.loc["E07000001", "months_covered"] == 2
    assert c.loc["E08000003", "home_force"] == "Greater Manchester Police"
    assert c.loc["E08000003", "months_covered"] == 0          # home force never published
    assert c.attrs["months"] == ["2026-01", "2026-02"]
    pop = pd.Series({"E07000001": 1000.0, "E07000002": 1000.0, "E08000003": 500000.0})
    r = crime_rate(c, pop, min_months=1)
    assert pd.isna(r["E08000003"])                             # missing, not ~0


def _pipr_xlsx(path):
    wb = openpyxl.Workbook()
    cov = wb.active; cov.title = "Cover sheet"
    cov.append(["The data tables in this spreadsheet were originally published at 9:30am on 18 November 2026."])
    t = wb.create_sheet("Table 1")
    t.append(["title"]); t.append(["note"])
    t.append(["Time period", "Area code", "Area name", "Region or country name", "Rental price detached",
              "Rental price semidetached", "Rental price terraced", "Rental price flat maisonette"])
    from datetime import datetime
    for m in (9, 10):
        t.append([datetime(2026, m, 1), "E09000033", "Westminster", "London", 9000, 7000, "[x]", 3200 + m])
        t.append([datetime(2026, m, 1), "E92000001", "England", "England", 1, 1, 1, 1])
    wb.save(path)


def test_read_pipr_keeps_every_official_rent_and_records_vintage(tmp_path):
    p = tmp_path / "ons_pipr_monthly.xlsx"; _pipr_xlsx(p)
    cur, hist, meta = read_pipr(p)
    assert meta["period"] == "2026-10" and meta["history_from"] == "2026-09"
    assert meta["released"] == "18 November 2026"
    assert set(cur.property_type) == {"Detached", "Semi-detached", "Flat"}   # [x] skipped, not imputed
    assert cur.loc[cur.property_type == "Detached", "rent"].iloc[0] == 9000  # very high, kept
    assert len(hist) == 6 and "E92000001" not in set(hist.la_code)


def test_read_population_finds_la_table(tmp_path):
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "MYE2 - Persons"
    ws.append(["Mid-2025 population"]); ws.append([])
    ws.append(["Code", "Name", "Geography", "All ages"])
    ws.append(["E08000016", "Barnsley", "Metropolitan District", 250000])
    ws.append(["E92000001", "England", "Country", 58000000])
    p = tmp_path / "mye25tablesew.xlsx"; wb.save(p)
    pop, meta = read_population(p)
    assert pop.to_dict() == {"E08000038": 250000.0}      # Barnsley re-coded to the ONS rent code
    assert meta["period"] == "mid-2025"


def test_cleaner_official_mode_keeps_high_rents_and_gaps(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "DATA_PROCESSED", tmp_path)
    df = pd.DataFrame({"la_code": ["A", "B", "C", "D"], "la_name": ["A", "B", "C", "D"],
                       "property_type": ["Flat"] * 4, "bedrooms": [2] * 4,
                       "median_annual_income": [30000, 40000, 35000, 32000],
                       "crime_per_1000": [5.0, None, 7.0, 6.0],
                       "median_monthly_rent": [900, 4645, 1000, 50]})     # 50 is impossible
    out = DataCleaner(mode="official").clean(df)
    assert 4645 in set(out.median_monthly_rent)          # genuine high rent kept
    assert 50 not in set(out.median_monthly_rent)        # impossible value dropped
    assert out.crime_per_1000.isna().sum() == 1          # gap stays a gap


def test_cleaner_rejects_unknown_mode():
    with pytest.raises(ValueError):
        DataCleaner(mode="loose")
