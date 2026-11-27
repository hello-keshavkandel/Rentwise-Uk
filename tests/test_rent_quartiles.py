"""ONS median & quartile rents: parsing rules, consistency with the official averages,
golden values from the ONS release, and how the app/API present them."""
from datetime import datetime

import openpyxl
import pandas as pd
import pytest

import config
from src.data.rent_quartiles import read_quartiles

P = config.DATA_PROCESSED
HAVE = (P / "rent_quartiles.csv").exists()
needs = pytest.mark.skipif(not HAVE, reason="rent quartiles not built")
HDR = ["Time period", "Area code", "Area name", "Region or country name", "Count of All categories",
       " All Lower Quartile (£)", "All Median (£)", "All Upper quartile (£)", "Count of rents Two Bedrooms",
       "Lower quartile Two Bedrooms (£)", "Median Two Bedrooms (£)", "Upper quartile Two Bedrooms (£)"]


def _xlsx(path):
    wb = openpyxl.Workbook()
    wb.active.title = "Cover sheet"
    wb.active.append(["originally published at 9:30am on 18 November 2026. Crown copyright"])
    t = wb.create_sheet("Table 1")
    t.append(["title"]); t.append(["note"]); t.append(HDR)
    for m in (8, 9):
        t.append([datetime(2026, m, 1), "E08000035", "Leeds", "Yorkshire", 6404, 827.4, 953.3, 1062.3,
                  "[c]", "[u]", 961.5, "[u]"])                    # 6-10 rents: median kept, quartiles suppressed
        t.append([datetime(2026, m, 1), "E06000053", "Isles of Scilly", "South West", 0, "[x]", "[x]", "[x]",
                  0, "[x]", "[x]", "[x]"])
        t.append([datetime(2026, m, 1), "S12000036", "Edinburgh", "Scotland", 9, 1, 1, 1, 9, 1, 1, 1])
    wb.save(path)


def test_parser_keeps_latest_month_and_never_fills_suppressed_cells(tmp_path):
    p = tmp_path / "q.xlsx"; _xlsx(p)
    df, meta = read_quartiles(p)
    assert meta["period"] == "2026-09" and meta["released"] == "18 November 2026"
    assert set(df.period) == {"2026-09"}                        # snapshot: latest month only
    assert "S12000036" not in set(df.area_code)                 # Scotland not in scope
    leeds2 = df[(df.area_code == "E08000035") & (df.measure == "2 bed")].iloc[0]
    assert leeds2["median"] == 962 and pd.isna(leeds2.lower_quartile) and pd.isna(leeds2["count"])
    scilly = df[df.area_code == "E06000053"]
    assert scilly["median"].isna().all()                        # [x] -> missing, not 0


@needs
def test_quartiles_are_ordered_and_cover_every_area():
    q = pd.read_csv(P / "rent_quartiles.csv")
    la = q[q.level == "la"]
    assert la.area_code.nunique() == 316
    full = la.dropna(subset=["lower_quartile", "median", "upper_quartile"])
    assert ((full.lower_quartile <= full["median"]) & (full["median"] <= full.upper_quartile)).all()
    assert la.loc[la.measure.isin(["All", "1 bed", "2 bed", "3 bed", "4+ bed"]), "median"].notna().all()


@needs
def test_bedroom_medians_agree_with_the_official_averages():
    """Two independent ONS measures from the same rents: by bedroom count the LA median should be
    close to the PIPR average, and the average should usually sit inside the middle half."""
    q = pd.read_csv(P / "rent_quartiles.csv")
    o = pd.read_csv(P / "official_rents.csv")
    m = q[q.level == "la"].merge(o[o.level == "la"], on=["area_code", "measure"])
    for meas in ["1 bed", "2 bed", "3 bed"]:
        g = m[m.measure == meas]
        assert g["median"].corr(g.rent) > 0.99
        assert 0.95 < (g["median"] / g.rent).median() < 1.03
        inside = ((g.rent >= g.lower_quartile) & (g.rent <= g.upper_quartile)).mean()
        assert inside > 0.95


@needs
@pytest.mark.skipif(not HAVE or pd.read_csv(P / "rent_quartiles.csv").period.max() != "2026-09",
                    reason="golden values are for the Sep 2026 snapshot")
def test_golden_values_from_the_ons_release():
    q = pd.read_csv(P / "rent_quartiles.csv").set_index(["area_name", "measure"])
    assert q.loc[("England", "All"), "median"] == 1052                     # ONS: £1,051.75
    assert q.loc[("Leeds", "2 bed"), ["lower_quartile", "median", "upper_quartile"]].tolist() == [913, 962, 1015]
    assert q.loc[("Westminster", "2 bed"), "median"] == 3230


@needs
def test_area_data_range_and_compare_columns():
    from src.models.area_data import AreaData
    d = AreaData()
    leeds = d.code_for("Leeds")
    r = d.rent_range(leeds, "2 bed")
    assert r["lower_quartile"] < r["median"] < r["upper_quartile"] and r["count"] > 1000
    assert d.rent_range(leeds, "Flat") is None                  # ONS publishes ranges by bedrooms only
    df = d.compare([leeds, d.code_for("Manchester")], "2 bed")
    assert df["Median rent (£/month)"].notna().all() and "Middle half of rents (£/month)" in df
    t = d.all_areas_table()
    assert t.loc[(t.la_code == leeds) & (t.measure == "2 bed"), "median_rent"].iloc[0] == r["median"]


@needs
@pytest.mark.skipif(not (config.MODELS_DIR / "rent_model_meta.json").exists(), reason="model not built")
def test_app_shows_what_most_people_pay():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(config.ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    at.sidebar.selectbox[0].set_value("Leeds"); at.sidebar.selectbox[1].set_value("2 bed"); at.run()
    assert not at.exception
    assert any("What most people pay" in m.value and "£913" in m.value for m in at.markdown)
