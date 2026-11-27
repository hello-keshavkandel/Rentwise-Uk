"""Phase C: official bedroom-level rents, personal affordability, sale prices & yield,
regional forecasts with honest bands, area comparison and the new API endpoints."""
import json
from datetime import datetime

import numpy as np
import openpyxl
import pandas as pd
import pytest

import config
from src.data.official_rents import read_pipr_all
from src.data.sale_prices import MIN_SALES, norm_name, summarise
from src.models.area_data import AreaData, affordability
from src.models.rent_forecast import backtest, conformal_widths, coverage

P = config.DATA_PROCESSED
HAVE_DATA = (P / "official_rents.csv").exists() and (P / "rent_trends.parquet").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="official rents not built")


# ------------------------------------------------------------------ official rents (unit)
def _pipr(path):
    wb = openpyxl.Workbook()
    wb.active.title = "Cover sheet"
    wb.active.append(["originally published at 9:30am on 18 November 2026."])
    t = wb.create_sheet("Table 1")
    t.append(["t"]); t.append(["n"])
    t.append(["Time period", "Area code", "Area name", "Region or country name",
              "Rental price", "Annual change", "Rental price two bed", "Annual change two bed",
              "Rental price flat maisonette", "Annual change flat maisonette"])
    for m in (9, 10):
        t.append([datetime(2026, m, 1), "E09000033", "Westminster", "London", 3200, 1.5, 3300, "[x]", 2974, -0.2])
        t.append([datetime(2026, m, 1), "E12000007", "London", "London", 2350, 3.8, 2400, 3.1, 2100, 2.0])
        t.append([datetime(2026, m, 1), "S12000036", "Edinburgh", "Scotland", 1500, 1, 1500, 1, 1500, 1])
    wb.save(path)


def test_read_pipr_all_keeps_every_measure_level_and_change(tmp_path):
    p = tmp_path / "pipr.xlsx"; _pipr(p)
    cur, hist, meta = read_pipr_all(p)
    assert meta["period"] == "2026-10" and meta["released"] == "18 November 2026"
    assert set(meta["measures"]) == {"All", "2 bed", "Flat"}           # only what the file has
    assert set(cur.level) == {"la", "region"}                           # Scotland not served
    w = cur[(cur.area_code == "E09000033")].set_index("measure")
    assert w.loc["2 bed", "rent"] == 3300 and pd.isna(w.loc["2 bed", "annual_change_pct"])  # [x] -> missing
    assert w.loc["Flat", "annual_change_pct"] == pytest.approx(-0.2)
    assert len(hist) == 2 * len(cur)


@needs_data
def test_shipped_official_rents_cover_every_area_and_measure():
    o = pd.read_csv(P / "official_rents.csv")
    la = o[o.level == "la"]
    assert la.area_code.nunique() == 316
    assert la.groupby("area_code").measure.nunique().min() == 9
    assert set(o.loc[o.level == "region", "area_name"]) >= {"London", "North West", "North East"}


@needs_data
def test_property_type_rents_match_the_model_dataset_exactly():
    o = pd.read_csv(P / "official_rents.csv")
    c = pd.read_csv(P / "rentwise_clean.csv")
    m = c.merge(o, left_on=["la_code", "property_type"], right_on=["area_code", "measure"])
    assert len(m) == len(c) and (m.rent == m.median_monthly_rent).all()


@needs_data
def test_trends_run_from_2015_to_the_latest_month():
    t = pd.read_parquet(P / "rent_trends.parquet")
    o = pd.read_csv(P / "official_rents.csv")
    assert t.period.min() == "2015-01" and t.period.max() == o.period.max()
    assert (t.rent > 0).all()


# ------------------------------------------------------------------ sale prices (unit)
def test_district_names_match_ons_names():
    assert norm_name("Bristol, City of") == norm_name("CITY OF BRISTOL")
    assert norm_name("Kingston upon Hull, City of") == norm_name("CITY OF KINGSTON UPON HULL")
    assert norm_name("The Vale of Glamorgan") == norm_name("Vale of Glamorgan")
    assert norm_name("RHONDDA CYNON TAFF") == norm_name("Rhondda Cynon Taf")
    assert norm_name("WREKIN") == norm_name("Telford and Wrekin")
    assert norm_name("St. Helens") == norm_name("ST HELENS")


def test_sale_price_summary_rules():
    rows = ([{"price": 200_000, "date": "2026-02-01", "ptype": "T", "district": "LEEDS", "category": "A"}] * MIN_SALES
            + [{"price": 9_000_000, "date": "2026-03-01", "ptype": "T", "district": "LEEDS", "category": "B"}] * 50
            + [{"price": 300_000, "date": "2026-03-01", "ptype": "F", "district": "LEEDS", "category": "A"}] * 3
            + [{"price": 1, "date": "2026-03-01", "ptype": "O", "district": "LEEDS", "category": "A"}]
            + [{"price": 5, "date": "2026-03-01", "ptype": "T", "district": "NOWHERE", "category": "A"}])
    out, meta = summarise(pd.DataFrame(rows), {"E08000035": "Leeds"})
    t = out.set_index("property_type")
    assert t.loc["Terraced", "median_price"] == 200_000            # category B ignored
    assert pd.isna(t.loc["Flat", "median_price"]) and t.loc["Flat", "sales"] == 3   # < MIN_SALES -> missing
    assert "Other" not in t.index                                   # 'O' type excluded
    assert meta["unmatched_districts"] == ["NOWHERE"]               # reported, never guessed


# ------------------------------------------------------------------ affordability
@pytest.mark.parametrize("rent,income,band,ok", [
    (750, 30_000, "within", True),        # exactly 30%
    (900, 30_000, "above the 30%", False),  # 36%
    (1100, 30_000, "well above", False),  # 44%
])
def test_affordability_bands(rent, income, band, ok):
    a = affordability(rent, income)
    assert band in a["band"] and a["affordable"] is ok
    assert a["max_rent_at_30pct"] == 750


def test_affordability_needs_both_numbers():
    assert affordability(None, 30_000) is None and affordability(900, 0) is None


# ------------------------------------------------------------------ forecasts
def test_conformal_widths_cover_and_grow_with_horizon():
    rng = np.random.default_rng(0)
    rel = rng.normal(0, 1, (400, 12)) * np.linspace(0.002, 0.02, 12)
    w = conformal_widths(rel)
    assert (np.diff(w[0.8]) > 0).sum() >= 10 and (w[0.95] > w[0.8]).all()
    cov = coverage(rel, w)
    assert cov[0.8] == pytest.approx(0.8, abs=0.03) and cov[0.95] >= 0.94


def test_backtest_on_a_smooth_series_is_accurate():
    y = 1000 * np.exp(np.arange(120) * 0.003)
    rel = backtest(y, "Damped-trend ETS", h=6, n_origins=4)
    assert rel.shape == (4, 6) and np.abs(rel).max() < 0.02


@needs_data
@pytest.mark.skipif(not (config.MODELS_DIR / "rent_forecasts.json").exists(), reason="forecasts not built")
def test_shipped_forecasts_are_sane_and_honest():
    f = json.loads((config.MODELS_DIR / "rent_forecasts.json").read_text())
    assert f["method"]["chosen"] in f["method"]["backtest_mape_pct"]
    assert set(f["method"]["out_of_time_coverage"]) == {"80%", "95%"}      # coverage is reported
    assert len(f["series"]) == 12                                          # 9 regions + E, W, UK
    for s in f["series"].values():
        assert len(s["forecast"]) == 12
        for p in s["forecast"]:
            assert p["low95"] <= p["low80"] <= p["rent"] <= p["high80"] <= p["high95"]
        assert 0.8 < s["forecast"][-1]["rent"] / s["last_rent"] < 1.25


# ------------------------------------------------------------------ area data
@needs_data
def test_area_data_answers_consistently():
    d = AreaData()
    cam, man, car = d.code_for("Camden"), d.code_for("Manchester"), d.code_for("Cardiff")
    assert d.official_rent(cam, "2 bed") > d.official_rent(man, "2 bed")
    assert d.gross_yield(cam, "2 bed") is None            # no bedroom data in Price Paid -> no yield
    y = d.gross_yield(man, "Flat")
    assert y is None or 2 < y < 15
    assert d.region_forecast(car)["name"] == "Wales"
    assert d.region_forecast(cam)["name"] == "London"
    df = d.compare([cam, man], "Flat", income=40_000)
    assert list(df.Area) == ["Camden", "Manchester"]
    assert pd.isna(df.loc[df.Area == "Manchester", "Crime per 1,000 residents"]).all()   # missing, not 0
    assert "Rent / your income (%)" in df.columns
    t = d.all_areas_table()
    assert len(t) == 316 * 9 and t.rent.notna().all()


# ------------------------------------------------------------------ API (real data)
@needs_data
@pytest.mark.skipif(not (config.MODELS_DIR / "rent_model_meta.json").exists(), reason="model not built")
class TestApiPhaseC:
    @pytest.fixture(scope="class")
    def ctx(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from src.api import security as sec
        from src.api.app import create_app
        from src.api.db import Database
        db = Database(":memory:")
        c = TestClient(create_app(db=db, rate_limit=1000))
        key, prefix, h = sec.new_api_key()
        db.run("INSERT INTO api_keys (user_id, name, prefix, key_hash, created_at) VALUES (NULL,'t',?,?,'t')",
               (prefix, h))
        return c, {"X-API-Key": key}

    def test_area_with_no_police_data_returns_null_crime(self, ctx):
        c, H = ctx
        r = c.get("/v1/areas/E08000003", headers=H)            # Manchester
        assert r.status_code == 200 and r.json()["crime_per_1000"] is None

    def test_rents_endpoint_has_bedrooms_and_personal_affordability(self, ctx):
        c, H = ctx
        j = c.get("/v1/areas/E08000035/rents?income_gbp_year=40000", headers=H).json()
        by = {r["measure"]: r for r in j["rents"]}
        assert {"All", "1 bed", "2 bed", "3 bed", "4+ bed", "Flat"} <= set(by)
        assert by["2 bed"]["vs_your_income"]["ratio"] == pytest.approx(by["2 bed"]["rent_gbp_month"] * 12 / 40000,
                                                                       abs=1e-4)
        assert by["2 bed"]["gross_yield_pct"] is None and j["sources"]["rents"]["period"]

    def test_trend_and_forecast(self, ctx):
        c, H = ctx
        t = c.get("/v1/areas/E08000035/trend?measure=2 bed", headers=H).json()
        assert t["points"][0]["period"] == "2015-01"
        assert c.get("/v1/areas/E08000035/trend?measure=5 bed", headers=H).status_code == 422
        f = c.get("/v1/areas/W06000015/forecast", headers=H).json()
        assert f["region"] == "Wales" and len(f["forecast"]) == 12

    def test_compare_and_validation(self, ctx):
        c, H = ctx
        j = c.get("/v1/compare?codes=E08000035,E08000003&measure=Flat", headers=H).json()
        assert [a["Area"] for a in j["areas"]] == ["Leeds", "Manchester"]
        assert c.get("/v1/compare?codes=E08000035", headers=H).status_code == 422
        assert c.get("/v1/compare?codes=E08000035,E99999999", headers=H).status_code == 404

    def test_predict_with_income(self, ctx):
        c, H = ctx
        j = c.post("/v1/predict", headers=H,
                   json={"la_code": "E09000007", "property_type": "Flat", "income_gbp_year": 60000}).json()
        assert j["your_affordability"]["ratio"] == pytest.approx(j["rent_gbp_month"] * 12 / 60000, abs=1e-4)
        assert c.post("/v1/predict", headers=H, json={"la_code": "E09000007", "property_type": "Flat",
                                                      "income_gbp_year": -5}).status_code == 422
