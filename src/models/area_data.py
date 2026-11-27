"""
area_data.py  —  the official, non-model facts about each area, in one place (Phase C).

RentWisePredictor answers "explain / estimate a rent". This class answers everything else
users asked for, from official data only, so the app and the API give identical answers:

  * official rents for every measure (all, 1/2/3/4+ bed, 4 property types) + annual change
  * rent history since 2015, and the area's REGIONAL 12-month forecast with conformal bands
  * median sale prices (HM Land Registry) and indicative gross rental yield
  * affordability for the user's OWN income, or the area's median earnings (ASHE)
  * side-by-side comparison of several areas, as a table ready for CSV download

Every figure carries its source and period. Missing is missing (None) — never zero.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

BENCHMARK = config.AFFORDABILITY_THRESHOLD          # 0.30: rent at most 30% of gross income
SEVERE = 0.40
MEASURE_ORDER = ["All", "1 bed", "2 bed", "3 bed", "4+ bed", "Flat", "Terraced", "Semi-detached", "Detached"]
MEASURE_LABEL = {"All": "All properties", "1 bed": "1 bedroom", "2 bed": "2 bedrooms", "3 bed": "3 bedrooms",
                 "4+ bed": "4 or more bedrooms", "Flat": "Flat / maisonette", "Terraced": "Terraced house",
                 "Semi-detached": "Semi-detached house", "Detached": "Detached house"}
TYPE_MEASURES = {"Flat", "Terraced", "Semi-detached", "Detached"}


def _clean(v):
    """NaN/None -> None; numpy scalars -> Python numbers (JSON-safe)."""
    if v is None:
        return None
    try:
        if isinstance(v, float) and math.isnan(v):
            return None
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v.item() if hasattr(v, "item") else v


def affordability(monthly_rent: float | None, annual_income: float | None) -> dict | None:
    """Share of gross annual income spent on rent, with a plain-English band."""
    if not monthly_rent or not annual_income or annual_income <= 0:
        return None
    r = monthly_rent * 12 / annual_income
    band = ("within the 30% guideline" if r <= BENCHMARK else
            "above the 30% guideline" if r <= SEVERE else "well above the guideline (over 40%)")
    return {"ratio": round(r, 4), "affordable": r <= BENCHMARK, "band": band,
            "max_rent_at_30pct": int(annual_income * BENCHMARK / 12)}


class AreaData:
    def __init__(self, processed_dir: Path | None = None, models_dir: Path | None = None):
        P = Path(processed_dir or config.DATA_PROCESSED)
        M = Path(models_dir or config.MODELS_DIR)
        o = pd.read_csv(P / "official_rents.csv")
        self.official = o[o.level == "la"].copy()
        self.region_official = o[o.level != "la"].copy()
        self.period = str(o["period"].max())
        names = pd.read_csv(P / "rent_areas.csv")
        self.region_codes = dict(zip(names.loc[names.level == "region", "area_name"],
                                     names.loc[names.level == "region", "area_code"]))
        self.nation_codes = dict(zip(names.loc[names.level == "nation", "area_name"],
                                     names.loc[names.level == "nation", "area_code"]))
        clean = pd.read_csv(P / "rentwise_clean.csv")
        a = clean.groupby("la_code").agg(la_name=("la_name", "first"), region=("region", "first"),
                                         income=("median_annual_income", "first"),
                                         crime=("crime_per_1000", "first"))
        self.areas = a
        self.trends_path = P / "rent_trends.parquet"
        self._trends = None
        sp = P / "sale_prices.csv"
        self.sales = pd.read_csv(sp) if sp.exists() else None
        fp = M / "rent_forecasts.json"
        self.forecasts = json.loads(fp.read_text()) if fp.exists() else None
        sf = P / "la_spatial_features.csv"
        self.spatial = pd.read_csv(sf).set_index("lad_code") if sf.exists() else None
        vp = P / "data_vintages.json"
        self.vintages = json.loads(vp.read_text()) if vp.exists() else {}

    # ---------------------------------------------------------------- lookups
    def code_for(self, la_name: str) -> str:
        m = self.areas.index[self.areas.la_name == la_name]
        if len(m) == 0:
            raise KeyError(la_name)
        return m[0]

    def measures(self) -> list[str]:
        have = set(self.official["measure"])
        return [m for m in MEASURE_ORDER if m in have]

    def official_rents(self, la_code: str) -> dict:
        """measure -> {rent, annual_change_pct, period} for every measure ONS publishes here."""
        rows = self.official[self.official.area_code == la_code]
        return {r.measure: {"rent": int(r.rent), "annual_change_pct": _clean(round(r.annual_change_pct, 2))
                            if pd.notna(r.annual_change_pct) else None, "period": r.period}
                for r in rows.itertuples()}

    def official_rent(self, la_code: str, measure: str) -> int | None:
        return (self.official_rents(la_code).get(measure) or {}).get("rent")

    def sale_price(self, la_code: str, property_type: str = "All") -> dict | None:
        if self.sales is None:
            return None
        m = self.sales[(self.sales.la_code == la_code) & (self.sales.property_type == property_type)]
        if m.empty:
            return None
        r = m.iloc[0]
        return {"median_price": _clean(r.median_price), "sales": int(r.sales),
                "period": (self.vintages.get("sale_prices") or {}).get("period")}

    def gross_yield(self, la_code: str, measure: str = "All") -> float | None:
        """Annual official rent / median sale price, % (types and 'All' only)."""
        if measure != "All" and measure not in TYPE_MEASURES:
            return None
        rent, sp = self.official_rent(la_code, measure), self.sale_price(la_code, measure)
        if not rent or not sp or not sp["median_price"]:
            return None
        return round(rent * 12 / sp["median_price"] * 100, 2)

    def area_income(self, la_code: str) -> float | None:
        return _clean(self.areas.loc[la_code, "income"]) if la_code in self.areas.index else None

    # ---------------------------------------------------------------- trends & forecasts
    def trends(self) -> pd.DataFrame:
        if self._trends is None:
            self._trends = pd.read_parquet(self.trends_path)
        return self._trends

    def trend(self, area_code: str, measure: str = "All") -> pd.DataFrame:
        t = self.trends()
        s = t[(t.area_code == area_code) & (t.measure == measure)][["period", "rent"]]
        return s.sort_values("period").reset_index(drop=True)

    def region_forecast(self, la_code: str) -> dict | None:
        """The 12-month forecast for the area's region (Wales for Welsh areas)."""
        if not self.forecasts or la_code not in self.areas.index:
            return None
        region = self.areas.loc[la_code, "region"]
        code = self.region_codes.get(region) or self.nation_codes.get(region)
        s = self.forecasts["series"].get(code)
        return None if s is None else {"area_code": code, **s, "method": self.forecasts["method"]}

    # ---------------------------------------------------------------- comparison
    def compare(self, la_codes: list[str], measure: str = "All", income: float | None = None) -> pd.DataFrame:
        """One row per area: official rent, change, affordability, prices, yield, crime, schools, amenities."""
        rows = []
        for c in la_codes:
            if c not in self.areas.index:
                continue
            a = self.areas.loc[c]
            off = self.official_rents(c).get(measure, {})
            rent = off.get("rent")
            own = affordability(rent, income) if income else None
            local = affordability(rent, _clean(a.income))
            sp = self.sale_price(c, measure if measure in TYPE_MEASURES else "All")
            sx = self.spatial.loc[c] if self.spatial is not None and c in self.spatial.index else None

            def per_km2(col):
                return None if sx is None or pd.isna(sx.get(col)) else round(math.expm1(sx[col]), 2)
            rows.append({
                "Area": a.la_name, "Code": c, "Region": a.region,
                f"Official rent, {MEASURE_LABEL.get(measure, measure)} (£/month)": rent,
                "Annual change (%)": off.get("annual_change_pct"),
                "Rent period": off.get("period"),
                "Local median earnings (£/year)": _clean(a.income),
                "Rent / local earnings (%)": round(local["ratio"] * 100, 1) if local else None,
                **({"Rent / your income (%)": round(own["ratio"] * 100, 1) if own else None} if income else {}),
                ("Median sale price, all homes (£)" if measure not in TYPE_MEASURES
                 else "Median sale price (£)"): sp["median_price"] if sp else None,
                # yield only where rent and price describe the same homes (types / all);
                # Price Paid has no bedroom counts, so bedroom measures get no yield
                "Gross yield (%)": self.gross_yield(c, measure),
                "Crime per 1,000 residents": None if pd.isna(a.crime) else round(float(a.crime), 1),
                "Schools Good/Outstanding (%)": (None if sx is None or pd.isna(sx.get("share_good_or_better"))
                                                 else round(sx["share_good_or_better"] * 100)),
                "Rail stations per km²": per_km2("dens_rail_station"),
                "GP surgeries per km²": per_km2("dens_gp_clinic"),
                "Supermarkets per km²": per_km2("dens_supermarket"),
                "Parks per km²": per_km2("dens_park"),
            })
        return pd.DataFrame(rows)

    def all_areas_table(self) -> pd.DataFrame:
        """Every area x every official measure — the analyst download."""
        o = self.official.merge(self.areas[["income"]], left_on="area_code", right_index=True, how="left")
        o["rent_to_local_earnings_pct"] = (o.rent * 12 / o.income * 100).round(1)
        o = o.rename(columns={"area_code": "la_code", "area_name": "la_name", "income": "median_annual_earnings"})
        return o[["la_code", "la_name", "region", "period", "measure", "rent", "annual_change_pct",
                  "median_annual_earnings", "rent_to_local_earnings_pct"]]

    def official_affordability(self) -> dict[str, float]:
        """la_code -> rent/earnings for ALL properties (official, for the map)."""
        o = self.official[self.official.measure == "All"].set_index("area_code")["rent"]
        inc = self.areas["income"]
        return {c: float(o[c] * 12 / inc[c]) for c in o.index if c in inc.index and pd.notna(inc[c])}
