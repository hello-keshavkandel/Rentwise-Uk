"""
streamlit_app.py  —  Phase 9 / Module 4 integration: the RentWise UK dashboard.

the single user-facing app for the whole four-module system. The sidebar
switches between two views:
  * Rent estimator  - the interactive Module 2 tool (rent + affordability + SHAP).
  * System results  - a read-only overview of all four modules' outputs.

Why this design : only dropdowns/sliders (no free text,
no uploads, no code execution) so there is no injection surface; the model is
loaded read-only and cached; every estimate carries a plain-English explanation
and an 'indicative, not financial advice' disclaimer.

Run it:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from src.models.predict import RentWisePredictor

st.set_page_config(page_title="RentWise UK", page_icon="🏠", layout="wide")
st.markdown("""<style>
 .main .block-container {padding-top: 2rem; max-width: 1100px;}
 h1 {color:#1F3864;} .stMetric{background:#F4F8FD;border:1px solid #E1E8F0;border-radius:12px;padding:14px;}
 .disclaimer{color:#6b7280;font-size:0.85rem;}
</style>""", unsafe_allow_html=True)

FRIENDLY = {"bedrooms":"Number of bedrooms","median_annual_income":"Local income",
 "crime_per_1000":"Crime per 1,000 residents",
 "property_type_ord":"Property type / size","income_band":"Area income band","year":"Year",
 "school_quality":"Local school quality (historic Ofsted)",
 "share_good_or_better":"Share of Good/Outstanding schools"}
AMENITY = {"rail_station":"Rail stations","bus_stop":"Bus stops","supermarket":"Supermarkets",
 "convenience":"Convenience shops","gp_clinic":"GP surgeries & clinics","hospital":"Hospitals",
 "pharmacy":"Pharmacies","restaurant_cafe":"Restaurants & cafés","pub_bar":"Pubs & bars",
 "park":"Parks","gym":"Gyms"}
def friendly(f):
    if f.endswith("__missing"):
        base = f[:-len("__missing")]
        if base.startswith(("school", "share_good")):
            return "School data not available (Wales — schools are inspected by Estyn)"
        return f"{FRIENDLY.get(base, base)}: no data for this area"
    if f.startswith("region_"):
        return "Being in " + f.replace("region_", "")
    if f.startswith("dens_"):
        return f"Density of {AMENITY.get(f[5:], f[5:]).lower()}"
    return FRIENDLY.get(f, f)

def month_label(period):
    """'2026-10' -> 'October 2026'."""
    import calendar
    try:
        y, m = str(period).split("-")[:2]
        return f"{calendar.month_name[int(m)]} {y}"
    except Exception:
        return "the latest release"

def amenity_per_km2(log_density):
    import math
    return math.expm1(log_density) if log_density is not None else None

@st.cache_resource
def get_predictor():
    return RentWisePredictor()

st.title("🏠 RentWise UK")
st.caption("Data-driven UK housing & long-term rental intelligence — predict, explain and map rents from open data")

view = st.sidebar.radio("View", ["Rent estimator", "Area map", "System results"])

# ---- observability: data/model freshness banner + methodology ----
with st.sidebar:
    try:
        from src.health import check
        _h = check()
        _icon = {"ok": "🟢", "degraded": "🟡", "down": "🔴"}.get(_h["status"], "⚪")
        _d = _h["data"].get("age_days"); _m = _h["model"].get("age_days")
        _b = _h["data"].get("backend")
        st.caption(f"{_icon} System {_h['status']}"
                   + (f" · data via {_b}" if _b else "")
                   + (f" · data {int(_d)}d old" if _d is not None else "")
                   + (f" · model {int(_m)}d old" if _m is not None else ""))
    except Exception:
        pass
    with st.expander("Data sources & dates"):
        try:
            import json as _json
            _v = _json.loads((config.DATA_PROCESSED / "data_vintages.json").read_text())
            _r, _e, _p, _c = (_v.get(k, {}) for k in ("rents", "earnings", "population", "crime"))
            st.markdown(
                f"- **Rents:** {_r.get('source')} — {month_label(_r.get('period'))}"
                f"{' (released ' + _r['released'] + ')' if _r.get('released') else ''}\n"
                f"- **Earnings:** {_e.get('source')} — {_e.get('period')}\n"
                f"- **Population:** {_p.get('source')} — {_p.get('period')}\n"
                f"- **Crime:** {_c.get('measure')} — {_c.get('period')} ({_c.get('months')} months)\n"
                "- **Amenities:** OpenStreetMap via Geofabrik — Nov 2026 · **Schools:** Ofsted & DfE — Oct 2026\n"
                "- **Boundaries:** ONS, May 2024")
            if _c.get("areas_without_police_data"):
                st.caption("No police data for: " + ", ".join(_c["areas_without_police_data"])
                           + " — shown as missing, never as zero.")
            if _c.get("areas_with_partial_months"):
                st.caption("Crime based on fewer months for: " + ", ".join(
                    f"{k} ({m})" for k, m in _c["areas_with_partial_months"].items()))
        except Exception:
            st.caption("Data dates unavailable — run the pipeline to rebuild them.")
    with st.expander("About & methodology"):
        st.markdown(
            "**RentWise UK** predicts UK rents and sale prices from **open government "
            "data** (ONS, HM Land Registry, data.police.uk) and explains every estimate.\n\n"
            "- **Rent (Modules 2+3):** XGBoost on income, crime, property type **plus local amenities "
            "and schools**. The **official ONS median** is shown first; the model (R² 0.95 served, 0.86 unseen) "
            "explains it with SHAP and estimates only where ONS publishes none; "
            "affordability vs the 30% income benchmark; **calibrated 80% intervals** (conformal).\n"
            "- **Sale price (Module 1):** HM Land Registry + EPC floor area/rooms; XGBoost R² 0.73 "
            "(0.57 without EPC).\n"
            "- **Geospatial (Module 3):** 506k OpenStreetMap amenities + historic Ofsted grades "
            "(18k schools) per area; see the **Area map**.\n"
            "- **Forecasting (Module 4):** ARIMA / Holt-Winters regional trends; rolling-origin "
            "backtest with recalibrated (conformal) 80/95% bands.\n\n"
            "Estimates are **indicative, not financial advice**. "
            "Code: github.com/hello-keshavkandel/Rentwise-Uk")

# ---------------------------------------------------------------- Rent estimator
if view == "Rent estimator":
    try:
        predictor = get_predictor()
    except Exception as e:
        st.error("Could not load the trained model. Run `python run_real_pipeline.py` first.")
        st.caption(f"Detail: {e}"); st.stop()

    with st.sidebar:
        st.header("Property details")
        area = st.selectbox("Local authority", predictor.area_names())
        ptype = st.selectbox("Property type", predictor.property_types)
        bedrooms = predictor.typical_bedrooms(ptype)   # tied to type in the ONS data
        st.caption(f"Typical size: **{bedrooms} bedrooms** (ONS reports rent by property type).")
        go = st.button("Check rent", type="primary", use_container_width=True)
        st.markdown("<p class='disclaimer'>Earnings, crime, schools and amenities are looked up "
                    "automatically for the selected area.</p>", unsafe_allow_html=True)

    if go:
        r = predictor.predict(area, ptype, bedrooms); pct = r["ratio"]*100
        st.subheader(f"{ptype} · {bedrooms} bed · {area}")
        c1,c2,c3 = st.columns(3)
        as_of = month_label(r.get("as_of"))
        if r["kind"] == "official":
            c1.metric("Official median rent", f"£{r['rent']:,}/month")
            st.caption(f"**Official ONS figure** for {as_of} (Price Index of Private Rents). "
                       f"RentWise's model estimates £{r['model_rent']:,} for the same area and type "
                       "and uses it only to explain what drives the figure.")
        else:
            c1.metric("Estimated monthly rent", f"£{r['rent']:,}")
            iv = r.get("interval")
            st.caption("**Model estimate** — ONS publishes no figure for this property type here. "
                       + (f"{int(iv['level']*100)}% range: **£{iv['low']:,} – £{iv['high']:,}** "
                          "(calibrated by conformal prediction)." if iv else ""))
        c2.metric("Rent-to-income", f"{pct:.0f}%"); c3.metric("Region", r["region"])
        if r["affordable"]:
            st.success(f"✅ Affordable — rent is {pct:.0f}% of local income (≤30% benchmark).")
        else:
            st.error(f"⚠️ Above the 30% benchmark — rent is {pct:.0f}% of local income.")
        st.subheader("What drives rent here?")
        top = r["contributions"][:6]
        labels = [friendly(f) for f,_ in top][::-1]; values = [v for _,v in top][::-1]
        colors = ["#1E7D32" if v>=0 else "#B00020" for v in values]
        fig, ax = plt.subplots(figsize=(7,3.5))
        ax.barh(labels, values, color=colors); ax.axvline(0, color="#333", linewidth=0.8)
        ax.set_xlabel("Impact on the rent estimate (£)"); ax.set_title("Top factors (SHAP)")
        fig.tight_layout(); st.pyplot(fig)
        with st.expander("Read the explanation in words"):
            for f,v in top:
                st.write(f"- **{friendly(f)}** {'increases' if v>=0 else 'decreases'} the estimate by £{abs(v):.0f}.")
            st.caption("These are associations learned from data across areas, not causes.")
        prof = predictor.spatial_profile(area)
        if prof:
            with st.expander("What's nearby (area profile)"):
                rows = [(AMENITY[k[5:]], amenity_per_km2(v)) for k, v in prof.items()
                        if k.startswith("dens_") and k[5:] in AMENITY]
                st.table({"Amenity": [r[0] for r in rows],
                          "Per km² (whole area)": [f"{r[1]:.2f}" for r in rows]})
                sq = prof.get("school_quality")
                st.write("**Schools:** " + (f"average historic Ofsted score {sq:.2f}/4, "
                         f"{prof['share_good_or_better']*100:.0f}% Good or Outstanding"
                         if sq is not None else "no Ofsted data (Wales — schools are inspected by Estyn)"))
                st.caption("Source: OpenStreetMap contributors (ODbL); Ofsted & DfE (historic 2019-framework grades).")
    else:
        st.info("👈 Choose a property on the left, then click **Estimate rent**.")

# --------------------------------------------------------------------- Area map
elif view == "Area map":
    import pandas as pd
    import streamlit.components.v1 as components
    st.subheader("Area map — England & Wales by local authority")
    P = config.DATA_PROCESSED
    geo = P / "lad_boundaries_wgs84.geojson"
    if not geo.exists():
        st.warning("Boundary file missing — run `make geo`."); st.stop()
    options = {"Rent-to-income ratio (affordability)": "ratio",
               "Average historic Ofsted school quality (England)": "school_quality",
               **{f"Density: {v.lower()} per km²": f"dens_{k}" for k, v in AMENITY.items()}}
    choice = st.selectbox("Show", list(options))

    @st.cache_data(show_spinner="Drawing map…")
    def map_html(metric: str) -> str:
        from src.viz.maps import choropleth
        from src.data.store import load_table          # DuckDB store first, CSV fallback
        clean = load_table("rentwise_clean")[["la_code", "la_name"]].drop_duplicates()
        if metric == "ratio":
            a = load_table("affordability").merge(clean, on="la_name")
            vals, label, cmap = dict(zip(a.la_code, a.rent_to_income)), "Rent-to-income ratio", "YlOrRd"
        else:
            sp = pd.read_csv(P / "la_spatial_features.csv")
            v = sp[metric] if metric == "school_quality" else sp[metric].map(amenity_per_km2)
            vals, label = dict(zip(sp.lad_code, v)), choice
            cmap = "YlGnBu"
        vals = {k: x for k, x in vals.items() if pd.notna(x)}
        return choropleth(vals, geo, value_name=label, fill_color=cmap)._repr_html_()

    components.html(map_html(options[choice]), height=620)
    st.caption("Grey = no data (e.g. Ofsted does not inspect Welsh schools). Boundaries: ONS (May 2024), "
               "contains OS data © Crown copyright. Amenities: © OpenStreetMap contributors.")

# --------------------------------------------------------------- System results
else:
    st.subheader("Full system — all four modules")
    st.caption("Read-only overview of each module's latest results (run `python run_real_pipeline.py` to refresh).")
    reports = config.ROOT / "reports"; figs = reports / "figures"
    modules = [
        ("Module 1 — Sale Price Prediction", "module1_summary.md", "m1_predicted_vs_actual.png"),
        ("Module 2 — Rental Price & Affordability (deployed model)",
         "rent_model_summary.md", "rent_shap_importance.png"),
        ("Module 3 — Geospatial Driver Analysis", "module3_summary.md", "m3_ablation.png"),
        ("Module 4 — Regional Trend Forecasting", "module4_summary.md", "m4_forecast_london.png"),
    ]
    for title, md_name, fig_name in modules:
        st.markdown(f"### {title}")
        col1, col2 = st.columns([3, 2])
        md_path = reports / md_name
        with col1:
            st.markdown(md_path.read_text() if md_path.exists() else "_Run the pipeline to generate results._")
        with col2:
            fig_path = figs / fig_name
            if fig_path.exists():
                st.image(str(fig_path), use_column_width=True)
        st.divider()

st.markdown("<p class='disclaimer'>Estimates are indicative, produced by machine-learning "
            "models on real open data (ONS, HM Land Registry, data.police.uk), and are "
            "<b>not financial advice</b>.</p>", unsafe_allow_html=True)
