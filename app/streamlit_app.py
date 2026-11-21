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
 "crime_per_1000":"Local crime level","transport_score":"Transport access",
 "property_type_ord":"Property type / size","income_band":"Area income band","year":"Year"}
def friendly(f):
    return "Being in "+f.replace("region_","") if f.startswith("region_") else FRIENDLY.get(f,f)

@st.cache_resource
def get_predictor():
    return RentWisePredictor()

st.title("🏠 RentWise UK")
st.caption("Data-driven UK housing & long-term rental intelligence — a four-module group system")

view = st.sidebar.radio("View", ["Rent estimator", "System results"])

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
    with st.expander("About & methodology"):
        st.markdown(
            "**RentWise UK** predicts UK rents and sale prices from **open government "
            "data** (ONS, HM Land Registry, data.police.uk) and explains every estimate.\n\n"
            "- **Rent (Module 2):** XGBoost, R² 0.90; SHAP explanations; affordability vs the "
            "30% income benchmark; **calibrated 80% prediction intervals** (conformal).\n"
            "- **Sale price (Module 1):** HM Land Registry + optional EPC floor area.\n"
            "- **Geospatial (Module 3):** crime driver; OSM/Ofsted layers + maps in progress.\n"
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
        go = st.button("Estimate rent", type="primary", use_container_width=True)
        st.markdown("<p class='disclaimer'>Income and crime are looked up "
                    "automatically for the selected area.</p>", unsafe_allow_html=True)

    if go:
        r = predictor.predict(area, ptype, bedrooms); pct = r["ratio"]*100
        st.subheader(f"{ptype} · {bedrooms} bed · {area}")
        c1,c2,c3 = st.columns(3)
        c1.metric("Predicted monthly rent", f"£{r['rent']:,}")
        c2.metric("Rent-to-income", f"{pct:.0f}%"); c3.metric("Region", r["region"])
        iv = r.get("interval")
        if iv:
            st.caption(f"{int(iv['level']*100)}% confidence range: "
                       f"**£{iv['low']:,} – £{iv['high']:,}** (calibrated by conformal prediction).")
        if r["affordable"]:
            st.success(f"✅ Affordable — rent is {pct:.0f}% of local income (≤30% benchmark).")
        else:
            st.error(f"⚠️ Above the 30% benchmark — rent is {pct:.0f}% of local income.")
        st.subheader("Why this estimate?")
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
    else:
        st.info("👈 Choose a property on the left, then click **Estimate rent**.")

# --------------------------------------------------------------- System results
else:
    st.subheader("Full system — all four modules")
    st.caption("Read-only overview of each module's latest results (run `python run_real_pipeline.py` to refresh).")
    reports = config.ROOT / "reports"; figs = reports / "figures"
    modules = [
        ("Module 1 — Sale Price Prediction", "module1_summary.md", "m1_predicted_vs_actual.png"),
        ("Module 2 — Rental Price & Affordability", "model_comparison.md", "07_shap_importance.png"),
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
