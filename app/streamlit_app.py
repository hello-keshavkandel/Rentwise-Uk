"""
streamlit_app.py  —  the RentWise UK dashboard.

Views (sidebar):
  * Rent check        official rent for any area and home size/type, affordability against YOUR
                      income, sale prices & yield, the trend since 2015, the regional forecast,
                      and what drives rent here (SHAP)
  * Compare areas     2-4 areas side by side, with trends and a CSV download
  * Area map          England & Wales choropleths (affordability, rent change, yield, schools, amenities)
  * Methodology & data  what every number is, its source, date and limits
  * System results    the four research modules' latest results

Design: official figures first, every number dated and sourced, missing shown as missing,
inputs are dropdowns and one optional number (no free text), income is never stored.

Run it:  streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from src.models.area_data import MEASURE_LABEL, TYPE_MEASURES, AreaData, affordability
from src.models.predict import RentWisePredictor

st.set_page_config(page_title="RentWise UK", page_icon="🏠", layout="wide")
st.markdown("""<style>
 .main .block-container {padding-top: 2rem; max-width: 1150px;}
 h1 {color:#1F3864;}
 [data-testid="stMetric"]{background:#F4F8FD;border:1px solid #D5DEEA;border-radius:12px;padding:14px;}
 .disclaimer{color:#4b5563;font-size:0.85rem;}
 a:focus, button:focus {outline: 3px solid #1F3864 !important; outline-offset: 2px;}
</style>""", unsafe_allow_html=True)

FRIENDLY = {"bedrooms": "Number of bedrooms", "median_annual_income": "Local income",
            "crime_per_1000": "Crime per 1,000 residents",
            "property_type_ord": "Property type / size", "income_band": "Area income band", "year": "Year",
            "school_quality": "Local school quality (historic Ofsted)",
            "share_good_or_better": "Share of Good/Outstanding schools"}
AMENITY = {"rail_station": "Rail stations", "bus_stop": "Bus stops", "supermarket": "Supermarkets",
           "convenience": "Convenience shops", "gp_clinic": "GP surgeries & clinics", "hospital": "Hospitals",
           "pharmacy": "Pharmacies", "restaurant_cafe": "Restaurants & cafés", "pub_bar": "Pubs & bars",
           "park": "Parks", "gym": "Gyms"}
BLUE, GREY, ORANGE = "#1F3864", "#6b7280", "#B45309"


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


def gbp(v):
    return "—" if v is None or v != v else f"£{v:,.0f}"


def _version(*rel_paths) -> str:
    """Cache key from file modification times. Streamlit Cloud can keep a cached object alive across a
    code push; keying on the code + data files rebuilds it whenever either changes (a stale cached
    AreaData without new methods once raised AttributeError on the live app)."""
    import os
    return "|".join(f"{p}:{os.path.getmtime(config.ROOT / p):.0f}" for p in rel_paths
                    if (config.ROOT / p).exists())


@st.cache_resource
def _predictor(version: str):
    return RentWisePredictor()


@st.cache_resource
def _area_data(version: str):
    return AreaData()


def get_predictor():
    return _predictor(_version("src/models/predict.py", "models/rent_model_meta.json",
                               "data/processed/earnings.csv", "data/processed/rentwise_clean.csv"))


def get_area_data():
    return _area_data(_version("src/models/area_data.py", "data/processed/official_rents.csv",
                               "data/processed/rent_quartiles.csv", "data/processed/earnings.csv",
                               "data/processed/sale_prices.csv", "models/rent_forecasts.json"))


def line_chart(df, title, y_title="£ per month", bands=None):
    """Accessible line chart: titled, labelled axes, text alternative offered by the caller."""
    import plotly.graph_objects as go
    fig = go.Figure()
    if bands is not None:
        for lo, hi, name, alpha in bands:
            fig.add_trace(go.Scatter(x=list(df.index) + list(df.index[::-1]),
                                     y=list(df[hi]) + list(df[lo][::-1]), fill="toself", name=name,
                                     line=dict(width=0), fillcolor=f"rgba(31,56,100,{alpha})", hoverinfo="skip"))
    # colour-blind-safe palette, and every line also differs by dash pattern (never colour alone)
    colors = [BLUE, ORANGE, "#0F766E", GREY]
    dashes = ["solid", "dash", "dot", "dashdot"]
    for i, c in enumerate([c for c in df.columns if bands is None or c not in sum(([b[0], b[1]] for b in bands), [])]):
        fig.add_trace(go.Scatter(x=df.index, y=df[c], name=c, mode="lines",
                                 line=dict(width=3 if i < 2 else 2, dash=dashes[i % 4], color=colors[i % 4])))
    fig.update_layout(title=title, yaxis_title=y_title, xaxis_title=None, height=360, margin=dict(t=50, b=10),
                      legend=dict(orientation="h", y=-0.15), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


st.title("🏠 RentWise UK")
st.caption("Official UK rents, affordability and trends for every local authority in England & Wales — "
           "explained, dated and sourced")

VIEWS = ["Rent check", "Compare areas", "Area map", "Methodology & data", "System results"]
view = st.sidebar.radio("View", VIEWS)

try:
    data = get_area_data()
except Exception as e:  # noqa: BLE001
    st.error("Data files are missing. Run `python run_real_pipeline.py` (or `make build`) first.")
    st.caption(f"Detail: {e}")
    st.stop()
V = data.vintages
AS_OF = month_label(data.period)

# ---- observability + sources (every view) ----
with st.sidebar:
    try:
        from src.health import check
        _h = check()
        _icon = {"ok": "🟢", "degraded": "🟡", "down": "🔴"}.get(_h["status"], "⚪")
        _d = _h["data"].get("age_days"); _m = _h["model"].get("age_days")
        st.caption(f"{_icon} System {_h['status']}"
                   + (f" · data {int(_d)}d old" if _d is not None else "")
                   + (f" · model {int(_m)}d old" if _m is not None else ""))
    except Exception:
        pass
    with st.expander("Data sources & dates"):
        _r, _e, _p, _c, _s = (V.get(k, {}) or {} for k in ("rents", "earnings", "population", "crime", "sale_prices"))
        st.markdown(
            f"- **Rents:** {_r.get('source')} — {month_label(_r.get('period'))}"
            f"{' (released ' + _r['released'] + ')' if _r.get('released') else ''}; history from {_r.get('history_from')}\n"
            f"- **Earnings:** ONS ASHE Table 8.7a, median gross annual pay of full-time employees by place of "
            f"residence — {_e.get('period')}\n"
            f"- **Rent ranges (median & quartiles):** ONS — "
            f"{month_label((V.get('rent_quartiles') or {}).get('period'))}\n"
            f"- **Sale prices:** {_s.get('source', 'HM Land Registry')} — {_s.get('period')}\n"
            f"- **Population:** {_p.get('source')} — {_p.get('period')}\n"
            f"- **Crime:** {_c.get('measure')} — {_c.get('period')} ({_c.get('months')} months)\n"
            "- **Amenities:** OpenStreetMap via Geofabrik — Nov 2026 · **Schools:** Ofsted & DfE — Oct 2026\n"
            "- **Boundaries:** ONS, May 2024")
        if _c.get("areas_without_police_data"):
            st.caption("No police data for: " + ", ".join(_c["areas_without_police_data"])
                       + " — shown as missing, never as zero.")


# ================================================================ Rent check
if view == "Rent check":
    areas = sorted(data.areas.la_name)
    with st.sidebar:
        st.header("Your search")
        area = st.selectbox("Local authority", areas, index=areas.index("Leeds") if "Leeds" in areas else 0)
        measures = data.measures()
        measure = st.selectbox("Home size or type", measures, format_func=lambda m: MEASURE_LABEL.get(m, m),
                               help="ONS publishes official rents by bedroom count and by property type.")
        income = st.number_input("Your gross annual income, £ (optional)", min_value=0, max_value=2_000_000,
                                 value=0, step=1000,
                                 help="Before tax, for everyone paying the rent. Used only on this page; "
                                      "never stored. Leave at 0 to compare with local full-time pay.")
    code = data.code_for(area)
    a = data.areas.loc[code]
    offs = data.official_rents(code)
    off = offs.get(measure)
    st.subheader(f"{MEASURE_LABEL.get(measure, measure)} · {area}")
    st.caption(f"{a.region} · local authority {code} · change the area, home or income in the sidebar "
               "(tap › on a phone)")

    pred = None
    if measure in TYPE_MEASURES:
        try:
            pred = get_predictor().predict(area, measure)
        except Exception as e:  # noqa: BLE001
            st.caption(f"Model explanation unavailable: {e}")
    if off:
        rent, kind = off["rent"], "official"
    elif pred:
        rent, kind = pred["rent"], "modelled"
    else:
        rent, kind = None, None

    c1, c2, c3 = st.columns(3)
    if kind == "official":
        ch = off.get("annual_change_pct")
        c1.metric("Official average rent", f"£{rent:,}",
                  help="ONS average monthly private rent (PIPR price level: a geometric-mean measure of rents "
                       "paid by tenants in new and existing lets). Not an asking rent.",
                  delta=f"{ch:+.1f}% in a year" if ch is not None else None, delta_color="off")
        st.caption(f"**Official ONS figure** for {AS_OF} — Price Index of Private Rents. "
                   + (f"RentWise's model estimates £{pred['model_rent']:,} for the same area and type and uses "
                      "it only to explain what drives the figure." if pred else ""))
    elif kind == "modelled":
        c1.metric("Estimated monthly rent", f"£{rent:,}")
        iv = pred.get("interval")
        st.caption("**Model estimate** — ONS publishes no figure for this home type here. "
                   + (f"{int(iv['level'] * 100)}% range: **£{iv['low']:,} – £{iv['high']:,}**." if iv else ""))
    else:
        c1.metric("Official average rent", "No figure")
        st.info("ONS publishes no official rent for this home size here, and RentWise does not estimate "
                "bedroom-specific rents.")

    local = affordability(rent, data.area_income(code))
    mine = affordability(rent, income) if income else None
    if local:
        c2.metric("Share of local pay", f"{local['ratio'] * 100:.0f}%",
                  help=f"Annual rent ÷ median gross annual pay of full-time employees living here "
                       f"(£{data.area_income(code):,.0f}, ASHE {V.get('earnings', {}).get('period')}) — "
                       "the earnings basis ONS uses for affordability.")
    if mine:
        c3.metric("Share of your income", f"{mine['ratio'] * 100:.0f}%",
                  help="Annual rent ÷ the gross annual income you entered.")
    elif local:
        c3.metric("Affordable rent (30%)", f"£{local['max_rent_at_30pct']:,}",
                  help="The monthly rent a median earner here could pay within the 30% guideline.")

    if mine:
        msg = (f"At £{income:,.0f} a year this rent is **{mine['ratio'] * 100:.0f}%** of your gross income — "
               f"**{mine['band']}**. Staying at or under 30% means a rent of up to **£{mine['max_rent_at_30pct']:,}"
               "/month**.")
        (st.success if mine["affordable"] else st.warning)(("✅ " if mine["affordable"] else "⚠️ ") + msg)
    elif local:
        msg = (f"For one full-time worker on the local median pay this rent is **{local['ratio'] * 100:.0f}%** of "
               "gross pay — "
               f"**{local['band']}**. Enter your own income on the left for a personal answer.")
        (st.success if local["affordable"] else st.warning)(("✅ " if local["affordable"] else "⚠️ ") + msg)

    # ---- what most people pay: ONS median and middle-half range (snapshot) ----
    rr = data.rent_range(code, measure) if measure not in TYPE_MEASURES else None
    if rr:
        homes = {"All": "homes", "1 bed": "1-bedroom homes", "2 bed": "2-bedroom homes",
                 "3 bed": "3-bedroom homes", "4+ bed": "homes with 4 or more bedrooms"}.get(measure, "homes")
        if rr["lower_quartile"] and rr["upper_quartile"]:
            st.markdown(f"**What most people pay:** half of rented {homes} here cost between "
                        f"**£{rr['lower_quartile']:,} and £{rr['upper_quartile']:,}** a month; the median is "
                        f"**£{rr['median']:,}**" + (f" ({rr['count']:,} rents" if rr["count"] else " (")
                        + f", {month_label(rr['period'])}).")
        else:
            st.markdown(f"**What most people pay:** median **£{rr['median']:,}** a month "
                        f"({month_label(rr['period'])}; too few rents for a reliable range).")
        st.caption("ONS median and quartiles — a one-month snapshot of rents actually paid. The median is the "
                   "middle rent; the official average above is ONS's mix-adjusted measure, so the two can differ.")
    elif measure in TYPE_MEASURES:
        st.caption("ONS publishes rent ranges by bedroom count, not property type — choose a bedroom size to see "
                   "what most people pay.")

    # ---- every official rent for this area ----
    with st.expander(f"All official rents in {area} ({AS_OF})", expanded=False):
        rows = []
        for m, r in sorted(offs.items(), key=lambda kv: data.measures().index(kv[0])):
            q = data.rent_range(code, m) if m not in TYPE_MEASURES else None
            rows.append({"Home": MEASURE_LABEL.get(m, m), "Average rent (£/month)": r["rent"],
                         "Change on a year (%)": r["annual_change_pct"],
                         "Median rent (£/month)": q["median"] if q else None,
                         "Middle half (£/month)": (f"{q['lower_quartile']:,}–{q['upper_quartile']:,}"
                                                   if q and q["lower_quartile"] and q["upper_quartile"] else None),
                         "Rent / local full-time pay (%)": round(r["rent"] * 12 / data.area_income(code) * 100, 1)
                         if data.area_income(code) else None})
        studio = data.rent_range(code, "Studio")
        if studio:
            rows.insert(1, {"Home": "Studio", "Average rent (£/month)": None, "Change on a year (%)": None,
                            "Median rent (£/month)": studio["median"],
                            "Middle half (£/month)": (f"{studio['lower_quartile']:,}–{studio['upper_quartile']:,}"
                                                      if studio["lower_quartile"] and studio["upper_quartile"]
                                                      else None),
                            "Rent / local full-time pay (%)": None})
        tbl = pd.DataFrame(rows)
        st.caption("Average: ONS PIPR, " + AS_OF + ". Median and middle half: ONS median & quartiles, "
                   + month_label((V.get("rent_quartiles") or {}).get("period")) + " (by bedrooms only). "
                   "Blank = not published.")
        shown = tbl.map(lambda v: "—" if v is None or (isinstance(v, float) and v != v)
                        else (f"{v:,}" if isinstance(v, int) else str(v)))
        st.dataframe(shown, hide_index=True, use_container_width=True)
        st.download_button("Download this table (CSV)", tbl.to_csv(index=False).encode(),
                           file_name=f"rentwise_{code}_{data.period}.csv", mime="text/csv")

    # ---- buy vs rent ----
    sp_type = measure if measure in TYPE_MEASURES else "All"
    sp = data.sale_price(code, sp_type)
    if sp and sp.get("median_price"):
        y = data.gross_yield(code, measure)
        st.markdown(f"**Buy vs rent:** {'homes' if sp_type == 'All' else MEASURE_LABEL[sp_type].lower() + 's'} "
                    f"here sold for a median **{gbp(sp['median_price'])}** ({sp['sales']:,} sales, {sp['period']})"
                    + (f" — an indicative gross yield of **{y:.1f}%** a year." if y else "."))
        st.caption("HM Land Registry Price Paid, standard-price sales. Yield = annual ONS average rent ÷ median sale "
                   "price; before costs, voids and tax; rented and sold homes are not the same homes.")

    # ---- trend since 2015 ----
    st.subheader("How rent has moved since 2015")
    reg_code = data.region_codes.get(a.region) or data.nation_codes.get(a.region)
    series = {area: data.trend(code, measure)}
    if reg_code:
        series[a.region] = data.trend(reg_code, measure)
    eng = data.nation_codes.get("England" if code.startswith("E") else "Wales")
    if eng and eng != reg_code:
        series["England" if code.startswith("E") else "Wales"] = data.trend(eng, measure)
    tdf = pd.concat({k: v.set_index("period")["rent"] for k, v in series.items() if not v.empty}, axis=1)
    if not tdf.empty:
        tdf.index = pd.to_datetime(tdf.index)
        line_chart(tdf, f"Official average rent — {MEASURE_LABEL.get(measure, measure).lower()}")
        first, last = tdf[area].dropna().iloc[0], tdf[area].dropna().iloc[-1]
        st.caption(f"{area}: £{first:,.0f} in {tdf[area].dropna().index[0]:%b %Y} → £{last:,.0f} in {AS_OF} "
                   f"({(last / first - 1) * 100:+.0f}%). Source: ONS PIPR.")
        with st.expander("Show the trend as a table"):
            st.dataframe(tdf.iloc[::-6].sort_index().rename_axis("Month").style.format("£{:,.0f}"),
                         use_container_width=True)

    # ---- regional forecast ----
    fc = data.region_forecast(code)
    if fc:
        st.subheader(f"Where rents in {fc['name']} may go next")
        hist = data.trend(fc["area_code"], "All").tail(36).set_index("period")["rent"]
        f = pd.DataFrame(fc["forecast"]).set_index("period")
        fdf = pd.DataFrame({"Official": hist}).join(f[["rent", "low80", "high80", "low95", "high95"]], how="outer")
        fdf.loc[hist.index[-1], ["rent", "low80", "high80", "low95", "high95"]] = hist.iloc[-1]
        fdf = fdf.rename(columns={"rent": "Forecast"})
        fdf.index = pd.to_datetime(fdf.index)
        line_chart(fdf, f"{fc['name']}, all properties: 12-month forecast",
                   bands=[("low95", "high95", "95% band", 0.10), ("low80", "high80", "80% band", 0.22)])
        last = fc["forecast"][-1]; m = fc["method"]
        st.caption(f"Forecast for {month_label(last['period'])}: **£{last['rent']:,}** (80% band £{last['low80']:,}–"
                   f"£{last['high80']:,}). Method: {m['chosen']} on the official series, chosen by rolling-origin "
                   f"backtest (MAPE {m['backtest_mape_pct'][m['chosen']]:.2f}%). The 80% band covered "
                   f"{m['out_of_time_coverage']['80%'] * 100:.0f}% of outcomes in recent out-of-time tests — treat it "
                   "as a guide. Regional, not area-specific; not a prediction of policy effects.")

    # ---- what drives rent (model explanation) ----
    if pred:
        st.subheader("What drives rent here?")
        top = pred["contributions"][:6]
        labels = [friendly(f) for f, _ in top][::-1]; values = [v for _, v in top][::-1]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.barh(labels, values, color=["#1E7D32" if v >= 0 else "#B00020" for v in values])
        for i, v in enumerate(values):
            ax.text(v, i, f" {'+' if v >= 0 else '−'}£{abs(v):.0f} ", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8)
        ax.axvline(0, color="#333", linewidth=0.8)
        ax.set_xlabel("Impact on the model's estimate (£/month)"); ax.set_title("Top factors (SHAP)")
        fig.tight_layout(); st.pyplot(fig)
        with st.expander("Read the explanation in words"):
            for f, v in top:
                st.write(f"- **{friendly(f)}** {'increases' if v >= 0 else 'decreases'} the estimate by £{abs(v):.0f}.")
            st.caption("Associations learned across areas, not causes. The model works by property type, "
                       "so explanations are shown for flats and houses, not bedroom counts.")
    elif measure not in TYPE_MEASURES:
        st.caption("Choose a property type (flat, terraced, semi-detached, detached) to see what drives rent here.")

    prof = get_predictor().spatial_profile(area) if code in data.areas.index else {}
    if prof:
        with st.expander("What's nearby (area profile)"):
            rows = [(AMENITY[k[5:]], amenity_per_km2(v)) for k, v in prof.items()
                    if k.startswith("dens_") and k[5:] in AMENITY]
            st.dataframe(pd.DataFrame({"Amenity": [r[0] for r in rows],
                                       "Per km² (whole area)": [round(r[1], 2) for r in rows]}),
                         hide_index=True, use_container_width=True)
            sq = prof.get("school_quality")
            crime = a.crime
            st.write("**Schools:** " + (f"average historic Ofsted score {sq:.2f}/4, "
                     f"{prof['share_good_or_better'] * 100:.0f}% Good or Outstanding"
                     if sq is not None else "no Ofsted data (Wales — schools are inspected by Estyn)"))
            st.write("**Crime:** " + (f"{crime:.0f} recorded street crimes per 1,000 residents a year"
                     if crime == crime else "no police data published for this area"))
            st.caption("OpenStreetMap contributors (ODbL); Ofsted & DfE; data.police.uk; ONS population.")

# ================================================================ Compare areas
elif view == "Compare areas":
    st.subheader("Compare areas side by side")
    areas = sorted(data.areas.la_name)
    default = [x for x in ("Manchester", "Leeds", "Bristol, City of") if x in areas]
    picked = st.multiselect("Areas (2 to 4)", areas, default=default, max_selections=4)
    cc1, cc2 = st.columns(2)
    measure = cc1.selectbox("Home size or type", data.measures(), format_func=lambda m: MEASURE_LABEL.get(m, m))
    income = cc2.number_input("Your gross annual income, £ (optional)", min_value=0, max_value=2_000_000,
                              value=0, step=1000, help="Never stored.")
    if len(picked) < 2:
        st.info("Choose at least two areas.")
    else:
        codes = [data.code_for(p) for p in picked]
        df = data.compare(codes, measure, income or None)
        def fmt(v):
            if v is None or (isinstance(v, float) and v != v):
                return "—"
            if isinstance(v, float):
                return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}".rstrip("0").rstrip(".")
            return f"{v:,}" if isinstance(v, int) else str(v)
        show = df.drop(columns=["Code"]).set_index("Area").T.map(fmt)
        st.dataframe(show, use_container_width=True, height=(len(show) + 1) * 35 + 3)
        st.caption(f"Rents: ONS PIPR, {AS_OF}. Earnings: ASHE {V.get('earnings', {}).get('period')}. "
                   f"Sale prices: HM Land Registry {V.get('sale_prices', {}).get('period')}. "
                   "Amenities per km² of the whole local authority (OpenStreetMap). “—” = no data, never zero.")
        st.download_button("Download comparison (CSV)", df.to_csv(index=False).encode(),
                           file_name=f"rentwise_compare_{data.period}.csv", mime="text/csv")
        tdf = pd.concat({p: data.trend(c, measure).set_index("period")["rent"] for p, c in zip(picked, codes)}, axis=1)
        if not tdf.empty:
            tdf.index = pd.to_datetime(tdf.index)
            line_chart(tdf, f"Official average rent since 2015 — {MEASURE_LABEL.get(measure, measure).lower()}")
    st.divider()
    st.markdown("**For analysts:** every area × every official measure, with local full-time pay and ratio.")
    st.download_button("Download all areas (CSV)", data.all_areas_table().to_csv(index=False).encode(),
                       file_name=f"rentwise_all_areas_{data.period}.csv", mime="text/csv")

# ================================================================ Area map
elif view == "Area map":
    import streamlit.components.v1 as components
    st.subheader("Area map — England & Wales by local authority")
    P = config.DATA_PROCESSED
    geo = P / "lad_boundaries_wgs84.geojson"
    if not geo.exists():
        st.warning("Boundary file missing — run `make geo`."); st.stop()
    options = {"Rent ÷ local full-time pay, all properties (official)": "ratio",
               "Official average rent, all properties (£/month)": "rent",
               "Rent change on a year (%)": "change",
               "Indicative gross rental yield (%)": "yield",
               "Average historic Ofsted school quality (England)": "school_quality",
               **{f"Density: {v.lower()} per km²": f"dens_{k}" for k, v in AMENITY.items()}}
    choice = st.selectbox("Show", list(options))

    @st.cache_data(show_spinner="Drawing map…")
    def map_html(metric: str, label: str, version: str = "") -> str:   # version: rebuild when data change
        from src.viz.maps import choropleth
        d = get_area_data()
        allr = d.official[d.official.measure == "All"].set_index("area_code")
        if metric == "ratio":
            vals, cmap = d.official_affordability(), "YlOrRd"
        elif metric == "rent":
            vals, cmap = allr["rent"].to_dict(), "YlOrRd"
        elif metric == "change":
            vals, cmap = allr["annual_change_pct"].to_dict(), "RdYlBu_r"
        elif metric == "yield":
            vals, cmap = {c: d.gross_yield(c) for c in allr.index}, "YlGn"
        else:
            sp = pd.read_csv(P / "la_spatial_features.csv")
            v = sp[metric] if metric == "school_quality" else sp[metric].map(amenity_per_km2)
            vals, cmap = dict(zip(sp.lad_code, v)), "YlGnBu"
        vals = {k: x for k, x in vals.items() if x is not None and pd.notna(x)}
        return choropleth(vals, geo, value_name=label, fill_color=cmap)._repr_html_()

    components.html(map_html(options[choice], choice,
                             _version("src/models/area_data.py", "data/processed/official_rents.csv",
                                      "data/processed/earnings.csv", "data/processed/sale_prices.csv",
                                      "data/processed/la_spatial_features.csv")), height=620)
    st.caption(f"Rents: ONS PIPR {AS_OF}; earnings: ASHE; prices: HM Land Registry. Grey = no data (e.g. Ofsted does "
               "not inspect Welsh schools). Boundaries: ONS (May 2024), contains OS data © Crown copyright. "
               "Amenities: © OpenStreetMap contributors.")

# ================================================================ Methodology
elif view == "Methodology & data":
    st.subheader("Methodology & data — what every number is")
    meta = {}
    try:
        meta = json.loads((config.MODELS_DIR / "rent_model_meta.json").read_text())
    except Exception:
        pass
    ev = meta.get("evaluation", {})
    rcv, gcv = ev.get("random_cv", {}) or {}, ev.get("grouped_cv", {}) or {}
    fm = (data.forecasts or {}).get("method", {})
    _r, _e, _s, _c = (V.get(k, {}) or {} for k in ("rents", "earnings", "sale_prices", "crime"))
    st.markdown(f"""
**What RentWise is.** A free tool that shows the **official** typical rent for every local authority in
England and Wales, how affordable it is, how it has moved, where it may go next, and *why* rents differ
between places. It covers areas, not individual homes: no open dataset records the rent of a specific flat.

#### The numbers
| Figure | What it is | Source & date |
|---|---|---|
| **Official average rent** | ONS's average monthly private rent for the area (the PIPR price level — a geometric-mean measure of rents actually paid in new *and* existing tenancies, adjusted for the mix of homes), by bedroom count (1–4+) or property type. It is not an asking rent and not a median | {_r.get('source')}, {month_label(_r.get('period'))} (released {_r.get('released')}) |
| **Change on a year** | % change in that average over 12 months, as published | ONS PIPR |
| **Median and middle half** | The middle rent, and the range covering the middle 50% of rents (lower to upper quartile), by bedroom count. A one-month snapshot: ONS advises it is for comparing places at one time, not for trends. Values based on 10 or fewer rents are suppressed by ONS and shown blank | ONS Private rent price median and quartiles, {month_label((V.get('rent_quartiles') or {}).get('period'))} (released {(V.get('rent_quartiles') or {}).get('released')}) |
| **Rent ÷ local pay** | Annual rent ÷ **median gross annual pay of full-time employees living in the area** — one earner; the same earnings basis ONS uses for its housing affordability ratios. Households with two earners will find rent more affordable than this | ASHE Table 8.7a (full-time, residence), {_e.get('period')} |
| **Rent ÷ your income** | Annual rent ÷ the gross annual income *you* enter (use household income if rent is shared). Never stored | — |
| **30% guideline** | Rent at or below 30% of gross income is widely used as the affordability line; above 40% is shown as “well above” | convention |
| **Median sale price** | Median price paid, standard-price sales (Land Registry category A), shown only with ≥ 20 sales | HM Land Registry, {_s.get('period')} |
| **Gross yield** | Annual ONS average rent ÷ median sale price for the same property type. Indicative: before costs, voids, tax; rented and sold homes differ | derived |
| **Crime** | Recorded street crimes per 1,000 residents, annualised over {_c.get('months')} months. Missing where a force published nothing — never zero | data.police.uk + ONS population |
| **Schools / amenities** | Historic Ofsted grades (England only); OpenStreetMap amenity counts per km² | Ofsted & DfE; © OSM contributors |

**How this compares with ONS's own affordability figure.** ONS's *Private rental affordability* bulletin divides
rent by the **household** income of private renters (FYE 2024: England 36.3%, London 41.6%, North East 19.8%). That
is lower than RentWise's one-earner figure because many renting households have more than one income. Use
"Rent ÷ your income" with your household's income for your own answer.

**Checked against the official releases.** RentWise's figures reproduce the ONS November 2026 release exactly
(UK £1,411, +3.8%; England £1,472; Wales £849; London £2,350; Kensington and Chelsea £3,765). See the
validation report in the project repository.

#### The model (explains rents; estimates only where ONS publishes nothing)
XGBoost on local earnings (all employee jobs), crime, region, property type, amenities and schools, trained on {meta.get('n_las', '316')} areas.
Accuracy: R² **{rcv.get('r2', 0):.2f}** / MAE **£{rcv.get('mae', 0):.0f}** for areas seen in training; R² **{gcv.get('r2', 0):.2f}** /
MAE **£{gcv.get('mae', 0):.0f}** when whole areas are held out (the honest test). Explanations use SHAP; they show
association, not cause. Only trained areas are served, because the 80% conformal range is calibrated there.

#### Forecasts
Regional (9 English regions + Wales), all properties, 12 months ahead. Three models compared by rolling-origin
backtest; **{fm.get('chosen', '—')}** chosen (MAPE {(fm.get('backtest_mape_pct') or {}).get(fm.get('chosen'), 0):.2f}%).
Bands are conformal (from the backtest's own errors). In out-of-time tests the 80% band covered
**{(fm.get('out_of_time_coverage') or {}).get('80%', 0) * 100:.0f}%** of outcomes and the 95% band
**{(fm.get('out_of_time_coverage') or {}).get('95%', 0) * 100:.0f}%** — slightly narrow, because the pace of rent growth has shifted more in recent years than before.
Forecasts extrapolate the official series and do not model policy changes such as the Renters' Rights Act.

#### Limits
- Areas, not individual homes. The median and middle-half range show the spread of rents, but a specific flat
  depends on its size, condition and street.
- Scotland and Northern Ireland are not covered yet.
- Rents are ONS's measure of rents paid by tenants (new and existing lets), which moves more slowly than advertised asking rents.
- Indicative information, **not financial advice**.

#### Licences
Contains public sector information licensed under the **Open Government Licence v3.0** (ONS, HM Land Registry,
Ofsted/DfE, police.uk). Amenity counts derived from **© OpenStreetMap contributors (ODbL)**.
Code: github.com/hello-keshavkandel/Rentwise-Uk · Built on an MSc dissertation (Group 27, UWS); product
development by Keshav Kandel.
""")
    st.download_button("Download all official rents (CSV)", data.all_areas_table().to_csv(index=False).encode(),
                       file_name=f"rentwise_all_areas_{data.period}.csv", mime="text/csv")

# ================================================================ System results
else:
    st.subheader("Full system — research modules")
    st.caption("Read-only overview of each module's latest results (run `python run_real_pipeline.py` to refresh).")
    reports = config.ROOT / "reports"; figs = reports / "figures"
    modules = [
        ("Module 1 — Sale Price Prediction", "module1_summary.md", "m1_predicted_vs_actual.png"),
        ("Module 2 — Rental Price & Affordability (deployed model)", "rent_model_summary.md",
         "rent_shap_importance.png"),
        ("Module 3 — Geospatial Driver Analysis", "module3_summary.md", "m3_ablation.png"),
        ("Module 4 — Regional Trend Forecasting (research)", "module4_summary.md", "m4_forecast_london.png"),
        ("Rent forecasts in the app (Phase C)", "rent_forecast_summary.md", None),
    ]
    for title, md_name, fig_name in modules:
        st.markdown(f"### {title}")
        col1, col2 = st.columns([3, 2])
        md_path = reports / md_name
        with col1:
            st.markdown(md_path.read_text() if md_path.exists() else "_Run the pipeline to generate results._")
        with col2:
            if fig_name and (figs / fig_name).exists():
                st.image(str(figs / fig_name), use_column_width=True)
        st.divider()

st.markdown("<p class='disclaimer'>Official figures from ONS, HM Land Registry and data.police.uk; model "
            "explanations and forecasts are indicative and <b>not financial advice</b>.</p>",
            unsafe_allow_html=True)
