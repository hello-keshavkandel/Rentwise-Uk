"""
maps.py  —  interactive choropleth maps for RentWise UK (Module 3 / dashboard).

Renders a per-local-authority value (predicted rent, affordability ratio, a spatial
index, …) onto a UK local-authority choropleth using folium. GeoJSON LA boundaries
are read directly (no geopandas needed), so this stays a light dependency.

    from src.viz.maps import choropleth
    m = choropleth({"E07000223": 0.44, "E08000035": 0.31},
                   geojson_path="data/processed/lad_boundaries_wgs84.geojson",
                   value_name="Rent-to-income", out_html="reports/maps/affordability.html")
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _as_frame(values, key_field: str, value_name: str) -> pd.DataFrame:
    if isinstance(values, dict):
        return pd.DataFrame({key_field: list(values.keys()),
                             value_name: list(values.values())})
    if isinstance(values, pd.Series):
        return values.rename(value_name).rename_axis(key_field).reset_index()
    df = values.copy()
    cols = list(df.columns)
    if key_field not in cols or value_name not in cols:
        df.columns = [key_field, value_name][:len(cols)]
    return df


def choropleth(values, geojson_path, key_field: str = "LAD24CD",
               value_name: str = "value", out_html: str | Path | None = None,
               center=(53.0, -1.5), zoom: int = 6, fill_color: str = "YlGnBu"):
    """Build a folium choropleth of `values` keyed by LA code; optionally save HTML."""
    import folium

    df = _as_frame(values, key_field, value_name)
    geojson_path = Path(geojson_path)
    geo = json.loads(geojson_path.read_text())

    m = folium.Map(location=list(center), zoom_start=zoom, tiles="OpenStreetMap")
    folium.Choropleth(
        geo_data=geo,
        data=df,
        columns=[key_field, value_name],
        key_on=f"feature.properties.{key_field}",
        fill_color=fill_color, fill_opacity=0.75, line_opacity=0.3,
        nan_fill_color="#eeeeee", legend_name=value_name,
    ).add_to(m)

    if out_html:
        out_html = Path(out_html)
        out_html.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_html))
    return m
