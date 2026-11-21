"""
module3_geo.py  —  Module 3 geospatial feature engineering.

Turns raw spatial inputs (OpenStreetMap amenities, Ofsted schools, crime) into
per-local-authority features and a PCA composite spatial index that feeds the shared
feature store.

Design note: the heavy "which local authority does this point fall in?" step
(point-in-polygon) needs geopandas and is OPTIONAL (`assign_lad`). The core
aggregation + PCA are pure pandas/sklearn, so they're always available and fully
unit-tested — the pipeline expects amenity/school rows that already carry a
`lad_code` (OSM extracts and Ofsted data both include the local authority).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("rentwise.module3")

OFSTED_SCORE = {"outstanding": 4, "good": 3, "requires improvement": 2, "inadequate": 1}


def aggregate_points(points: pd.DataFrame, lad_col: str = "lad_code",
                     category_col: str = "category") -> pd.DataFrame:
    """Count amenity points per local authority per category -> wide per-LA table."""
    counts = (points.groupby([lad_col, category_col]).size()
              .unstack(fill_value=0))
    counts.columns = [f"n_{c}" for c in counts.columns]
    return counts.reset_index()


def school_quality(schools: pd.DataFrame, lad_col: str = "lad_code",
                   rating_col: str = "ofsted_rating") -> pd.DataFrame:
    """Mean Ofsted rating per LA (Outstanding=4 … Inadequate=1)."""
    s = schools.copy()
    s["_score"] = s[rating_col].astype(str).str.strip().str.lower().map(OFSTED_SCORE)
    out = s.groupby(lad_col)["_score"].mean().rename("school_quality").reset_index()
    return out


def build_spatial_features(base_las: pd.DataFrame, *tables: pd.DataFrame,
                           lad_col: str = "lad_code") -> pd.DataFrame:
    """Left-merge per-LA feature tables onto a base LA list; missing counts -> 0."""
    out = base_las[[lad_col]].drop_duplicates().copy()
    for t in tables:
        out = out.merge(t, on=lad_col, how="left")
    count_cols = [c for c in out.columns if c.startswith("n_")]
    out[count_cols] = out[count_cols].fillna(0)
    return out


def pca_composite(df: pd.DataFrame, cols: list[str], name: str = "spatial_index",
                  seed: int = 42) -> pd.DataFrame:
    """Standardise `cols` and reduce to a single PCA composite spatial index."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    X = df[cols].astype(float)
    X = X.fillna(X.median())
    Xs = StandardScaler().fit_transform(X)
    pc = PCA(n_components=1, random_state=seed).fit_transform(Xs)[:, 0]
    # orient so higher = "more amenities/better" (positive correlation with the mean)
    if np.corrcoef(pc, Xs.mean(axis=1))[0, 1] < 0:
        pc = -pc
    out = df.copy()
    out[name] = pc
    return out


def assign_lad(points_latlon: pd.DataFrame, boundaries_geojson,
               lad_field: str = "LAD24CD", lat: str = "lat", lon: str = "lon") -> pd.DataFrame:
    """
    Optional: assign a local-authority code to lat/lon points via point-in-polygon.
    Requires geopandas; raises a clear error if it isn't installed.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("assign_lad needs geopandas+shapely (pip install geopandas). "
                           "If your points already have lad_code, skip this step.") from e
    gdf_pts = gpd.GeoDataFrame(
        points_latlon.copy(),
        geometry=[Point(xy) for xy in zip(points_latlon[lon], points_latlon[lat])],
        crs="EPSG:4326")
    bounds = gpd.read_file(boundaries_geojson).to_crs("EPSG:4326")
    joined = gpd.sjoin(gdf_pts, bounds[[lad_field, "geometry"]], how="left", predicate="within")
    out = points_latlon.copy()
    out["lad_code"] = joined[lad_field].values
    return out
