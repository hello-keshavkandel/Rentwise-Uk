"""
osm.py  —  rent-relevant amenities from OpenStreetMap, per local authority (Module 3).

Input : Geofabrik extracts (England + Wales) in .osm.pbf, from
        https://download.geofabrik.de/europe/united-kingdom.html
Output: data/raw/osm_amenities.csv   — one row per amenity: lad_code, category, lat, lon
        data/processed/osm_amenities_by_la.csv — counts per LA x category (small, committed)

We stream the PBF once with pyosmium (never loading it into memory), keep nodes and
closed ways whose tags match CATEGORIES (ways are reduced to the average of their
nodes' coordinates), then assign each point to a local authority by point-in-polygon
against the prepared WGS84 boundaries (src/data/geo.py), using a shapely STRtree.

    python -m src.data.osm          # reads data/raw/osm/*.osm.pbf

OSM is crowd-sourced: completeness varies by area (well-mapped cities vs sparse
rural areas), so counts are best read as relative amenity density, not a census.
Data © OpenStreetMap contributors, ODbL.
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("rentwise.osm")

# category -> list of (key, allowed values) — a feature matches if ANY pair matches
CATEGORIES: dict[str, list[tuple[str, set[str]]]] = {
    "rail_station":    [("railway", {"station", "halt"})],
    "bus_stop":        [("highway", {"bus_stop"})],
    "supermarket":     [("shop", {"supermarket"})],
    "convenience":     [("shop", {"convenience"})],
    "gp_clinic":       [("amenity", {"doctors", "clinic"}), ("healthcare", {"doctor", "clinic"})],
    "hospital":        [("amenity", {"hospital"})],
    "pharmacy":        [("amenity", {"pharmacy"})],
    "restaurant_cafe": [("amenity", {"restaurant", "cafe", "fast_food"})],
    "pub_bar":         [("amenity", {"pub", "bar"})],
    "park":            [("leisure", {"park"})],
    "gym":             [("leisure", {"fitness_centre"})],
}
_KEYS = {k for rules in CATEGORIES.values() for k, _ in rules}


def categorise(tags) -> str | None:
    """First matching category for an OSM tag mapping, else None."""
    for cat, rules in CATEGORIES.items():
        for k, vals in rules:
            v = tags.get(k)
            if v is not None and v in vals:
                return cat
    return None


def extract_points(pbf_path: str | Path) -> pd.DataFrame:
    """Stream one .osm.pbf/.osm file -> DataFrame[category, lat, lon, osm_type].

    Tag filtering runs in osmium's C++ layer (KeyFilter), so Python only sees the
    few hundred thousand candidate objects, not every node in England."""
    import osmium

    rows: list[tuple] = []
    fp = (osmium.FileProcessor(str(pbf_path))
          .with_locations()                                   # node coords for ways
          .with_filter(osmium.filter.EntityFilter(osmium.osm.NODE | osmium.osm.WAY))
          .with_filter(osmium.filter.KeyFilter(*sorted(_KEYS))))
    for obj in fp:
        cat = categorise(obj.tags)
        if not cat:
            continue
        if obj.is_node():
            if obj.location.valid():
                rows.append((cat, obj.location.lat, obj.location.lon, "n"))
        elif obj.is_way():
            try:
                pts = [(nd.lat, nd.lon) for nd in obj.nodes if nd.location.valid()]
            except osmium.InvalidLocationError:
                continue
            if len(pts) > 1 and pts[0] == pts[-1]:            # closed way: don't double-count
                pts = pts[:-1]
            if pts:
                rows.append((cat, sum(p[0] for p in pts) / len(pts),
                             sum(p[1] for p in pts) / len(pts), "w"))
    return pd.DataFrame(rows, columns=["category", "lat", "lon", "osm_type"])


def assign_lad(points: pd.DataFrame, boundaries_geojson: str | Path,
               code_field: str = "LAD24CD") -> pd.DataFrame:
    """Add lad_code by point-in-polygon (points outside every LA are dropped)."""
    from shapely import STRtree, points as mk_points
    from shapely.geometry import shape

    geo = json.loads(Path(boundaries_geojson).read_text())
    from shapely import make_valid
    polys, codes = [], []
    for f in geo["features"]:
        try:
            g = shape(f["geometry"])
        except ValueError:                       # malformed ring: rebuild from valid rings only
            log.warning("skipping malformed geometry for %s", f["properties"].get(code_field))
            continue
        g = g if g.is_valid else make_valid(g)
        if not g.is_empty:
            polys.append(g)
            codes.append(f["properties"][code_field])
    tree = STRtree(polys)
    pts = mk_points(points["lon"].to_numpy(), points["lat"].to_numpy())
    pt_idx, poly_idx = tree.query(pts, predicate="within")
    lad = pd.Series(pd.NA, index=range(len(points)), dtype="string")
    lad.iloc[pt_idx] = [codes[i] for i in poly_idx]
    out = points.reset_index(drop=True).assign(lad_code=lad.values)
    dropped = int(out["lad_code"].isna().sum())
    if dropped:
        log.info("assign_lad: %d points outside England & Wales LAs dropped", dropped)
    return out.dropna(subset=["lad_code"]).reset_index(drop=True)


def counts_by_la(points: pd.DataFrame) -> pd.DataFrame:
    """Wide LA x category count table (zeros filled)."""
    wide = (points.groupby(["lad_code", "category"]).size()
                  .unstack(fill_value=0).reset_index())
    for c in CATEGORIES:                        # stable schema even if a category is absent
        if c not in wide.columns:
            wide[c] = 0
    return wide[["lad_code", *CATEGORIES]]


def build(pbf_paths, boundaries, out_points=None, out_counts=None) -> dict:
    frames = []
    for p in pbf_paths:
        df = extract_points(p)
        log.info("OSM %s: %d candidate amenities", Path(p).name, len(df))
        frames.append(df)
    pts = assign_lad(pd.concat(frames, ignore_index=True), boundaries)
    counts = counts_by_la(pts)
    if out_points:
        Path(out_points).parent.mkdir(parents=True, exist_ok=True)
        pts[["lad_code", "category", "lat", "lon"]].to_csv(out_points, index=False)
    if out_counts:
        Path(out_counts).parent.mkdir(parents=True, exist_ok=True)
        counts.to_csv(out_counts, index=False)
    summary = {"points": len(pts), "las": int(counts["lad_code"].nunique()),
               "by_category": pts["category"].value_counts().to_dict()}
    log.info("OSM amenities: %s", summary)
    return summary


if __name__ == "__main__":  # pragma: no cover
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import config
    pbfs = sorted(glob.glob(str(config.DATA_RAW / "osm" / "*.osm.pbf")))
    if not pbfs:
        sys.exit("Put england-latest.osm.pbf and wales-latest.osm.pbf in data/raw/osm/")
    print(build(pbfs, config.DATA_PROCESSED / "lad_boundaries_wgs84.geojson",
                out_points=config.DATA_RAW / "osm_amenities.csv",
                out_counts=config.DATA_PROCESSED / "osm_amenities_by_la.csv"))
