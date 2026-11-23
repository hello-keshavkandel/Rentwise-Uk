"""
geo.py  —  prepare local-authority boundaries for web maps.

The ONS "Local Authority Districts (May 2024) Boundaries UK BGC" GeoJSON comes in
British National Grid (EPSG:27700, metres). Web maps (folium/Leaflet) need WGS84
longitude/latitude, so it must be reprojected once. While doing that we:

* keep England & Wales only (the scope of RentWise),
* map superseded LA codes to the ones our rent data uses (CODE_ALIASES),
* round coordinates to ~10 m (finer than the 20 m BGC source) and drop repeated points, shrinking the file so the
  deployed app can load it quickly,
* keep only the properties the maps need (code + name).

    python -m src.data.geo      # data/raw/geo/lad_boundaries.geojson
                                #   -> data/processed/lad_boundaries_wgs84.geojson

The processed file is small and committed, so the live app has maps without the
raw download. pyproj is only needed to run this preparation step.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("rentwise.geo")

# Codes that changed after the May 2024 boundary vintage. Barnsley and Sheffield
# were re-coded after a minor boundary change in April 2025; ONS rent data uses
# the new codes, the 2024 boundaries (and LSOA lookup) the old ones.
CODE_ALIASES = {
    "E08000016": "E08000038",   # Barnsley
    "E08000019": "E08000039",   # Sheffield
}

BNG = "EPSG:27700"
WGS84 = "EPSG:4326"


def _crs_of(geo: dict) -> str:
    """Declared CRS, else infer from coordinate magnitude (metres vs degrees)."""
    name = (geo.get("crs") or {}).get("properties", {}).get("name", "")
    if "27700" in name:
        return BNG
    if name and ("4326" in name or "CRS84" in name):
        return WGS84
    # undeclared: sniff the first coordinate
    c = geo["features"][0]["geometry"]["coordinates"]
    while isinstance(c[0], list):
        c = c[0]
    return BNG if abs(c[0]) > 180 or abs(c[1]) > 90 else WGS84


def _map_rings(coords, fn):
    """Apply fn to every [x, y] ring in a (Multi)Polygon coordinate tree."""
    if coords and isinstance(coords[0][0], (int, float)):
        return fn(coords)
    return [_map_rings(c, fn) for c in coords]


def prepare_boundaries(src: str | Path, dst: str | Path, countries=("E", "W"),
                       code_field: str = "LAD24CD", name_field: str = "LAD24NM",
                       precision: int = 4) -> dict:
    """Reproject/trim/alias an LA boundary GeoJSON for web maps. Returns a summary."""
    geo = json.loads(Path(src).read_text())
    crs = _crs_of(geo)
    if crs == BNG:
        from pyproj import Transformer
        tf = Transformer.from_crs(BNG, WGS84, always_xy=True)

        def to_wgs(xs, ys):
            return tf.transform(xs, ys)
    else:
        def to_wgs(xs, ys):
            return xs, ys

    def ring_fn(ring):
        xs, ys = zip(*[(p[0], p[1]) for p in ring])
        lons, lats = to_wgs(list(xs), list(ys))
        out, prev = [], None
        for lo, la in zip(lons, lats):
            pt = [round(lo, precision), round(la, precision)]
            if pt != prev:
                out.append(pt)
                prev = pt
        if len(out) >= 2 and out[0] != out[-1]:   # keep rings closed
            out.append(out[0])
        return out

    feats, aliased = [], 0
    for f in geo["features"]:
        code = f["properties"].get(code_field)
        if not code or code[0] not in countries:
            continue
        if code in CODE_ALIASES:
            code, aliased = CODE_ALIASES[code], aliased + 1
        g = f["geometry"]
        feats.append({"type": "Feature",
                      "properties": {code_field: code, name_field: f["properties"].get(name_field)},
                      "geometry": {"type": g["type"], "coordinates": _map_rings(g["coordinates"], ring_fn)}})

    out = {"type": "FeatureCollection", "features": feats}
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, separators=(",", ":")))
    summary = {"source_crs": crs, "features": len(feats), "aliased": aliased,
               "bytes": dst.stat().st_size}
    log.info("boundaries prepared: %s", summary)
    return summary


if __name__ == "__main__":  # pragma: no cover
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import config
    print(prepare_boundaries(config.DATA_RAW / "geo" / "lad_boundaries.geojson",
                             config.DATA_PROCESSED / "lad_boundaries_wgs84.geojson"))
