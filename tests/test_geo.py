"""prepare_boundaries(): BNG -> WGS84, England & Wales filter, code aliases."""
import json

import pytest

from src.data.geo import CODE_ALIASES, prepare_boundaries

pytest.importorskip("pyproj")


def _square(x, y, d=1000):
    return [[[x, y], [x + d, y], [x + d, y + d], [x, y + d], [x, y]]]


def _fc(features, crs="EPSG:27700"):
    return {"type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:{crs}"}},
            "features": features}


def _feat(code, name, coords, kind="Polygon"):
    return {"type": "Feature", "properties": {"LAD24CD": code, "LAD24NM": name, "BNG_E": 1},
            "geometry": {"type": kind, "coordinates": coords}}


def test_reprojects_filters_and_aliases(tmp_path):
    src = tmp_path / "raw.geojson"
    src.write_text(json.dumps(_fc([
        _feat("E08000016", "Barnsley", _square(435000, 405000)),          # aliased
        _feat("W06000015", "Cardiff", [_square(318000, 176000)], "MultiPolygon"),
        _feat("S12000036", "City of Edinburgh", _square(325000, 673000)),   # dropped
    ])))
    out = tmp_path / "wgs.geojson"
    s = prepare_boundaries(src, out)
    g = json.loads(out.read_text())
    codes = [f["properties"]["LAD24CD"] for f in g["features"]]
    assert codes == [CODE_ALIASES["E08000016"], "W06000015"] and s["aliased"] == 1
    assert set(g["features"][0]["properties"]) == {"LAD24CD", "LAD24NM"}
    lon, lat = g["features"][0]["geometry"]["coordinates"][0][0]
    assert -1.6 < lon < -1.4 and 53.4 < lat < 53.6                       # Barnsley-ish
    ring = g["features"][1]["geometry"]["coordinates"][0][0]
    assert ring[0] == ring[-1]                                            # stays closed


def test_wgs84_input_passes_through(tmp_path):
    src = tmp_path / "raw.geojson"
    src.write_text(json.dumps(_fc([_feat("E07000223", "Adur",
                                         [[[-0.3, 50.8], [-0.2, 50.8], [-0.2, 50.9], [-0.3, 50.8]]])],
                                  crs="EPSG:4326")))
    s = prepare_boundaries(src, tmp_path / "o.geojson")
    assert s["source_crs"] == "EPSG:4326" and s["features"] == 1
