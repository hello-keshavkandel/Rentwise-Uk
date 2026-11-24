"""OSM amenity extraction: tag categorisation, way centroids, LAD assignment."""
import json

import pytest

from src.data.osm import CATEGORIES, assign_lad, build, categorise, counts_by_la, extract_points

osmium = pytest.importorskip("osmium")
pytest.importorskip("shapely")

OSM_XML = """<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="test">
  <node id="1" lat="0.5" lon="0.5"><tag k="railway" v="station"/></node>
  <node id="2" lat="0.6" lon="0.6"><tag k="amenity" v="pub"/></node>
  <node id="3" lat="0.7" lon="0.7"><tag k="amenity" v="bench"/></node>
  <node id="4" lat="5.0" lon="5.0"><tag k="amenity" v="pharmacy"/></node>
  <node id="10" lat="1.2" lon="1.2"/><node id="11" lat="1.2" lon="1.4"/>
  <node id="12" lat="1.4" lon="1.4"/><node id="13" lat="1.4" lon="1.2"/>
  <way id="100"><nd ref="10"/><nd ref="11"/><nd ref="12"/><nd ref="13"/><nd ref="10"/>
    <tag k="shop" v="supermarket"/></way>
</osm>"""


def _boundaries(path):
    sq = lambda x, y: [[[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1], [x, y]]]  # noqa: E731
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"LAD24CD": "E1"}, "geometry": {"type": "Polygon", "coordinates": sq(0, 0)}},
        {"type": "Feature", "properties": {"LAD24CD": "W1"}, "geometry": {"type": "Polygon", "coordinates": sq(1, 1)}},
    ]}))
    return path


def test_categorise():
    assert categorise({"railway": "halt"}) == "rail_station"
    assert categorise({"healthcare": "doctor"}) == "gp_clinic"
    assert categorise({"amenity": "bench"}) is None


def test_extract_assign_count(tmp_path):
    osm = tmp_path / "t.osm"; osm.write_text(OSM_XML)
    pts = extract_points(osm)
    assert sorted(pts["category"]) == ["pharmacy", "pub_bar", "rail_station", "supermarket"]
    way = pts[pts.osm_type == "w"].iloc[0]
    assert way.lat == pytest.approx(1.3) and way.lon == pytest.approx(1.3)     # centre of the square
    lad = assign_lad(pts, _boundaries(tmp_path / "b.geojson"))
    assert len(lad) == 3                                   # pharmacy at (5,5) is outside -> dropped
    assert dict(zip(lad.category, lad.lad_code))["supermarket"] == "W1"
    c = counts_by_la(lad).set_index("lad_code")
    assert list(c.columns) == list(CATEGORIES)
    assert c.loc["E1", "rail_station"] == 1 and c.loc["E1", "pub_bar"] == 1 and c.loc["W1", "supermarket"] == 1


def test_build_writes_outputs(tmp_path):
    osm = tmp_path / "t.osm"; osm.write_text(OSM_XML)
    s = build([osm], _boundaries(tmp_path / "b.geojson"),
              out_points=tmp_path / "p.csv", out_counts=tmp_path / "c.csv")
    assert s["points"] == 3 and (tmp_path / "p.csv").exists() and (tmp_path / "c.csv").exists()
