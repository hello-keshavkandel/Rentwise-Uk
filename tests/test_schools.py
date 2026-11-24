"""score_schools / build_school_quality: priority order, no guessing, LAD join."""
import pandas as pd

from src.data.schools import OE, OEIF_DATE, SUB, UNGRADED, UNGRADED_DATE, build_school_quality, score_schools
from src.modules.module3_geo import school_quality


def _mi(rows):
    base = {"URN": None, OE: None, OEIF_DATE: None, UNGRADED: None, UNGRADED_DATE: None, **{c: None for c in SUB}}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_priority_overall_then_submean_then_ungraded_and_no_guessing():
    mi = _mi([
        {"URN": "1", OE: "2", **{c: "1" for c in SUB}, OEIF_DATE: "01/03/2022"},     # overall wins
        {"URN": "2", OE: "Not judged", SUB[0]: "1", SUB[1]: "2", SUB[2]: "2", SUB[3]: "3",
         OEIF_DATE: "10/10/2024"},                                                   # sub-mean = 2.0
        {"URN": "3", UNGRADED: "School remains Outstanding", UNGRADED_DATE: "05/05/2023"},
        {"URN": "4", UNGRADED: "Standards maintained"},                               # ambiguous -> blank
        {"URN": "5"},                                                                 # nothing -> blank
    ])
    s = score_schools(mi).set_index("urn")
    assert s.loc["1", "ofsted_grade"] == 2 and s.loc["1", "grade_source"] == "overall"
    assert s.loc["2", "ofsted_grade"] == 2.0 and s.loc["2", "grade_source"] == "sub_mean"
    assert s.loc["3", "ofsted_grade"] == 1 and s.loc["3", "grade_source"] == "ungraded"
    assert s.loc["3", "inspection_date"].year == 2023
    assert pd.isna(s.loc["4", "ofsted_grade"]) and pd.isna(s.loc["5", "ofsted_grade"])


def test_build_joins_gias_district_and_aggregates(tmp_path):
    mi = _mi([{"URN": "1", OE: "1", OEIF_DATE: "01/01/2022"},
              {"URN": "2", OE: "3", OEIF_DATE: "01/01/2023"},
              {"URN": "3", OE: "2", OEIF_DATE: "01/01/2021"}])
    mi.to_csv(tmp_path / "mi.csv", index=False)
    pd.DataFrame({"URN": ["1", "2", "3"], "DistrictAdministrative (code)": ["E1", "E1", "E2"]}
                 ).to_csv(tmp_path / "gias.csv", index=False, encoding="cp1252")
    _, la, summ = build_school_quality(tmp_path / "mi.csv", tmp_path / "gias.csv")
    la = la.set_index("lad_code")
    assert summ["scored"] == 3
    assert la.loc["E1", "school_quality"] == 3.0          # (5-1 + 5-3) / 2
    assert la.loc["E1", "share_good_or_better"] == 0.5
    assert la.loc["E2", "schools_scored"] == 1


def test_module3_school_quality_accepts_numeric_grades():
    out = school_quality(pd.DataFrame({"lad_code": ["A", "A"], "ofsted_rating": [1, 3]})).set_index("lad_code")
    assert out.loc["A", "school_quality"] == 3.0
