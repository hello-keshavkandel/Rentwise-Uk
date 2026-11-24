"""
schools.py  —  historic Ofsted school quality per local authority (Module 3).

Ofsted dropped single-word overall grades (Sept 2024) and moved to report cards
(Nov 2025), so there is no longer ONE current rating per school. Policy chosen
for RentWise: use each school's **last official grade under the 2019 framework
(OEIF)**, clearly labelled as historic, with its inspection date.

For each school (Ofsted monthly MI "latest inspections" file), in priority order:
  1. `overall`   — Latest OEIF overall effectiveness (1–4)
  2. `sub_mean`  — mean of the four OEIF key judgements (quality of education,
                   behaviour, personal development, leadership), used when the
                   overall grade was "Not judged" (Sept 2024 – Nov 2025 inspections).
                   Correlates 0.92 with the overall grade where both exist.
  3. `ungraded`  — an ungraded inspection that explicitly confirms a grade
                   ("School remains Outstanding" -> 1, "School remains Good…" -> 2).
                   Ambiguous outcomes ("Standards maintained", …) are NOT mapped.
Schools with only a new report card are left unscored (no mixing of scales).

The district (LAD) code comes from DfE Get Information About Schools (GIAS) by
URN. Welsh schools are inspected by Estyn, not Ofsted, so Welsh LAs get no score.

Ofsted scale: 1 = Outstanding … 4 = Inadequate. We also expose `quality` =
5 − grade (4 = best) so "higher is better" in features and maps.

    python -m src.data.schools
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("rentwise.schools")

OE = "Latest OEIF overall effectiveness"
SUB = ["Latest OEIF quality of education", "Latest OEIF behaviour and attitudes",
       "Latest OEIF personal development", "Latest OEIF effectiveness of leadership and management"]
OEIF_DATE = "Inspection start date of latest OEIF graded inspection"
UNGRADED = "Ungraded inspection overall outcome"
UNGRADED_DATE = "Date of latest ungraded inspection"

UNGRADED_MAP = {
    "school remains outstanding": 1.0,
    "school remains good": 2.0,
    "school remains good (improving) - s5 next": 2.0,
    "school remains good (concerns) - s5 next": 2.0,
}


def _read_csv(path, enc_first="utf-8-sig"):
    for enc in (enc_first, "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"could not decode {path}")


def score_schools(mi: pd.DataFrame) -> pd.DataFrame:
    """Per-school historic grade (1–4), its source and date, from the Ofsted MI file."""
    overall = pd.to_numeric(mi[OE], errors="coerce")
    sub_mean = mi[SUB].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    ungraded = mi[UNGRADED].astype("string").str.strip().str.lower().map(UNGRADED_MAP)
    ungraded = pd.to_numeric(ungraded, errors="coerce")

    grade = overall.copy()
    source = pd.Series(np.where(overall.notna(), "overall", None), index=mi.index, dtype=object)
    use_sub = grade.isna() & sub_mean.notna()
    grade[use_sub], source[use_sub] = sub_mean[use_sub], "sub_mean"
    use_ung = grade.isna() & ungraded.notna()
    grade[use_ung], source[use_ung] = ungraded[use_ung], "ungraded"

    oeif_date = pd.to_datetime(mi[OEIF_DATE], dayfirst=True, errors="coerce")
    ung_date = pd.to_datetime(mi.get(UNGRADED_DATE), dayfirst=True, errors="coerce")
    date = oeif_date.where(source.isin(["overall", "sub_mean"]), ung_date)

    return pd.DataFrame({"urn": mi["URN"].astype(str).str.strip(),
                         "ofsted_grade": grade, "grade_source": source,
                         "inspection_date": date})


def build_school_quality(mi_path, gias_path, out_schools=None, out_la=None):
    """Join MI grades to GIAS district codes; write per-school and per-LA tables."""
    mi = _read_csv(mi_path)
    gias = _read_csv(gias_path, enc_first="cp1252")[["URN", "DistrictAdministrative (code)"]]
    gias = gias.rename(columns={"URN": "urn", "DistrictAdministrative (code)": "lad_code"})
    gias["urn"] = gias["urn"].astype(str).str.strip()

    s = score_schools(mi).merge(gias, on="urn", how="left")
    s["quality"] = 5 - s["ofsted_grade"]                       # 4 = best
    scored = s.dropna(subset=["ofsted_grade", "lad_code"])

    la = (scored.groupby("lad_code")
                .agg(school_quality=("quality", "mean"),
                     schools_scored=("quality", "size"),
                     share_good_or_better=("ofsted_grade", lambda g: float((g <= 2).mean())),
                     median_inspection_year=("inspection_date",
                                             lambda d: float(d.dt.year.median())))
                .reset_index())

    summary = {"schools": len(s), "scored": len(scored),
               "by_source": scored["grade_source"].value_counts().to_dict(),
               "las": len(la)}
    log.info("school quality: %s", summary)
    for df, path in ((s, out_schools), (la, out_la)):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
    return s, la, summary


def _latest(pattern: str) -> Path | None:
    files = sorted(glob.glob(pattern))
    return Path(files[-1]) if files else None


if __name__ == "__main__":  # pragma: no cover
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import config
    d = config.DATA_RAW / "schools"
    mi = _latest(str(d / "Management_information*latest_inspections*.csv"))
    gias = _latest(str(d / "edubasealldata*.csv"))
    if not (mi and gias):
        sys.exit("Need the Ofsted MI 'latest inspections' CSV and GIAS edubasealldata CSV in data/raw/schools/")
    # per-school table feeds the source registry ('ofsted'); the small per-LA table is committed for the app
    _, _, summ = build_school_quality(
        mi, gias,
        out_schools=config.DATA_RAW / "ofsted_schools.csv",
        out_la=config.DATA_PROCESSED / "school_quality_by_la.csv")
    print(summ)
