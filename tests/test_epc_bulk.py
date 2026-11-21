"""load_bulk(): any file name, header-based detection, chunking, postcode filter."""
import pytest

from src.data.epc import build_epc_lookup, load_bulk

HDR = "LMK_KEY,ADDRESS1,ADDRESS2,POSTCODE,TOTAL_FLOOR_AREA,NUMBER_HABITABLE_ROOMS,PROPERTY_TYPE,LODGEMENT_DATE,CURRENT_ENERGY_RATING\n"


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HDR + "".join(rows))


def test_reads_any_csv_name_and_skips_recommendations(tmp_path):
    _write(tmp_path / "2025" / "domestic-2025.csv",
           ["1,12 High St,,AB1 2CD,80,4,House,2025-01-01,C\n"])
    _write(tmp_path / "certificates.csv",      # legacy name still works
           ["2,3 Low Rd,,EF3 4GH,55,3,Flat,2024-06-01,D\n"])
    (tmp_path / "recommendations.csv").write_text("LMK_KEY,IMPROVEMENT\n1,loft\n")
    (tmp_path / "notes.csv").write_text("a,b\n1,2\n")      # not a certificates table
    df = load_bulk(tmp_path)
    assert sorted(df["postcode"]) == ["AB1 2CD", "EF3 4GH"]
    assert {"total_floor_area", "number_habitable_rooms"} <= set(df.columns)


def test_postcode_filter_and_chunking(tmp_path):
    rows = [f"{i},{i} Road,,ZZ{i % 50} 1AA,{60 + i % 40},{2 + i % 4},House,2024-01-01,C\n"
            for i in range(1000)]
    _write(tmp_path / "big.csv", rows)
    keep = ["zz1 1aa", "ZZ2  1AA"]                       # case/space-insensitive
    df = load_bulk(tmp_path, postcodes=keep, chunksize=97)
    assert set(df["postcode"]) == {"ZZ1 1AA", "ZZ2 1AA"}
    assert len(df) == 40                                  # 20 rows per postcode


def test_no_certificate_files_raises(tmp_path):
    (tmp_path / "notes.csv").write_text("a,b\n1,2\n")
    with pytest.raises(FileNotFoundError):
        load_bulk(tmp_path)


def test_build_lookup_end_to_end(tmp_path):
    _write(tmp_path / "raw" / "x.csv",
           ["1,12 High St,,AB1 2CD,80,4,House,2025-01-01,C\n",
            "2,14 High St,,AB1 2CD,95,5,House,2025-02-01,C\n"])
    out = tmp_path / "epc_lookup.parquet"
    lk = build_epc_lookup(tmp_path / "raw", out, postcodes=["AB12CD"])
    assert len(lk) == 2 and set(lk["house_no"]) == {"12", "14"}
    assert out.exists() or out.with_suffix(".csv").exists()


def test_reads_csv_inside_zip_without_extracting(tmp_path):
    import zipfile
    z = tmp_path / "domestic-csv.zip"
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("domestic-csv/certificates-2025.csv",
                    HDR + "1,12 High St,,AB1 2CD,80,4,House,2025-01-01,C\n"
                          "2,3 Low Rd,,EF3 4GH,55,3,Flat,2024-06-01,D\n")
        zf.writestr("domestic-csv/recommendations-2025.csv", "LMK_KEY,IMPROVEMENT\n1,loft\n")
    df = load_bulk(tmp_path, postcodes=["AB12CD"], chunksize=1)
    assert list(df["postcode"]) == ["AB1 2CD"]
    assert not any(tmp_path.glob("**/certificates-2025.csv"))   # nothing extracted


def test_truncated_zip_is_skipped_not_fatal(tmp_path):
    (tmp_path / "partial.zip").write_bytes(b"PK\x03\x04 incomplete download")
    _write(tmp_path / "ok.csv", ["1,12 High St,,AB1 2CD,80,4,House,2025-01-01,C\n"])
    assert len(load_bulk(tmp_path)) == 1
