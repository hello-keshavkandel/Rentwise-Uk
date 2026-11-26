"""Every runtime dependency is pinned exactly.

An unpinned `pyarrow>=14` once resolved to a release that needs NumPy 2 on Python 3.11,
while we pin NumPy 1.26.4 — every Streamlit table then crashed. Exact pins make a fresh
install (Mac, CI, Docker, Streamlit Cloud) get the versions we tested."""
import re

import config


def test_runtime_requirements_are_exactly_pinned():
    loose = []
    for line in (config.ROOT / "requirements.txt").read_text().splitlines():
        spec = line.split("#", 1)[0].strip()
        if spec and not re.match(r"^[A-Za-z0-9_.\-\[\]]+==[\w.]+$", spec):
            loose.append(spec)
    assert not loose, f"pin these exactly in requirements.txt: {loose}"


def test_pyarrow_works_with_pinned_numpy():
    import numpy
    import pyarrow as pa          # raises ImportError if the pair is incompatible
    assert pa.array(numpy.arange(3)).to_pylist() == [0, 1, 2]
