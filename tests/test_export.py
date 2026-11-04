"""The download ZIP: what goes in comes back out, in strict CSV and JSON."""
import csv
import io
import json
import zipfile

import numpy as np
import pytest

import export_core as ec


def test_zip_round_trip():
    tables = {
        "trace": {
            "description": "a trace",
            "columns": {"pixel": np.arange(4), "rate": [1.5, float("nan"), 3.0, 1e-7]},
        }
    }
    meta = {"inputs": {"alpha": np.float64(45.0), "bad": float("inf"), "flag": np.bool_(True)}}
    blob = ec.build_zip(tables, meta)

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        assert sorted(zf.namelist()) == ["README.txt", "run.json", "trace.csv"]
        rows = list(csv.reader(io.StringIO(zf.read("trace.csv").decode("utf-8"))))
        run = json.loads(zf.read("run.json").decode("utf-8"))

    assert rows[0] == ["pixel", "rate"]
    assert [r[0] for r in rows[1:]] == ["0", "1", "2", "3"]
    assert rows[2][1] == ""                      # NaN → empty cell
    assert float(rows[4][1]) == 1e-7             # full precision
    assert run["inputs"] == {"alpha": 45.0, "bad": None, "flag": True}
    assert run["files"]["trace.csv"]["columns"] == ["pixel", "rate"]


def test_unequal_columns_raise():
    with pytest.raises(ValueError):
        ec.to_csv({"a": [1, 2], "b": [1]})


def test_floats_keep_full_precision():
    v = 0.1 + 0.2
    text = ec.to_csv({"v": [v]})
    assert float(text.splitlines()[1]) == v
