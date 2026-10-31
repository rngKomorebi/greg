"""
export_core.py — package GREG plot data for download.

An export is a ZIP holding one CSV per table plus ``run.json`` (inputs,
results, and a description of every CSV) and a short ``README.txt``.
CSV and JSON keep the data readable outside Python and safe to load, unlike a
pickled figure. The CSVs carry no comment lines, so they load directly with
``pandas.read_csv(path)`` or ``numpy.genfromtxt(path, delimiter=",",
names=True)``; an empty cell is a missing value (NaN).

Public API
----------
    build_zip(tables, meta) → bytes
    to_csv(columns)         → str

``tables`` is ``{name: {"description": str, "columns": {column: values}}}``,
where every column is a 1-D sequence of the same length.
"""

from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from datetime import datetime, timezone
from typing import Mapping, Sequence

import numpy as np

# Bump when the layout of run.json or the CSVs changes incompatibly.
EXPORT_FORMAT_VERSION = 1


def _cell(value) -> str:
    """Format one CSV cell: full float precision, NaN/None as an empty cell."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return ""
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    return str(value)


def to_csv(columns: Mapping[str, Sequence]) -> str:
    """Render equal-length columns as CSV text with a header row."""
    names = list(columns)
    data = [list(columns[n]) for n in names]
    lengths = {len(col) for col in data}
    if len(lengths) > 1:
        raise ValueError(
            "columns differ in length: "
            + ", ".join(f"{n}={len(c)}" for n, c in zip(names, data))
        )
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(names)
    for row in zip(*data):
        writer.writerow(_cell(v) for v in row)
    return buf.getvalue()


def _jsonable(obj):
    """Convert numpy types, tuples and non-finite floats to strict-JSON values."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    return obj


def _readme(meta: Mapping, tables: Mapping[str, Mapping]) -> str:
    lines = [
        f"GREG data export — {meta.get('page', '')}",
        f"Created {meta['created']}",
        "",
        "Files",
        "  run.json  inputs, results and a description of every CSV",
    ]
    for name, table in tables.items():
        lines.append(f"  {name}.csv  {table['description']}")
    lines += [
        "",
        "Loading",
        "  import pandas as pd",
        "  df = pd.read_csv('<name>.csv')",
        "",
        "  import numpy as np",
        "  arr = np.genfromtxt('<name>.csv', delimiter=',', names=True)",
        "",
        "Empty cells are missing values (NaN).",
        "",
    ]
    return "\n".join(lines)


def build_zip(tables: Mapping[str, Mapping], meta: Mapping) -> bytes:
    """Build the export ZIP from data tables and run metadata.

    ``meta`` is written to run.json after a ``format``/``created`` header and
    followed by a ``files`` entry mapping each CSV to its description and
    columns.
    """
    meta = {
        "format": f"greg-export/{EXPORT_FORMAT_VERSION}",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **meta,
        "files": {
            f"{name}.csv": {
                "description": t["description"],
                "columns": list(t["columns"]),
            }
            for name, t in tables.items()
        },
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, table in tables.items():
            zf.writestr(f"{name}.csv", to_csv(table["columns"]))
        zf.writestr(
            "run.json",
            json.dumps(_jsonable(meta), indent=2, ensure_ascii=False,
                       allow_nan=False),
        )
        zf.writestr("README.txt", _readme(meta, tables))
    return buf.getvalue()
