"""Minimal AGS4 reader/writer (UK/international geotechnical data transfer format).

Reads every GROUP into a DataFrame and converts LOCA + GEOL (+ optional HDPH
orientation and ISPT SPT results) into a :class:`BoreholeSet`.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd

from .boreholes import BoreholeSet


def read_ags4(src) -> dict[str, pd.DataFrame]:
    """Parse an AGS4 file (path, bytes or text) into {group: DataFrame}."""
    if hasattr(src, "read"):
        text = src.read()
    elif isinstance(src, (bytes, bytearray)):
        text = src
    elif isinstance(src, str) and "\n" in src:
        text = src
    else:
        text = Path(src).read_bytes()
    if isinstance(text, (bytes, bytearray)):
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                text = text.decode(enc)
                break
            except UnicodeDecodeError:
                continue
    groups: dict[str, pd.DataFrame] = {}
    cur, heads, rows, units = None, None, [], {}

    def flush():
        if cur and heads:
            df = pd.DataFrame(rows, columns=heads)
            df.attrs["units"] = units.get(cur, {})
            groups[cur] = df

    for rec in csv.reader(io.StringIO(text)):
        if not rec:
            continue
        tag = rec[0].strip().upper()
        if tag == "GROUP":
            flush()
            cur, heads, rows = rec[1].strip(), None, []
        elif tag == "HEADING":
            heads = [h.strip() for h in rec[1:]]
        elif tag == "UNIT" and heads:
            units[cur] = dict(zip(heads, rec[1:]))
        elif tag == "DATA" and heads:
            vals = rec[1:] + [""] * (len(heads) - len(rec) + 1)
            rows.append(vals[: len(heads)])
        elif tag == "<CONT>" and rows:
            for i, v in enumerate(rec[1:]):
                if v:
                    rows[-1][i] += v
    flush()
    return groups


def _num(s):
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")


def ags4_to_boreholes(groups: dict[str, pd.DataFrame], lith_field: str = "auto") -> BoreholeSet:
    """Convert parsed AGS4 groups into a BoreholeSet.

    lith_field: GEOL column used as lithology code: ``GEOL_GEOL`` (stratum), ``GEOL_LEG``
    (legend code), ``GEOL_GEO2`` or ``auto`` (first non-empty of GEOL_GEOL, GEOL_LEG).
    """
    if "LOCA" not in groups or "GEOL" not in groups:
        raise ValueError("AGS4 file must contain LOCA and GEOL groups")
    loca = groups["LOCA"]
    col = pd.DataFrame({
        "hole_id": loca["LOCA_ID"].astype(str),
        "x": _num(loca.get("LOCA_NATE", pd.Series([""] * len(loca)))),
        "y": _num(loca.get("LOCA_NATN", pd.Series([""] * len(loca)))),
        "z": _num(loca.get("LOCA_GL", pd.Series([""] * len(loca)))),
        "depth": _num(loca.get("LOCA_FDEP", pd.Series([""] * len(loca)))),
    })
    # fall back to LOCA_LOCX/LOCY/LOCZ (local grid) when national grid is missing
    for a, b in (("x", "LOCA_LOCX"), ("y", "LOCA_LOCY"), ("z", "LOCA_LOCZ")):
        if b in loca and col[a].isna().all():
            col[a] = _num(loca[b])
    col["azimuth"], col["dip"] = 0.0, 90.0
    if "HDPH" in groups:  # hole depth/orientation records
        h = groups["HDPH"]
        if "HDPH_ORNT" in h and "HDPH_INCL" in h:
            first = h.groupby("LOCA_ID").first()
            col["azimuth"] = col.hole_id.map(_num(first["HDPH_ORNT"])).fillna(0.0)
            col["dip"] = col.hole_id.map(_num(first["HDPH_INCL"])).fillna(90.0)
    geol = groups["GEOL"]
    if lith_field == "auto":
        lith_field = next((f for f in ("GEOL_GEOL", "GEOL_LEG", "GEOL_GEO2")
                           if f in geol and (geol[f].str.strip() != "").any()), "GEOL_DESC")
    iv = pd.DataFrame({
        "hole_id": geol["LOCA_ID"].astype(str),
        "top": _num(geol["GEOL_TOP"]),
        "base": _num(geol["GEOL_BASE"]),
        "lith": geol[lith_field].astype(str),
        "desc": geol.get("GEOL_DESC", pd.Series([""] * len(geol))).astype(str),
    })
    tests = {}
    if "ISPT" in groups:
        s = groups["ISPT"]
        from .insitu import normalise_ags_spt

        tests["SPT"] = normalise_ags_spt(pd.DataFrame({
            "hole_id": s["LOCA_ID"].astype(str), "depth": _num(s["ISPT_TOP"]),
            "value": s.get("ISPT_NVAL", pd.Series([""] * len(s))).replace("", np.nan)}))
    bs = BoreholeSet(collars=col, intervals=iv, tests=tests).clean()
    bs.collars["z_source"] = np.where(bs.collars["z"].notna(), "ags:LOCA_GL", "missing")
    return bs


def load_ags4(src, lith_field: str = "auto") -> BoreholeSet:
    return ags4_to_boreholes(read_ags4(src), lith_field=lith_field)


def write_ags4(path, groups: dict[str, pd.DataFrame], units: dict[str, dict] | None = None) -> Path:
    """Write groups to an AGS4 file (all TYPE = X unless given in df.attrs['types'])."""
    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    for g, df in groups.items():
        heads = list(df.columns)
        u = (units or {}).get(g, {})
        types = df.attrs.get("types", {})
        w.writerow(["GROUP", g])
        w.writerow(["HEADING", *heads])
        w.writerow(["UNIT", *[u.get(h, "") for h in heads]])
        w.writerow(["TYPE", *[types.get(h, "X") for h in heads]])
        for row in df.itertuples(index=False):
            w.writerow(["DATA", *["" if (isinstance(v, float) and np.isnan(v)) else v for v in row]])
        buf.write("\r\n")
    Path(path).write_text(buf.getvalue(), encoding="utf-8")
    return Path(path)
