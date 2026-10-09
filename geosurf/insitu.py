"""In-situ and laboratory test data attached to boreholes (SPT, lab results, ...).

Tests live in ``BoreholeSet.tests`` as {name: DataFrame}. Every table has at least
``hole_id, depth`` (along-hole, m); ``depth_to`` is optional. SPT tables carry ``n`` (blow
count, refusal -> ``n_cap``) and ``refusal`` (bool).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .boreholes import _norm

N_CAP = 50.0  # value given to refusal ("R", "50R", ">50") for plotting/statistics


def parse_n(v) -> tuple[float, bool]:
    """SPT text -> (N, refusal). '17' -> (17, False); 'R', '50R', '>50' -> (50, True)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan, False
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v), float(v) >= N_CAP
    s = str(v).strip().upper().replace(",", ".")
    if not s:
        return np.nan, False
    if "R" in s or s.startswith(">"):
        return N_CAP, True
    m = re.search(r"[\d.]+", s)
    return (float(m.group()), False) if m else (np.nan, False)


def _find(df, *names):
    norm = {_norm(c): c for c in df.columns}
    for n in names:
        if _norm(n) in norm:
            return norm[_norm(n)]
    return None


def spt_from_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generic SPT / sample table (e.g. workbook sheet 'SPT_samples')."""
    hid = _find(df, "HoleID", "hole_id", "LOCA_ID", "Sondeo")
    top = _find(df, "From", "depth", "Top", "ISPT_TOP", "Desde")
    bot = _find(df, "To", "depth_to", "Base", "Hasta")
    n = _find(df, "N_SPT", "N", "NSPT", "ISPT_NVAL", "value", "Nspt")
    if hid is None or top is None or n is None:
        raise ValueError("SPT table needs hole id, depth and N columns")
    parsed = [parse_n(v) for v in df[n]]
    out = pd.DataFrame({"hole_id": df[hid].astype(str).str.strip(),
                        "depth": pd.to_numeric(df[top], errors="coerce"),
                        "depth_to": pd.to_numeric(df[bot], errors="coerce") if bot else np.nan,
                        "n": [p[0] for p in parsed], "refusal": [p[1] for p in parsed],
                        "n_text": df[n].astype(str)})
    for extra in ("Test", "Blows", "Remarks"):
        c = _find(df, extra)
        if c:
            out[extra.lower()] = df[c].values
    return out.dropna(subset=["depth"]).reset_index(drop=True)


def lab_from_table(df: pd.DataFrame) -> pd.DataFrame:
    """Lab results table (one row per sample, any number of result columns)."""
    hid = _find(df, "HoleID", "hole_id", "LOCA_ID", "Sondeo")
    top = _find(df, "From", "depth", "Top", "SAMP_TOP")
    bot = _find(df, "To", "depth_to", "Base", "SAMP_BASE")
    if hid is None or top is None:
        raise ValueError("Lab table needs hole id and sample depth columns")
    out = df.copy()
    out.insert(0, "hole_id", df[hid].astype(str).str.strip())
    d0 = pd.to_numeric(df[top], errors="coerce")
    d1 = pd.to_numeric(df[bot], errors="coerce") if bot else d0
    out.insert(1, "depth", d0)
    out.insert(2, "depth_to", d1)
    out.insert(3, "depth_mid", (d0 + d1.fillna(d0)) / 2)
    out = out.drop(columns=[c for c in {hid, top, bot} if c and c not in ("hole_id",)], errors="ignore")
    return out.dropna(subset=["depth"]).reset_index(drop=True)


def tests_from_workbook(sheets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Pick up known test sheets from a geosurf workbook (missing sheets are ignored)."""
    out = {}
    for name in ("SPT_samples", "SPT"):
        if name in sheets:
            try:
                out["SPT"] = spt_from_table(sheets[name])
                break
            except ValueError:
                pass
    for name in ("Lab_tests", "Lab", "Laboratory"):
        if name in sheets:
            try:
                out["LAB"] = lab_from_table(sheets[name])
                break
            except ValueError:
                pass
    return out


def normalise_ags_spt(df: pd.DataFrame) -> pd.DataFrame:
    """AGS ISPT (hole_id, depth, value) -> SPT schema."""
    parsed = [parse_n(v) for v in df["value"]]
    return pd.DataFrame({"hole_id": df["hole_id"], "depth": df["depth"], "depth_to": np.nan,
                         "n": [p[0] for p in parsed], "refusal": [p[1] for p in parsed],
                         "n_text": df["value"].astype(str)})


def with_xyz(bs, df: pd.DataFrame, depth_col: str = "depth") -> pd.DataFrame:
    """Add x, y, z (desurveyed) and the modelled/logged unit at each test depth."""
    df = df[df["hole_id"].isin(bs.hole_ids)].copy()
    if df.empty:
        return df.assign(x=[], y=[], z=[], unit=[])
    xyz = np.vstack([bs.desurvey(h, [d])[0] for h, d in zip(df["hole_id"], df[depth_col])])
    df["x"], df["y"], df["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    iv = bs.intervals
    units = []
    for h, d in zip(df["hole_id"], df[depth_col]):
        m = iv[(iv.hole_id == h) & (iv.top <= d) & (iv.base > d)]
        units.append(m["unit"].iloc[0] if len(m) else "")
    df["unit"] = units
    return df


def spt_by_unit(bs) -> pd.DataFrame:
    """SPT N statistics per logged unit (refusals counted at N_CAP)."""
    spt = bs.tests.get("SPT")
    if spt is None or spt.empty:
        return pd.DataFrame()
    d = with_xyz(bs, spt)
    g = d.groupby("unit")
    return pd.DataFrame({"n_tests": g.size(), "n_refusal": g["refusal"].sum(),
                         "N_min": g["n"].min(), "N_median": g["n"].median(), "N_max": g["n"].max()}).reset_index()
