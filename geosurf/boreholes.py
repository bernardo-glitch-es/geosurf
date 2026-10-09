"""Borehole data: loading from CSV/Excel, column mapping, desurveying and LiDAR draping.

Internal (canonical) tables
---------------------------
collars   : hole_id, x, y, z, depth, azimuth, dip, z_source
survey    : hole_id, depth, azimuth, dip          (optional, for inclined holes)
intervals : hole_id, top, base, lith, desc, unit  (depths along hole, metres)

Dip convention: inclination below horizontal, 90 = vertical. Negative values
(mining convention, -90 = down) are accepted and converted with abs().
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# synonyms for automatic column detection (lower-case, non-alphanumerics stripped)
SYNONYMS: dict[str, list[str]] = {
    "hole_id": ["holeid", "hole", "bhid", "boreholeid", "borehole", "bh", "id", "locaid", "sondeo",
                "pointid", "name", "sondagem", "furo", "bhname", "location"],
    "x": ["x", "east", "easting", "e", "xcoord", "locanate", "utmx", "coordx", "este"],
    "y": ["y", "north", "northing", "n", "ycoord", "locanatn", "utmy", "coordy", "norte"],
    "z": ["z", "elev", "elevation", "rl", "gl", "groundlevel", "zcoord", "locagl", "cota", "collarz",
          "cotaboca"],
    "depth": ["depth", "totaldepth", "td", "eoh", "finaldepth", "locafdep", "maxdepth",
              "profundidad", "profundidade", "length"],
    "azimuth": ["azimuth", "azi", "az", "bearing", "azimut", "azimute"],
    "dip": ["dip", "inclination", "incl", "plunge", "inclinacion", "inclinacao"],
    "top": ["from", "top", "depthfrom", "fromdepth", "geoltop", "desde", "de", "topdepth"],
    "base": ["to", "base", "bottom", "depthto", "todepth", "geolbase", "hasta", "ate", "basedepth"],
    "lith": ["lith", "litho", "lithology", "unit", "code", "geolleg", "geolgeol", "formation",
             "geology", "litologia", "rock", "stratum", "layer", "legend"],
    "desc": ["desc", "description", "geoldesc", "descripcion", "descricao", "comments", "remarks"],
    "survey_depth": ["depth", "at", "surveydepth", "md", "measureddepth"],
}

ROLE_COLUMNS = {
    "collars": ["hole_id", "x", "y", "z", "depth", "azimuth", "dip"],
    "intervals": ["hole_id", "top", "base", "lith", "desc"],
    "survey": ["hole_id", "survey_depth", "azimuth", "dip"],
    "flat": ["hole_id", "x", "y", "z", "top", "base", "lith", "desc"],
}
REQUIRED = {
    "collars": ["hole_id", "x", "y"],
    "intervals": ["hole_id", "top", "base", "lith"],
    "survey": ["hole_id", "survey_depth", "azimuth", "dip"],
    "flat": ["hole_id", "x", "y", "top", "base", "lith"],
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def guess_columns(df: pd.DataFrame, role: str) -> dict[str, str | None]:
    """Map canonical fields -> source column names using synonyms (first match wins)."""
    norm = {_norm(c): c for c in df.columns}
    used: set[str] = set()
    out: dict[str, str | None] = {}
    for key in ROLE_COLUMNS[role]:
        out[key] = None
        for syn in SYNONYMS[key]:
            c = norm.get(syn)
            if c is not None and c not in used:
                out[key] = c
                used.add(c)
                break
    return out


def read_table(path_or_buffer, sheet=None) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Read CSV/TXT/XLSX. Excel without ``sheet`` returns a dict of all sheets."""
    name = getattr(path_or_buffer, "name", str(path_or_buffer))
    ext = Path(name).suffix.lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path_or_buffer, sheet_name=sheet)
    return pd.read_csv(path_or_buffer, sep=None, engine="python")


def _merge_tests(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = pd.concat([out[k], v], ignore_index=True) if k in out else v
    return out


def _apply_map(df: pd.DataFrame, mapping: dict[str, str | None], role: str) -> pd.DataFrame:
    missing = [k for k in REQUIRED[role] if not mapping.get(k)]
    if missing:
        raise ValueError(f"{role}: required columns not mapped: {missing}")
    out = pd.DataFrame({k: df[v].values for k, v in mapping.items() if v})
    out["hole_id"] = out["hole_id"].astype(str).str.strip()
    return out


@dataclass
class BoreholeSet:
    collars: pd.DataFrame
    intervals: pd.DataFrame
    survey: pd.DataFrame | None = None
    tests: dict[str, pd.DataFrame] = field(default_factory=dict)  # e.g. SPT from AGS
    crs: str | None = None

    # ----------------------------------------------------------------- builders
    @classmethod
    def from_tables(cls, collars: pd.DataFrame | None, intervals: pd.DataFrame,
                    survey: pd.DataFrame | None = None, collar_map=None, interval_map=None,
                    survey_map=None, flat: bool = False) -> "BoreholeSet":
        if flat:
            m = interval_map or guess_columns(intervals, "flat")
            iv = _apply_map(intervals, m, "flat")
            agg = {"x": "first", "y": "first", "base": "max"}
            if "z" in iv:
                agg["z"] = "first"
            col = iv.groupby("hole_id", sort=False).agg(agg).reset_index().rename(columns={"base": "depth"})
            iv = iv.drop(columns=[c for c in ("x", "y", "z") if c in iv])
        else:
            col = _apply_map(collars, collar_map or guess_columns(collars, "collars"), "collars")
            iv = _apply_map(intervals, interval_map or guess_columns(intervals, "intervals"), "intervals")
        sv = None
        if survey is not None and len(survey):
            sv = _apply_map(survey, survey_map or guess_columns(survey, "survey"), "survey")
            sv = sv.rename(columns={"survey_depth": "depth"})
        return cls(collars=col, intervals=iv, survey=sv).clean()

    def clean(self) -> "BoreholeSet":
        c = self.collars.copy()
        for k in ("x", "y", "z", "depth", "azimuth", "dip"):
            if k not in c:
                c[k] = np.nan
            c[k] = pd.to_numeric(c[k], errors="coerce")
        if "z_source" not in c:
            c["z_source"] = np.where(c["z"].notna(), "file", "missing")
        c["azimuth"] = c["azimuth"].fillna(0.0)
        c["dip"] = c["dip"].abs().fillna(90.0)
        c = c.dropna(subset=["x", "y"]).drop_duplicates("hole_id")
        iv = self.intervals.copy()
        for k in ("top", "base"):
            iv[k] = pd.to_numeric(iv[k], errors="coerce")
        iv = iv.dropna(subset=["top", "base"])
        iv["lith"] = iv["lith"].astype(str).str.strip()
        if "desc" not in iv:
            iv["desc"] = ""
        if "unit" not in iv:
            iv["unit"] = iv["lith"]
        iv = iv[iv["hole_id"].isin(c["hole_id"])].sort_values(["hole_id", "top"]).reset_index(drop=True)
        # total depth fallback = deepest interval
        td = iv.groupby("hole_id")["base"].max()
        c["depth"] = c["depth"].fillna(c["hole_id"].map(td))
        sv = self.survey
        if sv is not None:
            sv = sv.copy()
            for k in ("depth", "azimuth", "dip"):
                sv[k] = pd.to_numeric(sv[k], errors="coerce")
            sv["dip"] = sv["dip"].abs()
            sv = sv.dropna().sort_values(["hole_id", "depth"]).reset_index(drop=True)
        self.collars, self.intervals, self.survey = c.reset_index(drop=True), iv, sv
        return self

    def merge(self, other: "BoreholeSet") -> "BoreholeSet":
        sv = [s for s in (self.survey, other.survey) if s is not None]
        out = BoreholeSet(
            collars=pd.concat([self.collars, other.collars[~other.collars.hole_id.isin(self.collars.hole_id)]],
                              ignore_index=True),
            intervals=pd.concat([self.intervals, other.intervals[~other.intervals.hole_id.isin(self.collars.hole_id)]],
                                ignore_index=True),
            survey=pd.concat(sv, ignore_index=True) if sv else None,
            tests=_merge_tests(self.tests, other.tests), crs=self.crs or other.crs)
        return out.clean()

    # ----------------------------------------------------------------- info
    @property
    def hole_ids(self) -> list[str]:
        return self.collars["hole_id"].tolist()

    @property
    def lith_codes(self) -> list[str]:
        return self.intervals["lith"].drop_duplicates().tolist()

    def summary(self) -> pd.DataFrame:
        g = self.intervals.groupby("hole_id")
        s = self.collars.set_index("hole_id")[["x", "y", "z", "z_source", "depth", "dip"]].copy()
        s["n_intervals"] = g.size()
        s["units"] = g["unit"].apply(lambda u: " > ".join(pd.unique(u)))
        return s.reset_index()

    def qa(self) -> pd.DataFrame:
        """Interval consistency checks: gaps, overlaps, negative thickness, beyond TD."""
        rows = []
        td = self.collars.set_index("hole_id")["depth"]
        for hid, g in self.intervals.groupby("hole_id"):
            g = g.sort_values("top")
            for _, r in g[g.base <= g.top].iterrows():
                rows.append((hid, "non-positive thickness", r.top, r.base))
            prev = g.base.values[:-1]
            nxt = g.top.values[1:]
            for a, b in zip(prev, nxt):
                if b - a > 0.01:
                    rows.append((hid, "gap", a, b))
                elif a - b > 0.01:
                    rows.append((hid, "overlap", b, a))
            if g.top.min() > 0.01:
                rows.append((hid, "first interval not at 0 m", 0.0, g.top.min()))
            if hid in td and np.isfinite(td[hid]) and g.base.max() > td[hid] + 0.01:
                rows.append((hid, "interval below total depth", td[hid], g.base.max()))
        for _, r in self.collars[self.collars.z.isna()].iterrows():
            rows.append((r.hole_id, "missing collar elevation", np.nan, np.nan))
        return pd.DataFrame(rows, columns=["hole_id", "issue", "from", "to"])

    # ----------------------------------------------------------------- geometry
    def drape(self, terrain, mode: str = "missing", tolerance: float = 1.0) -> pd.DataFrame:
        """Sample the LiDAR/DEM at collars.

        mode: ``missing`` -> fill only missing z; ``all`` -> replace every z with the terrain.
        Returns a table comparing surveyed vs terrain elevation (dz = z - z_terrain).
        """
        c = self.collars
        zt = terrain.sample(c["x"].values, c["y"].values)
        c["z_terrain"] = zt
        c["dz_terrain"] = c["z"] - zt
        fill = c["z"].isna() if mode == "missing" else pd.Series(True, index=c.index)
        fill &= np.isfinite(zt)
        c.loc[fill, "z"] = zt[fill.values]
        c.loc[fill, "z_source"] = "lidar"
        rep = c[["hole_id", "x", "y", "z", "z_source", "z_terrain", "dz_terrain"]].copy()
        rep["flag"] = np.where(rep["dz_terrain"].abs() > tolerance, f"|dz| > {tolerance} m", "")
        return rep

    def _stations(self, hid: str) -> np.ndarray:
        """Survey stations (depth, azimuth, dip) for a hole, including collar and TD."""
        c = self.collars.set_index("hole_id").loc[hid]
        td = float(c["depth"]) if np.isfinite(c["depth"]) else 0.0
        if self.survey is not None and (self.survey.hole_id == hid).any():
            s = self.survey[self.survey.hole_id == hid][["depth", "azimuth", "dip"]].to_numpy(float)
            if s[0, 0] > 0:
                s = np.vstack([[0.0, s[0, 1], s[0, 2]], s])
        else:
            s = np.array([[0.0, c["azimuth"], c["dip"]]])
        if s[-1, 0] < td:
            s = np.vstack([s, [td, s[-1, 1], s[-1, 2]]])
        return s

    def desurvey(self, hid: str, depths) -> np.ndarray:
        """XYZ of along-hole depths (balanced tangential on survey stations)."""
        depths = np.atleast_1d(np.asarray(depths, float))
        c = self.collars.set_index("hole_id").loc[hid]
        s = self._stations(hid)
        if len(s) == 1:
            s = np.vstack([s, [max(depths.max(), 1.0), s[0, 1], s[0, 2]]])
        az = np.radians(s[:, 1])
        dp = np.radians(s[:, 2])
        # unit direction vectors (dip = inclination below horizontal)
        d = np.c_[np.cos(dp) * np.sin(az), np.cos(dp) * np.cos(az), -np.sin(dp)]
        seg = np.diff(s[:, 0])
        step = 0.5 * (d[:-1] + d[1:]) * seg[:, None]
        nodes = np.vstack([[0, 0, 0], np.cumsum(step, axis=0)])
        md = s[:, 0]
        off = np.c_[[np.interp(depths, md, nodes[:, k]) for k in range(3)]].T
        # extrapolate beyond last station along last direction
        beyond = depths > md[-1]
        if beyond.any():
            off[beyond] = nodes[-1] + (depths[beyond] - md[-1])[:, None] * d[-1]
        z0 = c["z"] if np.isfinite(c["z"]) else 0.0
        return off + np.array([c["x"], c["y"], z0])

    def traces(self) -> pd.DataFrame:
        """Interval table with XYZ of top and base (for plotting/export)."""
        rows = []
        for hid, g in self.intervals.groupby("hole_id", sort=False):
            pt = self.desurvey(hid, g["top"].values)
            pb = self.desurvey(hid, g["base"].values)
            gg = g.copy()
            gg[["xt", "yt", "zt"]] = pt
            gg[["xb", "yb", "zb"]] = pb
            rows.append(gg)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def to_csv(self, folder) -> list[Path]:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        out = [folder / "collars.csv", folder / "intervals.csv"]
        self.collars.to_csv(out[0], index=False)
        self.intervals.to_csv(out[1], index=False)
        if self.survey is not None:
            out.append(folder / "survey.csv")
            self.survey.to_csv(out[-1], index=False)
        return out
