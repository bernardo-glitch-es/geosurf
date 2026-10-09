"""Borehole + LiDAR -> GemPy implicit model -> geological surfaces.

Workflow
--------
1. ``Stratigraphy``: ordered units (youngest/top -> oldest/bottom) split into
   structural groups (series). The last unit is the basement.
2. ``extract_contacts``: each borehole's mapped intervals -> interface points.
   GemPy convention: the points of element *U* lie on the BASE of unit *U*.
3. ``estimate_orientations``: plane fits through contact points (global + local).
4. ``build_model`` / ``compute``: GemPy model with LiDAR topography.
5. ``extract_surfaces``: per-interface elevation grids from the scalar field,
   with erosion and topographic clipping -> ready for CAD/GIS.
"""
from __future__ import annotations

import contextlib
import io
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .boreholes import BoreholeSet
from .raster import Raster

log = logging.getLogger(__name__)

DEFAULT_COLORS = ["#c8a165", "#e9d36c", "#8fbf6a", "#d98c5f", "#7fa7c9", "#b48ec9", "#e07b91",
                  "#6cc5b0", "#a3a3a3", "#c2b280", "#5f8fbf", "#d4a373", "#9c6644", "#4d6a6d"]
IGNORE = "(ignore)"


# =============================================================================== stratigraphy
@dataclass
class Unit:
    name: str
    group: str = "Strata"
    color: str | None = None


@dataclass
class Stratigraphy:
    """Units ordered top -> bottom. The last unit is the basement (no surface of its own)."""
    units: list[Unit]
    relations: dict[str, str] = field(default_factory=dict)  # group -> "erode" | "onlap"

    @classmethod
    def from_list(cls, names: list[str], groups: list[str] | None = None,
                  colors: list[str] | None = None, relations: dict | None = None,
                  mode: str = "independent") -> "Stratigraphy":
        """mode ``independent``: one GemPy series per interface (each surface interpolated on
        its own, stacked by erosion - best for soils / weathering profiles with variable
        thickness). ``conformable``: all interfaces share one scalar field (layer-cake)."""
        if groups is None:
            groups = list(names) if mode == "independent" else ["Strata"] * len(names)
        colors = colors or [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(len(names))]
        return cls([Unit(n, g, c) for n, g, c in zip(names, groups, colors)], relations or {})

    @property
    def names(self) -> list[str]:
        return [u.name for u in self.units]

    @property
    def basement(self) -> str:
        return self.units[-1].name

    @property
    def surface_units(self) -> list[Unit]:
        """Units that own an interface (their base) = all but the basement."""
        return self.units[:-1]

    @property
    def groups(self) -> list[str]:
        out: list[str] = []
        for u in self.surface_units:
            if u.group not in out:
                out.append(u.group)
        return out

    def rank(self, name: str) -> int:
        return self.names.index(name)

    def color(self, name: str) -> str:
        for i, u in enumerate(self.units):
            if u.name == name:
                return u.color or DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        return "#999999"

    def to_dict(self) -> dict:
        return {"units": [u.__dict__ for u in self.units], "relations": self.relations}

    @classmethod
    def from_dict(cls, d: dict) -> "Stratigraphy":
        return cls([Unit(**u) for u in d["units"]], d.get("relations", {}))


def suggest_stratigraphy(bs: BoreholeSet) -> list[str]:
    """Order lithology codes by their mean relative position (normalised depth) in holes."""
    iv = bs.intervals[bs.intervals["unit"] != IGNORE].copy()
    mid = 0.5 * (iv.top + iv.base)
    td = iv.groupby("hole_id")["base"].transform("max").replace(0, np.nan)
    iv["pos"] = mid / td
    # order index within each hole is more robust than depth for variable thicknesses
    iv["seq"] = iv.groupby("hole_id").cumcount() / iv.groupby("hole_id")["top"].transform("count")
    score = iv.groupby("unit")[["pos", "seq"]].mean().mean(axis=1)
    return score.sort_values().index.tolist()


# =============================================================================== contacts
def extract_contacts(bs: BoreholeSet, strat: Stratigraphy, pinchout: str = "zero",
                     offset: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Borehole intervals -> interface points.

    pinchout: how to treat units missing between two logged units (and units missing at
      the top of a hole): ``zero`` adds zero-thickness points (stacked ``offset`` m apart)
      so the model thins the unit to nothing there; ``skip`` adds nothing.
    Returns (points[x,y,z,unit,hole_id,kind], issues[hole_id,issue,depth]).
    """
    names = strat.names
    rank = {n: i for i, n in enumerate(names)}
    pts, issues = [], []
    iv_all = bs.intervals
    collars = bs.collars.set_index("hole_id")
    for hid, g in iv_all.groupby("hole_id", sort=False):
        if not np.isfinite(collars.loc[hid, "z"]):
            issues.append((hid, "no collar elevation - hole skipped", np.nan))
            continue
        g = g[g["unit"].isin(names)].sort_values("top")
        unknown = set(iv_all.loc[iv_all.hole_id == hid, "unit"]) - set(names) - {IGNORE}
        for u in unknown:
            issues.append((hid, f"unit '{u}' not in stratigraphy - ignored", np.nan))
        if g.empty:
            continue
        # merge consecutive intervals of the same unit
        seq = []
        for r in g.itertuples():
            if seq and seq[-1][0] == r.unit:
                seq[-1][2] = max(seq[-1][2], r.base)
            else:
                seq.append([r.unit, r.top, r.base])

        def add(unit, depth, kind, dz=0.0):
            x, y, z = bs.desurvey(hid, [depth])[0]
            pts.append((x, y, z + dz, unit, hid, kind))

        # units absent at the top of the hole (eroded / not deposited)
        if pinchout == "zero":
            r0 = rank[seq[0][0]]
            for k, u in enumerate(names[:r0]):
                add(u, seq[0][1], "absent-top", dz=offset * (r0 - k))
        for (u, top, base), (l, ltop, _) in zip(seq[:-1], seq[1:]):
            ru, rl = rank[u], rank[l]
            if rl <= ru:
                issues.append((hid, f"out of sequence: {u} over {l}", base))
                continue
            depth = ltop if ltop > base else base  # gaps/ignored -> top of lower unit
            add(u, depth, "contact")
            if pinchout == "zero":
                for k, m in enumerate(names[ru + 1: rl]):
                    add(m, depth, "pinch-out", dz=-offset * (k + 1))
    pts_df = pd.DataFrame(pts, columns=["x", "y", "z", "unit", "hole_id", "kind"])
    return pts_df, pd.DataFrame(issues, columns=["hole_id", "issue", "depth"])


# =============================================================================== orientations
def _plane_fit(p: np.ndarray) -> tuple[np.ndarray, float]:
    c = p.mean(0)
    _, s, vt = np.linalg.svd(p - c)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    planarity = s[-1] / (s[0] + 1e-12)
    return n, planarity


def pole_to_dip(n) -> tuple[float, float]:
    n = np.asarray(n, float)
    dip = np.degrees(np.arccos(np.clip(abs(n[2]) / np.linalg.norm(n), -1, 1)))
    azi = (np.degrees(np.arctan2(n[0], n[1])) + 360) % 360
    return dip, azi


def dip_to_pole(dip, azimuth) -> np.ndarray:
    d, a = np.radians(dip), np.radians(azimuth)
    return np.c_[np.sin(d) * np.sin(a), np.sin(d) * np.cos(a), np.cos(d)]


ORI_SYNONYMS = {
    "x": ["x", "east", "easting", "e", "este", "utmx"],
    "y": ["y", "north", "northing", "n", "norte", "utmy"],
    "z": ["z", "elev", "elevation", "rl", "cota", "zcoord"],
    "dip": ["dip", "buzamiento", "inclination", "inclinacion", "dipangle"],
    "azimuth": ["azimuth", "dipdirection", "dipdir", "dipazimuth", "dd", "azi", "az",
                "direccionbuzamiento", "direccion", "azimut"],
    "unit": ["unit", "surface", "formation", "unidad", "element", "lith", "contact", "name"],
}


def parse_orientations(df: pd.DataFrame, strat: "Stratigraphy | None" = None) -> pd.DataFrame:
    """Normalise a structural-measurement table to x, y, z, G_x, G_y, G_z, unit.

    Accepts either dip + dip-direction (azimuth of dip, degrees) or pole vector columns
    G_x/G_y/G_z. Column names are matched loosely (EN/ES). Raises ValueError with a clear
    message when required columns are missing or no row matches a modelled surface."""
    import re as _re

    norm = {_re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    col = {k: next((norm[s] for s in syn if s in norm), None) for k, syn in ORI_SYNONYMS.items()}
    gcols = [norm.get(g) for g in ("gx", "gy", "gz")]
    missing = [k for k in ("x", "y", "z", "unit") if col[k] is None]
    has_g = all(gcols)
    if not has_g and (col["dip"] is None or col["azimuth"] is None):
        missing += [k for k in ("dip", "azimuth") if col[k] is None]
    if missing:
        raise ValueError(f"Orientation table: missing column(s) {missing}. Found: {list(df.columns)}. "
                         "Expected x, y, z, dip, azimuth (dip direction), unit.")
    out = pd.DataFrame({k: pd.to_numeric(df[col[k]], errors="coerce") for k in ("x", "y", "z")})
    out["unit"] = df[col["unit"]].astype(str).str.strip()
    if has_g:
        for g, c in zip(("G_x", "G_y", "G_z"), gcols):
            out[g] = pd.to_numeric(df[c], errors="coerce")
    else:
        dip = pd.to_numeric(df[col["dip"]], errors="coerce").abs()
        azi = pd.to_numeric(df[col["azimuth"]], errors="coerce") % 360
        bad = (dip > 90)
        if bad.any():
            raise ValueError(f"Orientation table: {int(bad.sum())} dip value(s) > 90°")
        g = dip_to_pole(dip.values, azi.values)
        out["G_x"], out["G_y"], out["G_z"] = g[:, 0], g[:, 1], g[:, 2]
    out = out.dropna().reset_index(drop=True)
    if strat is not None:
        valid = {u.name for u in strat.surface_units}
        unknown = sorted(set(out["unit"]) - valid)
        out = out[out["unit"].isin(valid)].reset_index(drop=True)
        if out.empty:
            raise ValueError(f"Orientation table: no rows match a modelled surface {sorted(valid)} "
                             f"(found {unknown}). Use the unit whose BASE the measurement describes.")
        out.attrs["ignored_units"] = unknown
    if out.empty:
        raise ValueError("Orientation table: no valid rows")
    return out


def estimate_orientations(points: pd.DataFrame, strat: Stratigraphy, local_k: int = 0,
                          max_dip: float = 60.0) -> pd.DataFrame:
    """One plane-fit orientation per surface at the centroid (+ optional local fits with k
    nearest contacts). Surfaces with < 3 non-collinear points get a horizontal orientation."""
    rows = []
    contacts = points[points.kind == "contact"]
    for u in strat.surface_units:
        p = contacts.loc[contacts.unit == u.name, ["x", "y", "z"]].to_numpy(float)
        if len(p) == 0:
            p = points.loc[points.unit == u.name, ["x", "y", "z"]].to_numpy(float)
        if len(p) == 0:
            continue
        c = p.mean(0)
        n = np.array([0, 0, 1.0])
        if len(p) >= 3:
            xy_spread = np.linalg.svd(p[:, :2] - p[:, :2].mean(0), compute_uv=False)
            if xy_spread[-1] > 1e-6 * (xy_spread[0] + 1e-9):  # not collinear in plan
                n, _ = _plane_fit(p)
        if pole_to_dip(n)[0] > max_dip:
            n = np.array([0, 0, 1.0])
        rows.append((*c, *n, u.name, "global-fit"))
        if local_k >= 3 and len(p) > local_k:
            from scipy.spatial import cKDTree

            tree = cKDTree(p[:, :2])
            for i in range(len(p)):
                _, idx = tree.query(p[i, :2], k=local_k)
                q = p[idx]
                sv = np.linalg.svd(q[:, :2] - q[:, :2].mean(0), compute_uv=False)
                if sv[-1] < 0.15 * sv[0]:
                    continue  # neighbours nearly collinear -> unstable
                nl, flat = _plane_fit(q)
                if pole_to_dip(nl)[0] <= max_dip and flat < 0.2:
                    rows.append((*q.mean(0), *nl, u.name, f"local-k{local_k}"))
    return pd.DataFrame(rows, columns=["x", "y", "z", "G_x", "G_y", "G_z", "unit", "source"])


# =============================================================================== model
@dataclass
class ModelConfig:
    extent: list[float] | None = None          # [xmin,xmax,ymin,ymax,zmin,zmax]
    resolution: tuple[int, int, int] = (50, 50, 60)
    pad_xy: float = 0.0
    pad_z: float = 10.0
    nugget: float = 2e-5                       # surface-point nugget (smoothing, model units)
    pinchout: str = "zero"
    pinchout_offset: float = 0.25
    local_orientations_k: int = 0
    topo_max_points: int = 40_000
    trend_alpha: float = 1.0                   # 0 = model elevations; 1 = model depth below ground
    trend_smooth: float = 0.0                  # gaussian smoothing (m) of the terrain used as trend
    project_name: str = "geosurf"


def auto_extent(bs: BoreholeSet, terrain: Raster | None, points: pd.DataFrame, cfg: ModelConfig):
    if terrain is not None:
        xmin, xmax, ymin, ymax = terrain.bounds
        ztop = np.nanmax(terrain.z)
    else:
        xmin, xmax = bs.collars.x.min(), bs.collars.x.max()
        ymin, ymax = bs.collars.y.min(), bs.collars.y.max()
        ztop = bs.collars.z.max()
    xmin, xmax, ymin, ymax = xmin - cfg.pad_xy, xmax + cfg.pad_xy, ymin - cfg.pad_xy, ymax + cfg.pad_xy
    bottoms = [bs.desurvey(h, [d])[0][2] for h, d in zip(bs.collars.hole_id, bs.collars.depth)
               if np.isfinite(d)]
    zbot = min([points.z.min()] + bottoms) if len(points) else min(bottoms)
    return [float(xmin), float(xmax), float(ymin), float(ymax),
            float(np.floor(zbot - cfg.pad_z)), float(np.ceil(ztop + cfg.pad_z))]


@dataclass
class ModelResult:
    strat: Stratigraphy
    extent: list[float]
    resolution: tuple[int, int, int]
    points: pd.DataFrame
    orientations: pd.DataFrame
    issues: pd.DataFrame
    geo_model: object
    solutions: object
    terrain: Raster | None
    surfaces: dict[str, Raster] = field(default_factory=dict)          # unclipped interfaces
    surfaces_clipped: dict[str, Raster] = field(default_factory=dict)  # eroded + clipped to DTM
    log: str = ""
    trend: Raster | None = None      # alpha * terrain, subtracted before modelling
    config: "ModelConfig | None" = None

    def to_model_z(self, x, y, z):
        """Real elevation -> GemPy model coordinate (removes terrain trend)."""
        if self.trend is None:
            return np.asarray(z, float)
        t = self.trend.sample(x, y)
        return np.asarray(z, float) - np.nan_to_num(t, nan=float(np.nanmean(self.trend.z)))

    def lith_at(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.array(xyz, float)
        xyz[:, 2] = self.to_model_z(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        return lith_at(self.geo_model, xyz)

    @property
    def unit_names(self) -> list[str]:
        """Model lithology ids (1-based) -> unit name."""
        names = [e.name for e in self.geo_model.structural_frame.structural_elements]
        names[-1] = self.strat.basement  # GemPy appends an auto-generated basement element
        return names


@contextlib.contextmanager
def _quiet():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def build_model(points: pd.DataFrame, orientations: pd.DataFrame, strat: Stratigraphy,
                extent, cfg: ModelConfig, terrain: Raster | None = None):
    import gempy as gp
    from gempy.core.data import (OrientationsTable, StructuralElement, StructuralFrame,
                                 StructuralGroup, SurfacePointsTable)
    from gempy_engine.core.data.stack_relation_type import StackRelationType

    rel = {"erode": StackRelationType.ERODE, "onlap": StackRelationType.ONLAP}
    groups = []
    for gname in strat.groups:
        els = []
        for u in [u for u in strat.surface_units if u.group == gname]:
            p = points[points.unit == u.name]
            o = orientations[orientations.unit == u.name]
            if len(p) == 0:
                log.warning("unit %s has no interface points - skipped", u.name)
                continue
            if len(o) == 0:  # every element needs to be in a group with >=1 orientation
                o = pd.DataFrame([[p.x.mean(), p.y.mean(), p.z.mean(), 0, 0, 1.0]],
                                 columns=["x", "y", "z", "G_x", "G_y", "G_z"])
            sp = SurfacePointsTable.from_arrays(p.x.values, p.y.values, p.z.values, names=u.name,
                                                nugget=np.full(len(p), cfg.nugget))
            ot = OrientationsTable.from_arrays(o.x.values, o.y.values, o.z.values, o.G_x.values,
                                               o.G_y.values, o.G_z.values, names=u.name)
            els.append(StructuralElement(name=u.name, surface_points=sp, orientations=ot,
                                         color=strat.color(u.name)))
        if els:
            groups.append(StructuralGroup(name=gname, elements=els,
                                          structural_relation=rel[strat.relations.get(gname, "erode")]))
    frame = StructuralFrame(structural_groups=groups, color_gen=gp.data.ColorsGenerator())
    with _quiet():
        model = gp.create_geomodel(project_name=cfg.project_name, extent=list(extent),
                                   resolution=list(cfg.resolution), refinement=1, structural_frame=frame)
    if terrain is not None:
        xyz = terrain.xyz()
        if len(xyz) > cfg.topo_max_points:
            step = int(np.ceil(np.sqrt(len(xyz) / cfg.topo_max_points)))
            sub = terrain.like(terrain.z[::step, ::step])
            sub.dx, sub.dy = terrain.dx * step, terrain.dy * step
            xyz = sub.xyz()
        xmin, xmax, ymin, ymax, zmin, zmax = extent
        k = (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) & (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax)
        xyz = xyz[k]
        xyz[:, 2] = np.clip(xyz[:, 2], zmin, zmax)
        with _quiet():
            gp.set_topography_from_arrays(model.grid, xyz)
    return model


def compute(model):
    import gempy as gp

    with _quiet() as buf:
        sol = gp.compute_model(model)
    return sol, buf.getvalue()


def lith_at(model, xyz: np.ndarray) -> np.ndarray:
    """Lithology id (1..n, top->bottom order of model elements) at arbitrary points."""
    import gempy as gp

    with _quiet():
        ids = gp.compute_model_at(model, np.asarray(xyz, float))
    return np.rint(ids).astype(int)


# =============================================================================== surfaces
def _crossing(F: np.ndarray, z: np.ndarray, v: float) -> np.ndarray:
    """Elevation where scalar field column F(z) crosses value v (topmost crossing).
    F: (nx, ny, nz) increasing upward. Returns (nx, ny); +inf if surface is above the box,
    -inf if below."""
    d = F - v
    s = np.sign(d)
    cross = (s[..., :-1] * s[..., 1:]) <= 0
    any_c = cross.any(-1)
    # topmost crossing index
    idx = cross.shape[-1] - 1 - np.argmax(cross[..., ::-1], axis=-1)
    i0 = np.take_along_axis(d, idx[..., None], -1)[..., 0]
    i1 = np.take_along_axis(d, (idx + 1)[..., None], -1)[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(i1 != i0, i0 / (i0 - i1), 0.0)
    zc = z[idx] + t * (z[idx + 1] - z[idx])
    out = np.where(any_c, zc, np.nan)
    allpos = (d > 0).all(-1)   # whole column above v -> surface below box
    allneg = (d < 0).all(-1)   # whole column below v -> surface above box
    out[allpos & ~any_c] = -np.inf
    out[allneg & ~any_c] = np.inf
    return out


def extract_surfaces(result: ModelResult, cell: float | None = None) -> ModelResult:
    """Fill ``result.surfaces`` and ``result.surfaces_clipped`` (dict unit -> Raster).

    Each raster is the elevation of the BASE of the named unit. ``cell`` resamples to a
    finer output grid (defaults to the terrain cell size, if any)."""
    model, sol, strat = result.geo_model, result.solutions, result.strat
    nx, ny, nz = result.resolution
    xmin, xmax, ymin, ymax, zmin, zmax = result.extent
    dz = (zmax - zmin) / nz
    zc = zmin + (np.arange(nz) + 0.5) * dz
    ra = sol.raw_arrays
    frame = model.structural_frame
    base = Raster(z=np.zeros((ny, nx)), x0=xmin, y0=ymax, dx=(xmax - xmin) / nx, dy=(ymax - ymin) / ny,
                  crs=result.terrain.crs if result.terrain is not None else None)
    raw: dict[str, np.ndarray] = {}
    group_of: dict[str, int] = {}
    for gi, grp in enumerate(frame.structural_groups):
        F = ra.scalar_field_matrix[gi].reshape(nx, ny, nz)
        vals = ra.scalar_field_at_surface_points[gi]
        for el, v in zip(grp.elements, vals):
            s = _crossing(F, zc, float(v))
            s = np.clip(s, zmin, zmax)            # +/-inf -> box limits
            raw[el.name] = s[:, ::-1].T             # (nx,ny) -> raster (ny,nx), north up
            group_of[el.name] = gi
    # erosion / onlap between groups (group 0 is youngest)
    final = {k: v.copy() for k, v in raw.items()}
    groups = frame.structural_groups
    for gi, grp in enumerate(groups):
        lowest = grp.elements[-1].name
        relation = getattr(grp.structural_relation, "name", str(grp.structural_relation)).upper()
        for name, gj in group_of.items():
            if gj > gi and relation == "ERODE":          # older surfaces cut by younger erosion
                final[name] = np.minimum(final[name], raw[lowest])
            if gj < gi and relation == "ONLAP":
                pass
        if relation == "ONLAP" and gi + 1 < len(groups):  # this (younger) group onlaps older top
            older_top = groups[gi + 1].elements[0].name
            for el in grp.elements:
                final[el.name] = np.maximum(final[el.name], raw[older_top])
    # stratigraphic consistency within a group: base(U_k) <= base(U_{k-1})
    order = [u.name for u in strat.surface_units if u.name in final]
    for a, b in zip(order[:-1], order[1:]):
        if group_of[a] == group_of[b]:
            final[b] = np.minimum(final[b], final[a])
    out_cell = cell or (result.terrain.dx if result.terrain is not None else None)
    target = base if not out_cell else base.resample(out_cell, bounds=base.bounds)
    X, Y = target.mesh()
    topo = result.terrain.sample(X, Y) if result.terrain is not None else None
    result.surfaces, result.surfaces_clipped = {}, {}
    tr = 0.0
    if result.trend is not None:
        tr = result.trend.sample(X, Y)
        tr = np.where(np.isfinite(tr), tr, np.nanmean(result.trend.z))
    for name in order:
        r_raw = base.like(raw[name], name)
        r_fin = base.like(final[name], name)
        if out_cell:
            r_raw = target.like(r_raw.sample(X, Y), name)
            r_fin = target.like(r_fin.sample(X, Y), name)
        r_raw.z = r_raw.z + tr
        r_fin.z = r_fin.z + tr
        result.surfaces[name] = r_raw
        clipped = r_fin.z if topo is None else np.minimum(r_fin.z, topo)
        result.surfaces_clipped[name] = target.like(clipped, name)
    if topo is not None:
        result.surfaces_clipped["_topography"] = target.like(topo, "topography")
    return result


# =============================================================================== pipeline
def run_pipeline(bs: BoreholeSet, strat: Stratigraphy, terrain: Raster | None = None,
                 cfg: ModelConfig | None = None, extra_orientations: pd.DataFrame | None = None,
                 output_cell: float | None = None) -> ModelResult:
    """End-to-end: contacts -> orientations -> GemPy -> surfaces."""
    cfg = cfg or ModelConfig()
    points, issues = extract_contacts(bs, strat, pinchout=cfg.pinchout, offset=cfg.pinchout_offset)
    if points.empty:
        raise ValueError("No interface points extracted - check unit mapping/stratigraphy")
    trend = None
    model_terrain = terrain
    if terrain is not None and cfg.trend_alpha > 0:
        tz = terrain.z
        if cfg.trend_smooth > 0:
            from scipy.ndimage import gaussian_filter

            tz = gaussian_filter(np.nan_to_num(tz, nan=np.nanmean(tz)), cfg.trend_smooth / terrain.dx)
        trend = terrain.like(cfg.trend_alpha * tz, "trend")
        points = points.copy()
        points["z_real"] = points["z"]
        t = trend.sample(points.x.values, points.y.values)
        points["z"] = points["z"] - np.nan_to_num(t, nan=float(np.nanmean(trend.z)))
        model_terrain = terrain.like(terrain.z - trend.z, "terrain_model_space")
    ori = estimate_orientations(points, strat, local_k=cfg.local_orientations_k)
    if extra_orientations is not None and len(extra_orientations):
        e = parse_orientations(extra_orientations, strat)
        e["source"] = "user"
        if trend is not None:
            e["z"] = e["z"] - trend.sample(e.x.values, e.y.values)
        ori = pd.concat([ori, e[ori.columns]], ignore_index=True)
    if cfg.extent:
        extent = list(cfg.extent)
    elif trend is not None:
        extent = auto_extent(bs, terrain, points, cfg)
        zt = model_terrain.z
        bottoms = [bs.desurvey(h, [d])[0] for h, d in zip(bs.collars.hole_id, bs.collars.depth)
                   if np.isfinite(d)]
        bz = [p[2] - trend.sample([p[0]], [p[1]])[0] for p in bottoms]
        extent[4] = float(np.floor(np.nanmin([points.z.min(), *bz]) - cfg.pad_z))
        extent[5] = float(np.ceil(np.nanmax(zt) + cfg.pad_z))
    else:
        extent = auto_extent(bs, terrain, points, cfg)
    model = build_model(points, ori, strat, extent, cfg, model_terrain)
    sol, txt = compute(model)
    res = ModelResult(strat=strat, extent=list(extent), resolution=tuple(cfg.resolution), points=points,
                      orientations=ori, issues=issues, geo_model=model, solutions=sol, terrain=terrain,
                      log=txt, trend=trend, config=cfg)
    return extract_surfaces(res, cell=output_cell)


def borehole_misfit(res: ModelResult) -> pd.DataFrame:
    """Modelled surface elevation vs logged contact elevation at each contact."""
    rows = []
    for name, r in res.surfaces.items():
        p = res.points[(res.points.unit == name) & (res.points.kind == "contact")]
        if p.empty:
            continue
        zm = r.sample(p.x.values, p.y.values)
        zcol = "z_real" if "z_real" in p else "z"
        for (hid, z), m in zip(p[["hole_id", zcol]].itertuples(index=False), zm):
            rows.append((hid, name, z, m, m - z))
    return pd.DataFrame(rows, columns=["hole_id", "surface", "z_logged", "z_model", "error"])
