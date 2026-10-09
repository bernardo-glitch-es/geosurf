"""Derived products and exports: isopachs, depth maps, volumes, cross-sections, CAD/GIS files."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .model import ModelResult
from .raster import Raster


# =============================================================================== derived grids
def surface_order(res: ModelResult) -> list[str]:
    return [k for k in res.surfaces_clipped if not k.startswith("_")]


def thickness_maps(res: ModelResult) -> dict[str, Raster]:
    """Thickness of each unit (top = overlying surface or terrain). Basement excluded."""
    order = surface_order(res)
    topo = res.surfaces_clipped.get("_topography")
    out = {}
    prev = topo.z if topo is not None else None
    for name in order:
        base = res.surfaces_clipped[name].z
        if prev is None:
            prev = base
            continue
        t = np.clip(prev - base, 0, None)
        out[name] = res.surfaces_clipped[name].like(t, f"thickness_{name}")
        prev = np.minimum(prev, base)
    return out


def depth_maps(res: ModelResult) -> dict[str, Raster]:
    """Depth below terrain to the base of each unit (0 where the unit is absent)."""
    topo = res.surfaces_clipped.get("_topography")
    if topo is None:
        return {}
    return {n: topo.like(np.clip(topo.z - res.surfaces_clipped[n].z, 0, None), f"depth_base_{n}")
            for n in surface_order(res)}


def volumes(res: ModelResult) -> pd.DataFrame:
    rows = []
    for name, r in thickness_maps(res).items():
        a = r.dx * r.dy
        t = r.z[np.isfinite(r.z)]
        rows.append((name, float(t.sum() * a), float((t > 0.01).sum() * a), float(t.mean()), float(t.max())))
    return pd.DataFrame(rows, columns=["unit", "volume_m3", "footprint_m2", "mean_thickness_m",
                                       "max_thickness_m"])


def distance_to_data(res: ModelResult) -> Raster:
    """Horizontal distance to nearest borehole contact - simple confidence proxy."""
    from scipy.spatial import cKDTree

    ref = next(iter(res.surfaces_clipped.values()))
    X, Y = ref.mesh()
    tree = cKDTree(res.points[["x", "y"]].drop_duplicates().to_numpy())
    d, _ = tree.query(np.c_[X.ravel(), Y.ravel()])
    return ref.like(d.reshape(X.shape), "distance_to_borehole")


# =============================================================================== sections
def densify(polyline, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(polyline, float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.r_[0, np.cumsum(seg)]
    s = np.arange(0, cum[-1] + 1e-9, spacing)
    if s[-1] < cum[-1]:
        s = np.r_[s, cum[-1]]
    xy = np.c_[np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])]
    return s, xy


def cross_section(res: ModelResult, polyline, spacing: float | None = None, nz: int = 150,
                  buffer: float = 25.0, with_block: bool = True) -> dict:
    """Section along a polyline. Returns distances, terrain profile, surface profiles, the
    GemPy lithology block (nz x n) and boreholes projected within ``buffer`` metres."""
    xmin, xmax, ymin, ymax, zmin, zmax = res.extent
    ref = next(iter(res.surfaces_clipped.values()))
    spacing = spacing or max(ref.dx, 1.0)
    s, xy = densify(polyline, spacing)
    topo = res.surfaces_clipped.get("_topography")
    tz = topo.sample(xy[:, 0], xy[:, 1]) if topo is not None else None
    profiles = {n: res.surfaces_clipped[n].sample(xy[:, 0], xy[:, 1]) for n in surface_order(res)}
    # vertical range: real elevations covered by the model
    zlo = np.nanmin([np.nanmin(v) for v in profiles.values()] + [zmin if res.trend is None else np.inf])
    zhi = np.nanmax(tz) if tz is not None else zmax
    zlo = zlo - 0.1 * (zhi - zlo)
    zs = np.linspace(zlo, zhi + 0.02 * (zhi - zlo), nz)
    block = None
    if with_block:
        n = len(s)
        S, Z = np.meshgrid(np.arange(n), zs)
        pts = np.c_[xy[S.ravel(), 0], xy[S.ravel(), 1], Z.ravel()]
        ids = res.lith_at(pts).reshape(nz, n).astype(float)
        if tz is not None:
            ids[Z > tz[None, :]] = np.nan
        block = ids
    # boreholes near the line
    holes = []
    p = np.asarray(polyline, float)
    seg_start = p[:-1]
    seg_vec = np.diff(p, axis=0)
    seg_len = np.hypot(*seg_vec.T)
    cum = np.r_[0, np.cumsum(seg_len)]
    for _, c in res.points.drop_duplicates("hole_id").iterrows():
        best = None
        for i, (a, v, L) in enumerate(zip(seg_start, seg_vec, seg_len)):
            t = np.clip(np.dot([c.x - a[0], c.y - a[1]], v) / (L * L + 1e-12), 0, 1)
            q = a + t * v
            d = np.hypot(c.x - q[0], c.y - q[1])
            if best is None or d < best[0]:
                best = (d, cum[i] + t * L)
        if best and best[0] <= buffer:
            holes.append({"hole_id": c.hole_id, "offset": best[0], "chainage": best[1]})
    return {"chainage": s, "xy": xy, "z": zs, "terrain": tz, "profiles": profiles, "block": block,
            "unit_names": res.unit_names, "holes": holes}


# =============================================================================== meshes
def grid_tin(r: Raster, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a raster (two triangles per cell, NaN cells dropped)."""
    z = r.z[::stride, ::stride]
    X, Y = r.mesh()
    X, Y = X[::stride, ::stride], Y[::stride, ::stride]
    ny, nx = z.shape
    verts = np.c_[X.ravel(), Y.ravel(), z.ravel()]
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    tris = np.r_[np.c_[a, c, b], np.c_[b, c, d]]
    ok = np.isfinite(verts[tris, 2]).all(1)
    tris = tris[ok]
    used = np.unique(tris)
    remap = -np.ones(len(verts), int)
    remap[used] = np.arange(len(used))
    return verts[used], remap[tris]


def write_obj(path, verts, tris, name="surface") -> Path:
    with open(path, "w") as f:
        f.write(f"# geosurf\no {name}\n")
        np.savetxt(f, verts, fmt="v %.3f %.3f %.3f")
        np.savetxt(f, tris + 1, fmt="f %d %d %d")
    return Path(path)


def write_vtk(path, verts, tris, scalars: dict[str, np.ndarray] | None = None) -> Path:
    """Legacy ASCII VTK PolyData (opens in ParaView, PyVista, Leapfrog via import)."""
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\ngeosurf surface\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(verts)} double\n")
        np.savetxt(f, verts, fmt="%.3f")
        f.write(f"POLYGONS {len(tris)} {4 * len(tris)}\n")
        np.savetxt(f, np.c_[np.full(len(tris), 3), tris], fmt="%d")
        sc = {"elevation": verts[:, 2], **(scalars or {})}
        f.write(f"POINT_DATA {len(verts)}\n")
        for k, v in sc.items():
            f.write(f"SCALARS {k} double 1\nLOOKUP_TABLE default\n")
            np.savetxt(f, v, fmt="%.3f")
    return Path(path)


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def write_dxf(path, layers: dict[str, dict]) -> Path:
    """layers: {name: {"tin": (verts, tris) | None, "contours": [Nx3 arrays] | None,
    "polylines": [Nx3 arrays], "color": "#rrggbb"}} -> DXF R2010 (3DFACE + 3D polylines)."""
    import ezdxf

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for name, spec in layers.items():
        lname = name.replace(" ", "_")[:250]
        lay = doc.layers.add(lname)
        if spec.get("color"):
            lay.rgb = _hex_to_rgb(spec["color"])
        if spec.get("tin") is not None:
            v, t = spec["tin"]
            for tri in t:
                a, b, c = v[tri[0]], v[tri[1]], v[tri[2]]
                msp.add_3dface([a, b, c, c], dxfattribs={"layer": lname})
        for key in ("contours", "polylines"):
            for line in spec.get(key) or []:
                if len(line) >= 2:
                    msp.add_polyline3d([tuple(p) for p in line], dxfattribs={"layer": lname})
    doc.saveas(path)
    return Path(path)


def contours(r: Raster, interval: float) -> list[np.ndarray]:
    """3D contour polylines of a raster at a regular interval."""
    from skimage import measure

    z = r.z
    if not np.isfinite(z).any():
        return []
    lo = np.ceil(np.nanmin(z) / interval) * interval
    hi = np.nanmax(z)
    zf = np.where(np.isfinite(z), z, np.nanmin(z) - 1e3)
    out = []
    for lev in np.arange(lo, hi + 1e-9, interval):
        for c in measure.find_contours(zf, lev):
            x = r.x0 + (c[:, 1] + 0.5) * r.dx
            y = r.y0 - (c[:, 0] + 0.5) * r.dy
            out.append(np.c_[x, y, np.full(len(c), lev)])
    return out


def gempy_meshes(res: ModelResult) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """True 3D GemPy dual-contouring meshes in real coordinates (trend added back)."""
    out = {}
    model, sol = res.geo_model, res.solutions
    names = [e.name for e in model.structural_frame.structural_elements]
    tf = model.input_transform
    for i, m in enumerate(getattr(sol, "dc_meshes", []) or []):
        try:
            v = model.grid.transform.apply_inverse(m.vertices) if hasattr(model.grid, "transform") else m.vertices
            v = tf.apply_inverse(v)
        except Exception:
            v = tf.apply_inverse(m.vertices)
        v = np.array(v, float)
        if res.trend is not None:
            t = res.trend.sample(v[:, 0], v[:, 1])
            v[:, 2] += np.nan_to_num(t, nan=float(np.nanmean(res.trend.z)))
        if i < len(names):
            out[names[i]] = (v, np.asarray(m.edges, int))
    return out


# =============================================================================== export
def export_all(res: ModelResult, folder, formats=("tif", "asc", "xyz", "dxf", "obj", "vtk"),
               contour_interval: float | None = 1.0, max_faces: int = 60_000,
               sections: dict[str, list] | None = None, include_gempy_meshes: bool = False,
               boreholes=None) -> list[Path]:
    folder = Path(folder)
    files: list[Path] = []
    sub = {k: folder / k for k in ("surfaces", "thickness", "depth", "meshes", "cad", "sections", "tables")}
    for p in sub.values():
        p.mkdir(parents=True, exist_ok=True)
    grids = {**{f"base_{k}": v for k, v in res.surfaces_clipped.items() if not k.startswith("_")}}
    if "_topography" in res.surfaces_clipped:
        grids["terrain"] = res.surfaces_clipped["_topography"]
    for name, r in grids.items():
        if "tif" in formats:
            files.append(r.to_geotiff(sub["surfaces"] / f"{name}.tif"))
        if "asc" in formats:
            files.append(r.to_esri_ascii(sub["surfaces"] / f"{name}.asc"))
        if "xyz" in formats:
            files.append(r.to_xyz(sub["surfaces"] / f"{name}.csv"))
    for kind, maps in (("thickness", thickness_maps(res)), ("depth", depth_maps(res))):
        for name, r in maps.items():
            files.append(r.to_geotiff(sub[kind] / f"{r.name}.tif"))
    files.append(distance_to_data(res).to_geotiff(sub["surfaces"] / "distance_to_borehole.tif"))
    # meshes + CAD
    layers = {}
    for name, r in grids.items():
        stride = max(1, int(np.ceil(np.sqrt(2 * r.nx * r.ny / max_faces))))
        v, t = grid_tin(r, stride)
        col = res.strat.color(name.replace("base_", "")) if name != "terrain" else "#888888"
        if "obj" in formats:
            files.append(write_obj(sub["meshes"] / f"{name}.obj", v, t, name))
        if "vtk" in formats:
            files.append(write_vtk(sub["meshes"] / f"{name}.vtk", v, t))
        layers[name] = {"tin": (v, t), "color": col,
                        "contours": contours(r, contour_interval) if contour_interval else None}
    if include_gempy_meshes:
        for name, (v, t) in gempy_meshes(res).items():
            files.append(write_obj(sub["meshes"] / f"gempy_{name}.obj", v, t, name))
    if "dxf" in formats:
        files.append(write_dxf(sub["cad"] / "surfaces_tin.dxf",
                               {k: {**s, "contours": None} for k, s in layers.items()}))
        if contour_interval:
            files.append(write_dxf(sub["cad"] / f"contours_{contour_interval:g}m.dxf",
                                   {f"{k}_contours": {"contours": s["contours"], "color": s["color"]}
                                    for k, s in layers.items()}))
        if boreholes is not None:
            tr = boreholes.traces()
            bl = {}
            for r in tr.itertuples():
                key = f"BH_{r.unit}"
                bl.setdefault(key, {"polylines": [], "color": res.strat.color(r.unit)})
                bl[key]["polylines"].append(np.array([[r.xt, r.yt, r.zt], [r.xb, r.yb, r.zb]]))
            files.append(write_dxf(sub["cad"] / "boreholes.dxf", bl))
    for name, line in (sections or {}).items():
        sec = cross_section(res, line, with_block=False)
        df = pd.DataFrame({"chainage": sec["chainage"], "x": sec["xy"][:, 0], "y": sec["xy"][:, 1]})
        if sec["terrain"] is not None:
            df["terrain"] = sec["terrain"]
        for k, v in sec["profiles"].items():
            df[f"base_{k}"] = v
        files.append(sub["sections"] / f"{name}.csv")
        df.to_csv(files[-1], index=False)
        lines = {f"{name}_terrain": {"polylines": [np.c_[sec["xy"], sec["terrain"]]], "color": "#888888"}} \
            if sec["terrain"] is not None else {}
        for k, v in sec["profiles"].items():
            ok = np.isfinite(v)
            lines[f"{name}_base_{k}"] = {"polylines": [np.c_[sec["xy"][ok], v[ok]]], "color": res.strat.color(k)}
        if "dxf" in formats:
            files.append(write_dxf(sub["sections"] / f"{name}_3d.dxf", lines))
    # tables
    files.append(sub["tables"] / "interface_points.csv")
    res.points.to_csv(files[-1], index=False)
    files.append(sub["tables"] / "orientations.csv")
    res.orientations.to_csv(files[-1], index=False)
    files.append(sub["tables"] / "volumes.csv")
    volumes(res).to_csv(files[-1], index=False)
    from .model import borehole_misfit

    files.append(sub["tables"] / "borehole_misfit.csv")
    borehole_misfit(res).to_csv(files[-1], index=False)
    for p in sub.values():
        if not any(p.iterdir()):
            p.rmdir()
    return files


def zip_folder(folder) -> bytes:
    folder = Path(folder)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(folder))
    return buf.getvalue()
