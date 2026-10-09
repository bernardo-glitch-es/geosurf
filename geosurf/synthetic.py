"""Synthetic site generator: LiDAR (LAZ + GeoTIFF), boreholes (CSV, Excel, AGS4) and the
"true" surfaces, so the whole workflow can be tested and the result checked against truth.

Fictitious site in a granitic valley, ETRS89 / UTM 29N (EPSG:25829):
    MG  Made ground (fill platform, SE corner)
    AL  Alluvium (valley floor; logged as CL clay / SA sand)
    RS  Residual soil - completely weathered granite ("jabre")
    GR  Granite bedrock (basement)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .ags4 import write_ags4
from .raster import Raster

CRS = "EPSG:25829"
X0, Y0, W, H = 545_000.0, 4_790_000.0, 600.0, 500.0


def topo(x, y):
    """Natural ground plus a fill platform (made ground) in the SE corner."""
    nat = _natural(x, y)
    return nat + _platform(x, y) * np.maximum(0.0, 56.0 - nat)


def _platform(x, y):
    u, v = (x - X0) / W, (y - Y0) / H
    return np.clip(1 - np.hypot((u - 0.78) / 0.18, (v - 0.2) / 0.15), 0, 0.15) / 0.15


def _natural(x, y):
    u, v = (x - X0) / W, (y - Y0) / H
    valley = 18 * (1 - np.exp(-((v - 0.45 - 0.15 * u) ** 2) / 0.03))
    return 42 + 22 * u + valley + 3 * np.sin(6 * u) * np.cos(4 * v)


def true_surfaces(x, y) -> dict[str, np.ndarray]:
    """Elevation of the BASE of each unit (GemPy convention)."""
    u, v = (x - X0) / W, (y - Y0) / H
    t = topo(x, y)
    nat = _natural(x, y)
    base_mg = np.where(_platform(x, y) > 0.02, np.minimum(t, nat - 0.5), t)
    al_th = np.clip(7.5 * np.exp(-((v - 0.45 - 0.15 * u) ** 2) / 0.012) - 0.6, 0, None)
    base_al = base_mg - al_th
    rs_th = 6 + 4 * np.sin(3 * u + 1) + 2.5 * np.cos(5 * v) + 2 * u
    base_rs = base_al - rs_th
    return {"MG": base_mg, "AL": base_al, "RS": base_rs}


def _log_hole(x, y, z0, depth, az, dip, rng):
    """Walk down an (inclined) hole and log unit changes against the true surfaces."""
    md = np.arange(0, depth + 1e-9, 0.05)
    d = np.radians(dip)
    a = np.radians(az)
    xs, ys, zs = x + md * np.cos(d) * np.sin(a), y + md * np.cos(d) * np.cos(a), z0 - md * np.sin(d)
    s = true_surfaces(xs, ys)
    unit = np.where(zs > s["MG"], "MG", np.where(zs > s["AL"], "AL", np.where(zs > s["RS"], "RS", "GR")))
    rows, start = [], 0.0
    for i in range(1, len(md)):
        if unit[i] != unit[i - 1]:
            rows.append([round(start, 1), round(md[i], 1), unit[i - 1]])
            start = md[i]
    rows.append([round(start, 1), round(depth, 1), unit[-1]])
    out = []
    for top, base, u in rows:
        if base <= top:
            continue
        if u == "AL":  # alluvium logged as clay over sand
            mid = round(top + (base - top) * rng.uniform(0.3, 0.7), 1)
            out += [[top, mid, "CL"], [mid, base, "SA"]] if base - top > 1.0 else [[top, base, "CL"]]
        else:
            out.append([top, base, u])
    return out


DESC = {"MG": "MADE GROUND: brown sandy gravel with brick fragments", "CL": "Soft grey silty CLAY",
        "SA": "Medium dense brown fine to coarse SAND", "RS": "Completely weathered GRANITE recovered as "
        "dense clayey gravelly SAND (jabre)", "GR": "Moderately weathered pink coarse grained GRANITE",
        "CORE LOSS": "No recovery"}


def make_site(folder, n_holes: int = 24, seed: int = 7, lidar_density: float = 1.5) -> dict[str, Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    files: dict[str, Path] = {}

    # ---------------------------------------------------------------- LiDAR point cloud
    import laspy

    n = int(W * H * lidar_density)
    gx, gy = X0 + rng.uniform(0, W, n), Y0 + rng.uniform(0, H, n)
    gz = topo(gx, gy) + rng.normal(0, 0.05, n)
    veg_n = n // 4
    vi = rng.choice(n, veg_n, replace=False)
    vx, vy = gx[vi] + rng.normal(0, 0.5, veg_n), gy[vi] + rng.normal(0, 0.5, veg_n)
    vz = topo(vx, vy) + rng.gamma(2.0, 3.0, veg_n)
    bld = (np.abs(gx - (X0 + 470)) < 15) & (np.abs(gy - (Y0 + 100)) < 10)  # building on platform
    cls = np.full(n, 2, np.uint8)
    cls[bld] = 6
    gz[bld] += 7.0
    x = np.r_[gx, vx]
    y = np.r_[gy, vy]
    z = np.r_[gz, vz]
    c = np.r_[cls, np.full(veg_n, 5, np.uint8)]
    hdr = laspy.LasHeader(point_format=1, version="1.4")
    hdr.offsets = [X0, Y0, 0]
    hdr.scales = [0.01, 0.01, 0.01]
    las = laspy.LasData(hdr)
    las.x, las.y, las.z, las.classification = x, y, z, c
    files["lidar_laz"] = folder / "site_lidar.laz"
    las.write(files["lidar_laz"])

    # ---------------------------------------------------------------- DEM GeoTIFF (1 m)
    dem = Raster(z=np.zeros((int(H), int(W))), x0=X0, y0=Y0 + H, dx=1.0, dy=1.0, crs=CRS, name="dem")
    X, Y = dem.mesh()
    dem.z = topo(X, Y)
    files["dem_tif"] = dem.to_geotiff(folder / "site_dem_1m.tif")
    for k, s in true_surfaces(X, Y).items():
        dem.like(s, k).to_geotiff(folder / f"truth_base_{k}.tif")

    # ---------------------------------------------------------------- boreholes
    hx = X0 + 30 + (W - 60) * rng.random(n_holes)
    hy = Y0 + 30 + (H - 60) * rng.random(n_holes)
    # make sure the valley axis is sampled
    hx[:5] = X0 + np.linspace(60, W - 60, 5)
    hy[:5] = Y0 + H * (0.45 + 0.15 * (hx[:5] - X0) / W)
    collars, ivs, survey = [], [], []
    for i in range(n_holes):
        hid = f"BH{i + 1:02d}"
        z0 = float(topo(hx[i], hy[i]))
        s = true_surfaces(np.array([hx[i]]), np.array([hy[i]]))
        depth = round(float(z0 - s["RS"][0]) + rng.uniform(3, 8), 1)
        az, dip = (0.0, 90.0)
        if i in (6, 13):
            az, dip = float(rng.uniform(0, 360)), 70.0
            survey += [[hid, 0.0, az, dip], [hid, depth, az + 4, dip - 3]]
        log = _log_hole(hx[i], hy[i], z0, depth, az, dip, rng)
        if i == 9:  # a core-loss interval inside RS
            for r in log:
                if r[2] == "RS" and r[1] - r[0] > 3:
                    m = round((r[0] + r[1]) / 2, 1)
                    log = [*[q for q in log if q is not r], [r[0], m, "RS"], [m, m + 0.6, "CORE LOSS"],
                           [m + 0.6, r[1], "RS"]]
                    log.sort(key=lambda q: q[0])
                    break
        zl = round(z0 + rng.normal(0, 0.08), 2)
        collars.append([hid, round(hx[i], 2), round(hy[i], 2), zl, depth, az, dip])
        ivs += [[hid, t, b, u, DESC[u]] for t, b, u in log]
    col = pd.DataFrame(collars, columns=["HoleID", "Easting", "Northing", "RL", "TotalDepth", "Azimuth", "Dip"])
    iv = pd.DataFrame(ivs, columns=["HoleID", "From", "To", "Lith", "Description"])
    sv = pd.DataFrame(survey, columns=["HoleID", "Depth", "Azimuth", "Dip"])

    # first 16 holes -> CSV/Excel (a few without RL to demonstrate LiDAR draping)
    csv_ids = col.HoleID.iloc[:16]
    c1 = col[col.HoleID.isin(csv_ids)].copy()
    c1.loc[c1.index[[2, 11]], "RL"] = np.nan
    files["collars_csv"] = folder / "collars.csv"
    c1.to_csv(files["collars_csv"], index=False)
    files["lithology_csv"] = folder / "lithology.csv"
    iv[iv.HoleID.isin(csv_ids)].to_csv(files["lithology_csv"], index=False)
    files["survey_csv"] = folder / "survey.csv"
    sv[sv.HoleID.isin(csv_ids)].to_csv(files["survey_csv"], index=False)
    files["boreholes_xlsx"] = folder / "boreholes.xlsx"
    with pd.ExcelWriter(files["boreholes_xlsx"]) as xw:
        c1.to_excel(xw, sheet_name="Collars", index=False)
        iv[iv.HoleID.isin(csv_ids)].to_excel(xw, sheet_name="Lithology", index=False)
        sv[sv.HoleID.isin(csv_ids)].to_excel(xw, sheet_name="Survey", index=False)

    # remaining holes -> AGS4 (vertical holes only in this file), with SPT tests
    ags_col = col[~col.HoleID.isin(csv_ids)]
    ags_iv = iv[iv.HoleID.isin(ags_col.HoleID)]
    loca = pd.DataFrame({"LOCA_ID": ags_col.HoleID, "LOCA_TYPE": "CP", "LOCA_NATE": ags_col.Easting,
                         "LOCA_NATN": ags_col.Northing, "LOCA_GL": ags_col.RL,
                         "LOCA_FDEP": ags_col.TotalDepth})
    geol = pd.DataFrame({"LOCA_ID": ags_iv.HoleID, "GEOL_TOP": ags_iv.From, "GEOL_BASE": ags_iv.To,
                         "GEOL_DESC": ags_iv.Description, "GEOL_LEG": "", "GEOL_GEOL": ags_iv.Lith})
    spt = []
    for hid, td in zip(ags_col.HoleID, ags_col.TotalDepth):
        for d in np.arange(1.5, td - 1, 1.5):
            spt.append([hid, round(d, 2), int(np.clip(5 + 3 * d + rng.normal(0, 4), 1, 50))])
    ispt = pd.DataFrame(spt, columns=["LOCA_ID", "ISPT_TOP", "ISPT_NVAL"])
    proj = pd.DataFrame({"PROJ_ID": ["SYN-001"], "PROJ_NAME": ["Synthetic granite valley site"],
                         "PROJ_LOC": ["Galicia (fictitious)"], "PROJ_CLNT": ["Demo"]})
    tran = pd.DataFrame({"TRAN_ISNO": ["1"], "TRAN_DATE": ["2026-10-08"], "TRAN_PROD": ["geosurf"],
                         "TRAN_STAT": ["FINAL"], "TRAN_AGS": ["4.1"], "TRAN_RCON": ["+"],
                         "TRAN_DLIM": ["|"]})
    files["ags4"] = write_ags4(folder / "site_investigation.ags",
                               {"PROJ": proj, "TRAN": tran, "LOCA": loca, "GEOL": geol, "ISPT": ispt},
                               units={"LOCA": {"LOCA_NATE": "m", "LOCA_NATN": "m", "LOCA_GL": "m",
                                               "LOCA_FDEP": "m"},
                                      "GEOL": {"GEOL_TOP": "m", "GEOL_BASE": "m"},
                                      "ISPT": {"ISPT_TOP": "m"}})
    # structural measurement example (optional orientations): none needed - flat-lying units
    return files


if __name__ == "__main__":
    import sys

    for k, v in make_site(sys.argv[1] if len(sys.argv) > 1 else "examples/synthetic_site").items():
        print(f"{k:16s} {v}")


DEMO_LITH_MAP = {"MG": "MG", "CL": "AL", "SA": "AL", "RS": "RS", "GR": "GR", "CORE LOSS": "(ignore)"}
DEMO_UNITS = [("MG", "Made ground"), ("AL", "Alluvium"), ("RS", "Residual soil"), ("GR", "Granite")]


def write_demo_project(folder) -> Path:
    """project.json for the synthetic site (used by ``geosurf run``)."""
    import json

    from .model import ModelConfig, Stratigraphy
    from .project import make_project_dict

    strat = Stratigraphy.from_list([u for u, _ in DEMO_UNITS])
    pj = make_project_dict(
        "synthetic_site", CRS, {"path": "site_lidar.laz", "cell": 2.0, "stat": "mean"},
        [{"type": "tables", "collars": "collars.csv", "intervals": "lithology.csv", "survey": "survey.csv"},
         {"type": "ags4", "path": "site_investigation.ags"}],
        DEMO_LITH_MAP, strat, ModelConfig(),
        {"folder": "output", "cell": 2.0, "formats": ["tif", "asc", "xyz", "dxf", "obj", "vtk"],
         "contour_interval": 1.0,
         "sections": {"A": [[X0 + 50, Y0 + 50], [X0 + 300, Y0 + 450], [X0 + 580, Y0 + 300]]}})
    p = Path(folder) / "project.json"
    p.write_text(json.dumps(pj, indent=2))
    return p
