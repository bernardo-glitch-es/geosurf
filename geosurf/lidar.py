"""LiDAR loading: LAS/LAZ point clouds -> gridded digital terrain model (DTM)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .raster import Raster, fill_nan

GROUND_CLASS = 2  # ASPRS standard ground class


@dataclass
class LidarInfo:
    n_points: int
    bounds: tuple[float, float, float, float, float, float]
    classes: dict[int, int]
    crs: str | None
    point_format: int
    version: str


def las_info(path) -> LidarInfo:
    """Read header + class histogram (streams the file in chunks)."""
    import laspy

    classes: dict[int, int] = {}
    with laspy.open(path) as f:
        h = f.header
        for chunk in f.chunk_iterator(2_000_000):
            u, c = np.unique(np.asarray(chunk.classification), return_counts=True)
            for k, v in zip(u.tolist(), c.tolist()):
                classes[k] = classes.get(k, 0) + v
        crs = None
        try:
            parsed = h.parse_crs()
            crs = parsed.to_string() if parsed is not None else None
        except Exception:  # pyproj missing or no CRS VLR
            crs = None
        return LidarInfo(n_points=h.point_count,
                         bounds=(h.mins[0], h.maxs[0], h.mins[1], h.maxs[1], h.mins[2], h.maxs[2]),
                         classes=dict(sorted(classes.items())), crs=crs,
                         point_format=h.point_format.id, version=str(h.version))


def las_to_dtm(path, cell: float = 2.0, classes: list[int] | None = None, stat: str = "mean",
               fill: bool = True, crs: str | None = None, bounds=None) -> Raster:
    """Grid a LAS/LAZ point cloud into a DTM.

    Parameters
    ----------
    cell: output cell size in point-cloud units (m).
    classes: classification codes kept. ``None`` -> ground (2) if present, otherwise all points.
    stat: ``mean`` | ``min`` | ``max`` elevation per cell (``min`` is robust to residual vegetation).
    fill: interpolate empty cells (buildings, water, gaps).
    """
    import laspy

    info = las_info(path)
    if classes is None:
        classes = [GROUND_CLASS] if info.classes.get(GROUND_CLASS, 0) > 0 else None
    xmin, xmax, ymin, ymax = bounds or info.bounds[:4]
    nx = max(2, int(np.ceil((xmax - xmin) / cell)))
    ny = max(2, int(np.ceil((ymax - ymin) / cell)))
    n = nx * ny
    acc = np.full(n, np.inf if stat == "min" else (-np.inf if stat == "max" else 0.0))
    cnt = np.zeros(n)
    with laspy.open(path) as f:
        for ch in f.chunk_iterator(2_000_000):
            x, y, z = np.asarray(ch.x), np.asarray(ch.y), np.asarray(ch.z)
            keep = np.ones(len(x), bool)
            if classes is not None:
                keep &= np.isin(np.asarray(ch.classification), classes)
            keep &= (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)
            x, y, z = x[keep], y[keep], z[keep]
            col = np.clip(((x - xmin) / cell).astype(int), 0, nx - 1)
            row = np.clip(((ymax - y) / cell).astype(int), 0, ny - 1)
            idx = row * nx + col
            cnt += np.bincount(idx, minlength=n)
            if stat == "mean":
                acc += np.bincount(idx, weights=z, minlength=n)
            elif stat == "min":
                np.minimum.at(acc, idx, z)
            else:
                np.maximum.at(acc, idx, z)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = acc / cnt if stat == "mean" else acc
    z[cnt == 0] = np.nan
    z = z.reshape(ny, nx)
    coverage = float(np.isfinite(z).mean())
    if fill:
        z = fill_nan(z)
    r = Raster(z=z, x0=xmin, y0=ymax, dx=cell, dy=cell, crs=crs or info.crs,
               name=Path(str(path)).stem + "_dtm")
    r.meta.update(source=str(path), classes=classes, stat=stat, coverage=coverage,
                  n_points=info.n_points)
    return r


def load_terrain(path, cell: float = 2.0, **kw) -> Raster:
    """Load any supported terrain source: .las/.laz point cloud or GeoTIFF DEM."""
    ext = Path(str(path)).suffix.lower()
    if ext in (".las", ".laz"):
        return las_to_dtm(path, cell=cell, **kw)
    if ext in (".tif", ".tiff"):
        r = Raster.from_geotiff(path)
        if kw.get("crs"):
            r.crs = kw["crs"]
        return r
    raise ValueError(f"Unsupported terrain format: {ext}")


def terrain_from_points(xyz, cell: float = 1.0, pad: float = 10.0, crs: str | None = None,
                        name: str = "collar_terrain") -> Raster:
    """Fallback terrain when no LiDAR is available: linear TIN interpolation of surveyed
    points (e.g. borehole collars), nearest-neighbour outside the hull. Coarse – use only
    until a LiDAR/DTM arrives."""
    from scipy.interpolate import griddata

    p = np.asarray(xyz, float)
    p = p[np.isfinite(p).all(1)]
    if len(p) < 3:
        raise ValueError("Need at least 3 points with x, y, z")
    xmin, ymin = p[:, :2].min(0) - pad
    xmax, ymax = p[:, :2].max(0) + pad
    nx, ny = int(np.ceil((xmax - xmin) / cell)), int(np.ceil((ymax - ymin) / cell))
    r = Raster(z=np.zeros((ny, nx)), x0=xmin, y0=ymin + ny * cell, dx=cell, dy=cell, crs=crs, name=name)
    X, Y = r.mesh()
    z = griddata(p[:, :2], p[:, 2], (X, Y), method="linear")
    out = ~np.isfinite(z)
    if out.any():
        z[out] = griddata(p[:, :2], p[:, 2], (X[out], Y[out]), method="nearest")
    r.z = z
    r.meta.update(source="points", n_points=len(p))
    return r
