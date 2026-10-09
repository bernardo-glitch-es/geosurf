"""Light-weight regular grid (raster) used for terrain models and output surfaces.

Convention: ``z`` has shape (ny, nx); row 0 is the NORTH edge (GeoTIFF convention).
``x0, y0`` is the upper-left corner of the upper-left cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Raster:
    z: np.ndarray
    x0: float
    y0: float
    dx: float
    dy: float
    crs: str | None = None
    name: str = "raster"
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ geometry
    @property
    def ny(self) -> int:
        return self.z.shape[0]

    @property
    def nx(self) -> int:
        return self.z.shape[1]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) of the cell edges."""
        return self.x0, self.x0 + self.nx * self.dx, self.y0 - self.ny * self.dy, self.y0

    @property
    def xc(self) -> np.ndarray:
        return self.x0 + (np.arange(self.nx) + 0.5) * self.dx

    @property
    def yc(self) -> np.ndarray:
        return self.y0 - (np.arange(self.ny) + 0.5) * self.dy

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.xc, self.yc)

    def xyz(self, drop_nan: bool = True) -> np.ndarray:
        X, Y = self.mesh()
        pts = np.c_[X.ravel(), Y.ravel(), self.z.ravel()]
        return pts[np.isfinite(pts[:, 2])] if drop_nan else pts

    def like(self, z: np.ndarray, name: str | None = None) -> "Raster":
        return Raster(z=z, x0=self.x0, y0=self.y0, dx=self.dx, dy=self.dy, crs=self.crs,
                      name=name or self.name)

    # ------------------------------------------------------------------ sampling
    def sample(self, x, y) -> np.ndarray:
        """Bilinear interpolation at arbitrary points (NaN outside)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        fc = (x - self.x0) / self.dx - 0.5
        fr = (self.y0 - y) / self.dy - 0.5
        c0 = np.floor(fc).astype(int)
        r0 = np.floor(fr).astype(int)
        tc = fc - c0
        tr = fr - r0
        c0c = np.clip(c0, 0, self.nx - 1)
        c1c = np.clip(c0 + 1, 0, self.nx - 1)
        r0c = np.clip(r0, 0, self.ny - 1)
        r1c = np.clip(r0 + 1, 0, self.ny - 1)
        z = self.z
        v = (z[r0c, c0c] * (1 - tc) * (1 - tr) + z[r0c, c1c] * tc * (1 - tr)
             + z[r1c, c0c] * (1 - tc) * tr + z[r1c, c1c] * tc * tr)
        outside = (fc < -0.5) | (fc > self.nx - 0.5) | (fr < -0.5) | (fr > self.ny - 0.5)
        return np.where(outside, np.nan, v)

    def resample(self, dx: float, bounds: tuple | None = None) -> "Raster":
        xmin, xmax, ymin, ymax = bounds or self.bounds
        nx = max(2, int(round((xmax - xmin) / dx)))
        ny = max(2, int(round((ymax - ymin) / dx)))
        out = Raster(z=np.zeros((ny, nx)), x0=xmin, y0=ymax, dx=(xmax - xmin) / nx,
                     dy=(ymax - ymin) / ny, crs=self.crs, name=self.name)
        X, Y = out.mesh()
        out.z = self.sample(X, Y)
        return out

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_geotiff(cls, path, name: str | None = None) -> "Raster":
        import rasterio

        with rasterio.open(path) as src:
            z = src.read(1).astype(float)
            if src.nodata is not None:
                z[z == src.nodata] = np.nan
            t = src.transform
            if abs(t.b) > 1e-12 or abs(t.d) > 1e-12:
                raise ValueError("Rotated GeoTIFFs are not supported")
            dy = -t.e
            z_, y0 = (z, t.f) if dy > 0 else (z[::-1], t.f + z.shape[0] * t.e)
            return cls(z=z_, x0=t.c, y0=y0, dx=t.a, dy=abs(dy),
                       crs=src.crs.to_string() if src.crs else None,
                       name=name or Path(str(path)).stem)

    def to_geotiff(self, path, nodata: float = -9999.0) -> Path:
        import rasterio
        from rasterio.transform import from_origin

        z = np.where(np.isfinite(self.z), self.z, nodata).astype("float32")
        with rasterio.open(path, "w", driver="GTiff", height=self.ny, width=self.nx, count=1,
                           dtype="float32", crs=self.crs, nodata=nodata, compress="deflate",
                           transform=from_origin(self.x0, self.y0, self.dx, self.dy)) as dst:
            dst.write(z, 1)
        return Path(path)

    def to_esri_ascii(self, path, nodata: float = -9999.0) -> Path:
        if abs(self.dx - self.dy) > 1e-6 * self.dx:
            raise ValueError("Esri ASCII grid requires square cells")
        z = np.where(np.isfinite(self.z), self.z, nodata)
        xmin, _, ymin, _ = self.bounds
        hdr = (f"ncols {self.nx}\nnrows {self.ny}\nxllcorner {xmin:.4f}\nyllcorner {ymin:.4f}\n"
               f"cellsize {self.dx:.6f}\nNODATA_value {nodata}\n")
        with open(path, "w") as f:
            f.write(hdr)
            np.savetxt(f, z, fmt="%.3f")
        return Path(path)

    def to_xyz(self, path) -> Path:
        np.savetxt(path, self.xyz(), fmt="%.3f", delimiter=",", header="x,y,z", comments="")
        return Path(path)


def fill_nan(z: np.ndarray) -> np.ndarray:
    """Fill NaN cells: linear interpolation inside the convex hull, nearest outside."""
    from scipy.interpolate import griddata

    bad = ~np.isfinite(z)
    if not bad.any() or bad.all():
        return z
    r, c = np.indices(z.shape)
    pts = np.c_[r[~bad], c[~bad]]
    vals = z[~bad]
    out = z.copy()
    lin = griddata(pts, vals, (r[bad], c[bad]), method="linear")
    still = ~np.isfinite(lin)
    if still.any():
        lin[still] = griddata(pts, vals, (r[bad][still], c[bad][still]), method="nearest")
    out[bad] = lin
    return out
