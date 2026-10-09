"""Command line: ``geosurf demo <folder>`` | ``geosurf run project.json`` | ``geosurf dtm in.laz out.tif``."""
from __future__ import annotations

import argparse
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(prog="geosurf", description="LiDAR + borehole -> GemPy geological surfaces")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="generate the synthetic demo site (+ project.json)")
    d.add_argument("folder", nargs="?", default="examples/synthetic_site")
    r = sub.add_parser("run", help="run a project.json end-to-end and export surfaces")
    r.add_argument("project")
    t = sub.add_parser("dtm", help="grid a LAS/LAZ point cloud to a GeoTIFF DTM")
    t.add_argument("las")
    t.add_argument("out")
    t.add_argument("--cell", type=float, default=1.0)
    t.add_argument("--stat", default="mean", choices=["mean", "min", "max"])
    t.add_argument("--classes", default=None, help="comma separated, default ground (2)")
    t.add_argument("--crs", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.cmd == "demo":
        from .synthetic import make_site, write_demo_project

        files = make_site(a.folder)
        files["project"] = write_demo_project(a.folder)
        for k, v in files.items():
            print(f"{k:16s} {v}")
    elif a.cmd == "run":
        from .model import borehole_misfit
        from .outputs import volumes
        from .project import run_project

        out = run_project(a.project)
        res = out["result"]
        print(f"model extent {res.extent}, {len(res.points)} interface points")
        if len(res.issues):
            print("issues:\n", res.issues.to_string(index=False))
        print(borehole_misfit(res).groupby("surface")["error"].describe().round(3).to_string())
        print(volumes(res).round(1).to_string(index=False))
        print(f"{len(out['files'])} files written")
    elif a.cmd == "dtm":
        from .lidar import las_to_dtm

        cls = [int(c) for c in a.classes.split(",")] if a.classes else None
        r = las_to_dtm(a.las, cell=a.cell, classes=cls, stat=a.stat, crs=a.crs)
        r.to_geotiff(a.out)
        print(f"DTM {r.nx}x{r.ny} @ {r.dx} m, coverage {r.meta['coverage']:.1%} -> {a.out}")
    print(f"done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
