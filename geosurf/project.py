"""Project files (JSON) so a model set up in the app can be re-run automatically (CLI/batch).

Example ``project.json``::

    {
      "name": "my_site",
      "crs": "EPSG:25829",
      "terrain": {"path": "lidar.laz", "cell": 2.0, "classes": [2], "stat": "mean"},
      "boreholes": [
        {"type": "tables", "collars": "collars.csv", "intervals": "lithology.csv", "survey": "survey.csv",
         "collar_map": null, "interval_map": null, "survey_map": null},
        {"type": "ags4", "path": "site.ags", "lith_field": "auto"},
        {"type": "flat", "path": "logs.xlsx", "sheet": "Logs", "map": null}
      ],
      "drape": {"mode": "missing", "tolerance": 1.0},
      "lith_map": {"CL": "AL", "SA": "AL", "CORE LOSS": "(ignore)"},
      "stratigraphy": {"units": [{"name": "MG", "group": "MG", "color": "#c8a165"}, ...]},
      "model": {"resolution": [50, 50, 60], "trend_alpha": 1.0, ...},
      "output": {"folder": "out", "cell": 2.0, "formats": ["tif", "dxf"], "contour_interval": 1.0,
                 "sections": {"A": [[x1, y1], [x2, y2]]}}
    }
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .ags4 import load_ags4
from .boreholes import BoreholeSet, read_table
from .lidar import load_terrain
from .model import IGNORE, ModelConfig, Stratigraphy, run_pipeline


def _p(base: Path, p):
    return p if p is None or Path(p).is_absolute() else base / p


def load_boreholes(specs: list[dict], base: Path) -> BoreholeSet:
    out = None
    for s in specs:
        t = s.get("type", "tables")
        if t == "ags4":
            bs = load_ags4(_p(base, s["path"]), lith_field=s.get("lith_field", "auto"))
        elif t == "flat":
            df = read_table(_p(base, s["path"]), sheet=s.get("sheet", 0))
            bs = BoreholeSet.from_tables(None, df, interval_map=s.get("map"), flat=True)
        else:
            def rd(key, sheet_key):
                if not s.get(key):
                    return None
                return read_table(_p(base, s[key]), sheet=s.get(sheet_key, 0))
            bs = BoreholeSet.from_tables(rd("collars", "collars_sheet"), rd("intervals", "intervals_sheet"),
                                         rd("survey", "survey_sheet"), collar_map=s.get("collar_map"),
                                         interval_map=s.get("interval_map"), survey_map=s.get("survey_map"))
        out = bs if out is None else out.merge(bs)
    return out


def apply_lith_map(bs: BoreholeSet, lith_map: dict[str, str]) -> None:
    bs.intervals["unit"] = bs.intervals["lith"].map(lambda c: lith_map.get(c, c))


def run_project(path) -> dict:
    """Run a project file end to end; returns {'result', 'files', 'boreholes', 'terrain'}."""
    from .outputs import export_all

    path = Path(path)
    base = path.parent
    pj = json.loads(path.read_text())
    terrain = None
    if pj.get("terrain"):
        t = pj["terrain"]
        kw = {k: t[k] for k in ("classes", "stat") if k in t}
        terrain = load_terrain(_p(base, t["path"]), cell=t.get("cell", 2.0), crs=pj.get("crs"), **kw)
    bs = load_boreholes(pj["boreholes"], base)
    if terrain is not None and pj.get("drape", {}).get("mode", "missing") != "none":
        bs.drape(terrain, mode=pj.get("drape", {}).get("mode", "missing"),
                 tolerance=pj.get("drape", {}).get("tolerance", 1.0))
    apply_lith_map(bs, pj.get("lith_map", {}))
    strat = Stratigraphy.from_dict(pj["stratigraphy"])
    cfg = ModelConfig(**{k: (tuple(v) if k == "resolution" else v) for k, v in pj.get("model", {}).items()})
    out = pj.get("output", {})
    res = run_pipeline(bs, strat, terrain, cfg, output_cell=out.get("cell"))
    files = export_all(res, _p(base, out.get("folder", "output")),
                       formats=tuple(out.get("formats", ("tif", "asc", "xyz", "dxf", "obj", "vtk"))),
                       contour_interval=out.get("contour_interval", 1.0), sections=out.get("sections"),
                       boreholes=bs)
    return {"result": res, "files": files, "boreholes": bs, "terrain": terrain}


def make_project_dict(name, crs, terrain_spec, borehole_specs, lith_map, strat: Stratigraphy,
                      cfg: ModelConfig, output: dict, drape: dict | None = None) -> dict:
    c = asdict(cfg)
    c["resolution"] = list(c["resolution"])
    return {"name": name, "crs": crs, "terrain": terrain_spec, "boreholes": borehole_specs,
            "drape": drape or {"mode": "missing", "tolerance": 1.0},
            "lith_map": {k: v for k, v in lith_map.items()}, "stratigraphy": strat.to_dict(),
            "model": c, "output": output}


__all__ = ["run_project", "make_project_dict", "load_boreholes", "apply_lith_map", "IGNORE", "pd"]
