"""End-to-end on the synthetic site, checked against the known true surfaces."""
import json

import numpy as np
import pandas as pd
import pytest

from geosurf import BoreholeSet, ModelConfig, Stratigraphy, las_to_dtm, load_ags4, run_pipeline
from geosurf.model import IGNORE, borehole_misfit, extract_contacts
from geosurf.outputs import cross_section, export_all, thickness_maps, volumes
from geosurf.synthetic import DEMO_LITH_MAP, make_site, true_surfaces, write_demo_project


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    d = tmp_path_factory.mktemp("site")
    files = make_site(d, lidar_density=0.6)
    dtm = las_to_dtm(files["lidar_laz"], cell=4.0)
    bs = BoreholeSet.from_tables(pd.read_csv(files["collars_csv"]), pd.read_csv(files["lithology_csv"]),
                                 pd.read_csv(files["survey_csv"])).merge(load_ags4(files["ags4"]))
    bs.drape(dtm)
    bs.intervals["unit"] = bs.intervals["lith"].map(DEMO_LITH_MAP)
    strat = Stratigraphy.from_list(["MG", "AL", "RS", "GR"])
    res = run_pipeline(bs, strat, dtm, ModelConfig(resolution=(40, 40, 50)))
    return d, files, dtm, bs, res


def test_contacts(site):
    _, _, _, bs, res = site
    pts, issues = extract_contacts(bs, res.strat)
    assert set(pts.unit) == {"MG", "AL", "RS"}
    assert ((pts.kind == "contact") & (pts.unit == "RS")).sum() == len(bs.collars)  # every hole reaches GR
    assert issues.empty


def test_honours_boreholes(site):
    res = site[-1]
    mf = borehole_misfit(res)
    assert mf.error.abs().max() < 0.3


def test_close_to_truth(site):
    res = site[-1]
    X, Y = res.surfaces_clipped["RS"].mesh()
    tr = true_surfaces(X, Y)
    topo = res.surfaces_clipped["_topography"].z
    rmse = {k: np.sqrt(np.nanmean((res.surfaces_clipped[k].z - np.minimum(tr[k], topo)) ** 2)) for k in tr}
    assert rmse["MG"] < 1.0 and rmse["AL"] < 3.0 and rmse["RS"] < 6.0, rmse


def test_surfaces_consistent_with_gempy_block(site):
    res = site[-1]
    rng = np.random.default_rng(1)
    r = res.surfaces["RS"]
    xs = rng.uniform(r.bounds[0] + 50, r.bounds[1] - 50, 15)
    ys = rng.uniform(r.bounds[2] + 50, r.bounds[3] - 50, 15)
    z = r.sample(xs, ys)
    above = res.lith_at(np.c_[xs, ys, z + 1.5])
    below = res.lith_at(np.c_[xs, ys, z - 1.5])
    names = res.unit_names
    assert all(names[i - 1] != "GR" for i in above)
    assert all(names[i - 1] == "GR" for i in below)


def test_stacking_and_volumes(site):
    res = site[-1]
    s = res.surfaces_clipped
    assert np.all(s["AL"].z <= s["MG"].z + 1e-6) and np.all(s["RS"].z <= s["AL"].z + 1e-6)
    assert np.all(s["MG"].z <= s["_topography"].z + 1e-6)
    th = thickness_maps(res)
    assert all(np.nanmin(t.z) >= 0 for t in th.values())
    v = volumes(res).set_index("unit")
    assert v.loc["RS", "volume_m3"] > v.loc["AL", "volume_m3"] > v.loc["MG", "volume_m3"] > 0


def test_section_and_export(site, tmp_path):
    _, _, dtm, bs, res = site
    x0, x1, y0, y1 = dtm.bounds
    sec = cross_section(res, [[x0 + 20, y0 + 250], [x1 - 20, y0 + 250]], nz=40)
    assert sec["block"].shape == (40, len(sec["chainage"])) and len(sec["holes"]) > 0
    files = export_all(res, tmp_path / "out", sections={"A": [[x0 + 20, y0 + 20], [x1 - 20, y1 - 20]]},
                       boreholes=bs, max_faces=5000)
    names = {f.name for f in files}
    assert {"base_RS.tif", "base_RS.asc", "surfaces_tin.dxf", "boreholes.dxf", "A_3d.dxf", "volumes.csv"} <= names
    import ezdxf

    doc = ezdxf.readfile(tmp_path / "out" / "cad" / "surfaces_tin.dxf")
    assert len(doc.modelspace().query("3DFACE")) > 1000


def test_project_cli(site):
    d = site[0]
    p = write_demo_project(d)
    pj = json.loads(p.read_text())
    pj["model"]["resolution"] = [30, 30, 40]
    pj["terrain"]["cell"] = 5.0
    pj["output"]["formats"] = ["tif"]
    p.write_text(json.dumps(pj))
    from geosurf.cli import main

    assert main(["run", str(p)]) == 0
    assert (d / "output" / "surfaces" / "base_AL.tif").exists()
