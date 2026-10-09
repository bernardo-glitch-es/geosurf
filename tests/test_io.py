import numpy as np
import pandas as pd
import pytest

from geosurf.ags4 import load_ags4, read_ags4, write_ags4
from geosurf.boreholes import BoreholeSet, guess_columns
from geosurf.lidar import las_to_dtm
from geosurf.raster import Raster


def plane(x, y):
    return 100 + 0.05 * (x - 1000) - 0.02 * (y - 2000)


@pytest.fixture
def ramp():
    r = Raster(z=np.zeros((50, 60)), x0=1000, y0=2050, dx=1.0, dy=1.0, crs="EPSG:25830")
    X, Y = r.mesh()
    r.z = plane(X, Y)
    return r


def test_raster_sample_and_geotiff(ramp, tmp_path):
    x, y = np.array([1010.3, 1055.0]), np.array([2010.7, 2040.2])
    assert np.allclose(ramp.sample(x, y), plane(x, y), atol=1e-9)
    assert np.isnan(ramp.sample([900.0], [2010.0]))[0]
    back = Raster.from_geotiff(ramp.to_geotiff(tmp_path / "r.tif"))
    assert back.z.shape == ramp.z.shape and np.allclose(back.z, ramp.z, atol=1e-4)
    assert back.bounds == pytest.approx(ramp.bounds)
    assert "25830" in back.crs


def test_las_to_dtm(tmp_path):
    import laspy

    rng = np.random.default_rng(0)
    n = 20000
    x, y = 1000 + rng.uniform(0, 100, n), 2000 + rng.uniform(0, 80, n)
    z = plane(x, y)
    cls = np.full(n, 2, np.uint8)
    cls[:3000] = 5
    z[:3000] += 10  # vegetation must be filtered out
    h = laspy.LasHeader(point_format=1, version="1.4")
    h.offsets, h.scales = [1000, 2000, 0], [0.001] * 3
    las = laspy.LasData(h)
    las.x, las.y, las.z, las.classification = x, y, z, cls
    las.write(tmp_path / "t.laz")
    r = las_to_dtm(tmp_path / "t.laz", cell=2.0)
    X, Y = r.mesh()
    err = r.z - plane(X, Y)
    assert np.nanmax(np.abs(err[2:-2, 2:-2])) < 0.15
    assert r.meta["classes"] == [2]


def test_guess_columns_and_desurvey():
    col = pd.DataFrame({"BHID": ["A", "B"], "Easting": [0, 10], "Northing": [0, 0], "Elevation": [50, 52],
                        "EOH": [20, 20], "Azi": [90, 0], "Dip": [-60, -90]})
    iv = pd.DataFrame({"Hole": ["A", "A", "B"], "From": [0, 5, 0], "To": [5, 20, 20], "Litho": ["X", "Y", "X"]})
    g = guess_columns(col, "collars")
    assert g == {"hole_id": "BHID", "x": "Easting", "y": "Northing", "z": "Elevation", "depth": "EOH",
                 "azimuth": "Azi", "dip": "Dip"}
    bs = BoreholeSet.from_tables(col, iv)
    p = bs.desurvey("A", [10.0])[0]  # 60 deg below horizontal towards east
    assert p == pytest.approx([10 * np.cos(np.radians(60)), 0, 50 - 10 * np.sin(np.radians(60))])
    assert bs.desurvey("B", [7.5])[0] == pytest.approx([10, 0, 44.5])


def test_qa_and_drape(ramp):
    col = pd.DataFrame({"id": ["A", "B"], "x": [1010, 1020], "y": [2010, 2020], "z": [np.nan, 99.0]})
    iv = pd.DataFrame({"id": ["A", "A", "B"], "from": [0, 3, 0], "to": [2, 6, 4], "lith": ["S", "C", "S"]})
    bs = BoreholeSet.from_tables(col, iv)
    qa = bs.qa()
    assert {"gap", "missing collar elevation"} <= set(qa.issue)
    rep = bs.drape(ramp, mode="missing", tolerance=0.5)
    a = bs.collars.set_index("hole_id")
    assert a.loc["A", "z"] == pytest.approx(plane(1010, 2010)) and a.loc["A", "z_source"] == "lidar"
    assert a.loc["B", "z"] == 99.0 and rep.set_index("hole_id").loc["B", "flag"] != ""


def test_ags4_roundtrip(tmp_path):
    loca = pd.DataFrame({"LOCA_ID": ["BH1"], "LOCA_NATE": [500.0], "LOCA_NATN": [600.0], "LOCA_GL": [12.5],
                         "LOCA_FDEP": [8.0]})
    geol = pd.DataFrame({"LOCA_ID": ["BH1", "BH1"], "GEOL_TOP": [0, 3.2], "GEOL_BASE": [3.2, 8.0],
                         "GEOL_DESC": ['Soft "grey" CLAY, with sand', "SAND"], "GEOL_GEOL": ["CL", "SA"]})
    p = write_ags4(tmp_path / "x.ags", {"LOCA": loca, "GEOL": geol})
    g = read_ags4(p)
    assert g["GEOL"].loc[0, "GEOL_DESC"] == 'Soft "grey" CLAY, with sand'
    bs = load_ags4(p)
    assert bs.collars.loc[0, "z"] == 12.5 and bs.lith_codes == ["CL", "SA"]
