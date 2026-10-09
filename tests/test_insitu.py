import numpy as np
import pandas as pd
import pytest

from geosurf.boreholes import BoreholeSet
from geosurf.insitu import lab_from_table, parse_n, spt_by_unit, spt_from_table, with_xyz
from geosurf.insitu import tests_from_workbook as load_wb_tests


@pytest.mark.parametrize("v,exp", [("17", (17, False)), (17, (17, False)), ("R", (50, True)),
                                   ("50R", (50, True)), (">50", (50, True)), ("", (np.nan, False))])
def test_parse_n(v, exp):
    n, r = parse_n(v)
    assert (np.isnan(n) and np.isnan(exp[0])) or n == exp[0]
    assert r == exp[1]


def _bs():
    col = pd.DataFrame({"HoleID": ["A"], "Easting": [0.0], "Northing": [0.0], "RL": [20.0]})
    iv = pd.DataFrame({"HoleID": ["A", "A"], "From": [0, 2], "To": [2, 6], "Lith": ["G-V", "G-IV"]})
    return BoreholeSet.from_tables(col, iv)


def test_workbook_tests_and_stats():
    sheets = {"SPT_samples": pd.DataFrame({"HoleID": ["A", "A"], "Test": ["MI-1", "SPT-1"], "From": [1.0, 3.0],
                                           "To": [1.6, 3.2], "Blows": ["5-6-7-8", "50R"], "N_SPT": [13, "R"]}),
              "Lab_tests": pd.DataFrame({"HoleID": ["A"], "Sample": ["MI-1"], "From": [1.0], "To": [1.6],
                                         "USCS": ["SM"], "LL": [30.0]})}
    t = load_wb_tests(sheets)
    assert list(t["SPT"].n) == [13, 50] and list(t["SPT"].refusal) == [False, True]
    assert t["LAB"].loc[0, "depth_mid"] == pytest.approx(1.3)
    bs = _bs()
    bs.tests = t
    d = with_xyz(bs, t["SPT"])
    assert list(d.unit) == ["G-V", "G-IV"] and d.z.tolist() == pytest.approx([19.0, 17.0])
    s = spt_by_unit(bs).set_index("unit")
    assert s.loc["G-IV", "n_refusal"] == 1


def test_spt_table_requires_columns():
    with pytest.raises(ValueError):
        spt_from_table(pd.DataFrame({"x": [1]}))
    assert len(lab_from_table(pd.DataFrame({"HoleID": ["A"], "From": [1.0]}))) == 1


def test_parse_orientations():
    from geosurf.model import Stratigraphy, parse_orientations

    st_ = Stratigraphy.from_list(["TV", "G-V", "G-IV"])
    ok = pd.DataFrame({"Este": [1.0], "Norte": [2.0], "Cota": [3.0], "Buzamiento": [30], "DipDir": [90],
                       "Unidad": ["G-V"]})
    o = parse_orientations(ok, st_)
    assert o.G_x[0] == pytest.approx(0.5) and o.G_z[0] == pytest.approx(np.cos(np.radians(30)))
    with pytest.raises(ValueError, match="missing column"):
        parse_orientations(pd.DataFrame({"HoleID": ["A"], "From": [0], "To": [1], "Lith": ["X"]}), st_)
    with pytest.raises(ValueError, match="no rows match"):
        parse_orientations(ok.assign(Unidad="G-IV"), st_)  # basement has no surface
