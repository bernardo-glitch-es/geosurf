"""geosurf - Streamlit interface: LiDAR + boreholes -> GemPy geological surfaces.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geosurf import outputs as out  # noqa: E402
from geosurf import viz  # noqa: E402
from geosurf.ags4 import load_ags4  # noqa: E402
from geosurf.boreholes import REQUIRED, ROLE_COLUMNS, BoreholeSet, guess_columns, read_table  # noqa: E402
from geosurf.lidar import las_info, las_to_dtm, terrain_from_points  # noqa: E402
from geosurf.model import (DEFAULT_COLORS, IGNORE, ModelConfig, Stratigraphy, Unit,  # noqa: E402
                           borehole_misfit, run_pipeline, suggest_stratigraphy)
from geosurf.project import make_project_dict  # noqa: E402
from geosurf.raster import Raster  # noqa: E402

st.set_page_config(page_title="geosurf · GemPy site modeller", page_icon="⛰️", layout="wide")
S = st.session_state
WIDE = {"width": "stretch"}


# ============================================================================ helpers
def workdir() -> Path:
    if "work" not in S:
        S.work = Path(tempfile.mkdtemp(prefix="geosurf_"))
    return S.work


def save_upload(f) -> Path:
    p = workdir() / f.name
    p.write_bytes(f.getbuffer())
    return p


def rebuild_boreholes():
    srcs = S.get("sources", [])
    if not srcs:
        S.bs = None
        return
    bs = None
    for s in srcs:
        b = s["bs"]
        b = BoreholeSet(collars=b.collars.copy(), intervals=b.intervals.copy(),
                        survey=None if b.survey is None else b.survey.copy(), tests=dict(b.tests))
        bs = b if bs is None else bs.merge(b)
    S.bs = bs
    S.drape_report = None
    S.res = None
    if S.get("terrain") is not None:  # fill missing collar z from LiDAR automatically
        dm = S.get("drape_mode", {"mode": "missing", "tolerance": 0.5})
        if dm["mode"] != "none":
            S.drape_report = bs.drape(S.terrain, mode=dm["mode"], tolerance=dm["tolerance"])


def apply_map():
    bs = S.get("bs")
    if bs is not None:
        m = S.get("lith_map", {})
        bs.intervals["unit"] = bs.intervals["lith"].map(lambda c: m.get(c, c))


def strat_from_df(df: pd.DataFrame) -> Stratigraphy:
    df = df.dropna(subset=["unit"])
    df = df[df["unit"].astype(str).str.strip() != ""].sort_values("order")
    units = [Unit(str(r.unit), str(r.group or r.unit), r.color or None) for r in df.itertuples()]
    return Stratigraphy(units, {str(g): "erode" for g in df["group"].unique()})


def load_demo():
    from geosurf.synthetic import CRS, DEMO_LITH_MAP, DEMO_UNITS, make_site

    folder = workdir() / "demo"
    with st.spinner("Generating synthetic site (LiDAR point cloud, DEM, boreholes, AGS4)..."):
        files = make_site(folder)
        dtm = las_to_dtm(files["lidar_laz"], cell=2.0, crs=CRS)
    S.terrain = dtm
    S.terrain_spec = {"path": str(files["lidar_laz"]), "cell": 2.0, "stat": "mean"}
    S.crs = CRS
    b1 = BoreholeSet.from_tables(pd.read_csv(files["collars_csv"]), pd.read_csv(files["lithology_csv"]),
                                 pd.read_csv(files["survey_csv"]))
    b2 = load_ags4(files["ags4"])
    S.sources = [
        {"label": "CSV tables (collars / lithology / survey)", "bs": b1,
         "spec": {"type": "tables", "collars": str(files["collars_csv"]), "intervals": str(files["lithology_csv"]),
                  "survey": str(files["survey_csv"])}},
        {"label": "AGS4: site_investigation.ags", "bs": b2,
         "spec": {"type": "ags4", "path": str(files["ags4"])}}]
    rebuild_boreholes()
    S.lith_map = dict(DEMO_LITH_MAP)
    S.units_df = pd.DataFrame({"order": range(len(DEMO_UNITS)), "unit": [u for u, _ in DEMO_UNITS],
                               "description": [d for _, d in DEMO_UNITS],
                               "group": [u for u, _ in DEMO_UNITS],
                               "color": DEFAULT_COLORS[: len(DEMO_UNITS)]})
    apply_map()
    x0, x1, y0, y1 = dtm.bounds
    S.sections = {"A": [[x0 + 50, y0 + 50], [x0 + 300, y1 - 50], [x1 - 20, y0 + 300]],
                  "B": [[x0 + 20, y0 + 250], [x1 - 20, y0 + 250]]}
    S.demo_truth = folder


# ============================================================================ sidebar
with st.sidebar:
    st.title("⛰️ geosurf")
    st.caption("LiDAR + borehole data → GemPy implicit geological model → surfaces for CAD/GIS")
    if st.button("Load synthetic demo site", type="primary", **WIDE):
        load_demo()
        st.rerun()
    st.divider()
    t = S.get("terrain")
    bs = S.get("bs")
    st.markdown("**Status**")
    st.markdown(("✅" if t is not None else "⬜") + " Terrain" + (f" — {t.nx}×{t.ny} @ {t.dx:g} m" if t is not None else ""))
    st.markdown(("✅" if bs is not None else "⬜") + " Boreholes" + (f" — {len(bs.collars)} holes" if bs is not None else ""))
    st.markdown(("✅" if S.get("units_df") is not None else "⬜") + " Stratigraphy")
    st.markdown(("✅" if S.get("res") is not None else "⬜") + " Model")
    st.divider()
    S.crs = st.text_input("Coordinate reference system", S.get("crs", "EPSG:25829"),
                          help="Written to GeoTIFF outputs, e.g. EPSG:25829 (ETRS89 / UTM 29N), EPSG:25830, EPSG:32637")
    if st.button("Reset session", **WIDE):
        for k in list(S.keys()):
            del S[k]
        st.rerun()

tabs = st.tabs(["1 · Terrain (LiDAR)", "2 · Boreholes", "3 · Stratigraphy & settings", "4 · Model & 3D",
                "5 · Maps", "6 · Sections", "7 · Export"])

# ============================================================================ 1 terrain
with tabs[0]:
    st.subheader("Terrain model from LiDAR")
    c1, c2 = st.columns([1, 2])
    with c1:
        f = st.file_uploader("LAS / LAZ point cloud or GeoTIFF DEM", type=["las", "laz", "tif", "tiff"])
        if f is not None:
            p = save_upload(f)
            if p.suffix.lower() in (".las", ".laz"):
                if S.get("las_path") != str(p):
                    S.las_path, S.las_info = str(p), las_info(p)
                info = S.las_info
                st.write(f"**{info.n_points:,} points** · LAS {info.version} · PDRF {info.point_format}")
                st.caption(f"x {info.bounds[0]:.1f}–{info.bounds[1]:.1f} · y {info.bounds[2]:.1f}–{info.bounds[3]:.1f} · "
                           f"z {info.bounds[4]:.1f}–{info.bounds[5]:.1f}")
                names = {0: "never classified", 1: "unclassified", 2: "ground", 3: "low veg", 4: "medium veg",
                         5: "high veg", 6: "building", 7: "noise", 9: "water", 17: "bridge", 18: "high noise"}
                st.dataframe(pd.DataFrame({"class": list(info.classes), "name": [names.get(k, "") for k in info.classes],
                                           "points": list(info.classes.values())}), hide_index=True, **WIDE)
                default = [2] if 2 in info.classes else list(info.classes)
                cls = st.multiselect("Classes used for the DTM", list(info.classes), default=default,
                                     help="ASPRS class 2 = ground. Vegetation/buildings should be excluded.")
                cell = st.number_input("DTM cell size (m)", 0.25, 50.0, 2.0, 0.25)
                stat = st.selectbox("Cell statistic", ["mean", "min", "max"],
                                    help="'min' is more robust if ground classification left some vegetation")
                if st.button("Build DTM", type="primary"):
                    with st.spinner("Gridding point cloud..."):
                        S.terrain = las_to_dtm(p, cell=cell, classes=cls or None, stat=stat, crs=S.crs)
                    S.terrain_spec = {"path": p.name, "cell": cell, "classes": cls, "stat": stat}
                    S.res = None
                    st.rerun()
            else:
                if st.button("Load DEM", type="primary"):
                    r = Raster.from_geotiff(p)
                    r.crs = r.crs or S.crs
                    S.terrain, S.terrain_spec, S.res = r, {"path": p.name}, None
                    st.rerun()
        if S.get("terrain") is None:
            with st.expander("No LiDAR yet? Build a provisional terrain from borehole collars"):
                st.caption("Linear interpolation of collar elevations (Z/RL). Coarse – replace with LiDAR/DTM when available. "
                           "Load boreholes first (tab 2).")
                cc1, cc2 = st.columns(2)
                pc = cc1.number_input("Cell (m)", 0.25, 20.0, 1.0, 0.25, key="pt_cell")
                pp = cc2.number_input("Padding (m)", 0.0, 500.0, 10.0, 5.0, key="pt_pad")
                bsx = S.get("bs")
                if st.button("Build terrain from collars", disabled=bsx is None):
                    c = bsx.collars
                    S.terrain = terrain_from_points(c[["x", "y", "z"]].to_numpy(float), cell=pc, pad=pp, crs=S.crs)
                    S.terrain_spec = None
                    S.res = None
                    st.rerun()
        t = S.get("terrain")
        if t is not None:
            st.success(f"Terrain: {t.nx}×{t.ny} cells @ {t.dx:g} m · z {np.nanmin(t.z):.1f}–{np.nanmax(t.z):.1f} m")
            if "coverage" in t.meta:
                st.caption(f"Cells with LiDAR returns: {t.meta['coverage']:.1%} (gaps interpolated)")
            st.download_button("Download DTM (GeoTIFF)", t.to_geotiff(workdir() / "dtm.tif").read_bytes(),
                               "dtm.tif")
    with c2:
        t = S.get("terrain")
        if t is not None:
            bsx = S.get("bs")
            fig = viz.fig_map(t.resample(max(t.dx, max(t.nx, t.ny) * t.dx / 400)), "LiDAR terrain (DTM)",
                              colorscale="Earth", collars=None if bsx is None else bsx.collars, contour=5.0)
            st.plotly_chart(fig, **WIDE)
        else:
            st.info("Upload a LiDAR point cloud (.las/.laz) or a DEM (.tif), or load the synthetic demo site from the sidebar.")

# ============================================================================ 2 boreholes
with tabs[1]:
    st.subheader("Borehole / geotechnical data")
    S.setdefault("sources", [])
    with st.expander("➕ Add a data source", expanded=not S.sources):
        kind = st.radio("Format", ["geosurf workbook (.xlsx with 'Collars' + 'Lithology' [+ 'Survey'] sheets)",
                                   "CSV/Excel tables (collars + intervals [+ survey])",
                                   "Single table (one row per interval with coordinates)", "AGS4"],
                        horizontal=False)

        def mapping_ui(df: pd.DataFrame, role: str, key: str) -> dict:
            g = guess_columns(df, role)
            cols = ["—"] + list(df.columns)
            m = {}
            cc = st.columns(len(ROLE_COLUMNS[role]))
            for i, k in enumerate(ROLE_COLUMNS[role]):
                lbl = k + (" *" if k in REQUIRED[role] else "")
                v = cc[i].selectbox(lbl, cols, index=cols.index(g[k]) if g.get(k) in cols else 0, key=f"{key}_{k}")
                m[k] = None if v == "—" else v
            return m

        def table_input(label, key):
            f = st.file_uploader(label, type=["csv", "txt", "xlsx", "xls"], key=key)
            if f is None:
                return None, None, None
            p = save_upload(f)
            data = read_table(p)
            sheet = None
            if isinstance(data, dict):
                sheet = st.selectbox(f"Sheet ({f.name})", list(data), key=key + "_sheet")
                data = data[sheet]
            st.dataframe(data.head(5), hide_index=True, **WIDE)
            return data, p, sheet

        if kind.startswith("geosurf"):
            f = st.file_uploader("Workbook (.xlsx)", type=["xlsx", "xlsm"], key="up_wb")
            if f is not None:
                p = save_upload(f)
                sheets = read_table(p)
                st.caption("Sheets: " + ", ".join(sheets))
                if st.button("Add workbook", type="primary"):
                    try:
                        sv = sheets.get("Survey")
                        b = BoreholeSet.from_tables(sheets["Collars"], sheets["Lithology"], sv)
                        from geosurf.insitu import tests_from_workbook

                        b.tests = tests_from_workbook(sheets)
                        missing = sheets["Collars"].iloc[:, 0][~sheets["Collars"].iloc[:, 0].astype(str).isin(b.hole_ids)]
                        S.sources.append({"label": f"Workbook: {p.name}", "bs": b,
                                          "spec": {"type": "tables", "collars": p.name, "intervals": p.name,
                                                   "survey": p.name if sv is not None else None,
                                                   "collars_sheet": "Collars", "intervals_sheet": "Lithology",
                                                   "survey_sheet": "Survey"}})
                        if len(missing):
                            st.session_state.wb_warning = ("Skipped (no X/Y coordinates): " + ", ".join(map(str, missing)))
                        rebuild_boreholes()
                        st.rerun()
                    except KeyError as e:
                        st.error(f"Sheet {e} not found – expected 'Collars' and 'Lithology'.")
                    except Exception as e:
                        st.error(str(e))
        elif kind.startswith("CSV"):
            dc, pc, sc = table_input("Collars (hole id, x, y, [z, depth, azimuth, dip])", "up_col")
            mc = mapping_ui(dc, "collars", "mc") if dc is not None else None
            di, pi, si = table_input("Intervals / lithology (hole id, from, to, lithology code)", "up_iv")
            mi = mapping_ui(di, "intervals", "mi") if di is not None else None
            ds, ps, ss = table_input("Survey (optional, for inclined holes)", "up_sv")
            ms = mapping_ui(ds, "survey", "ms") if ds is not None else None
            if dc is not None and di is not None and st.button("Add boreholes", type="primary"):
                try:
                    b = BoreholeSet.from_tables(dc, di, ds, collar_map=mc, interval_map=mi, survey_map=ms)
                    S.sources.append({"label": f"Tables: {pc.name} + {pi.name}", "bs": b,
                                      "spec": {"type": "tables", "collars": pc.name, "intervals": pi.name,
                                               "survey": ps.name if ps else None, "collars_sheet": sc or 0,
                                               "intervals_sheet": si or 0, "survey_sheet": ss or 0,
                                               "collar_map": mc, "interval_map": mi, "survey_map": ms}})
                    rebuild_boreholes()
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        elif kind.startswith("Single"):
            df, pf, sf = table_input("Interval table with coordinates", "up_flat")
            if df is not None:
                mf = mapping_ui(df, "flat", "mf")
                if st.button("Add boreholes", type="primary"):
                    try:
                        b = BoreholeSet.from_tables(None, df, interval_map=mf, flat=True)
                        S.sources.append({"label": f"Table: {pf.name}", "bs": b,
                                          "spec": {"type": "flat", "path": pf.name, "sheet": sf or 0, "map": mf}})
                        rebuild_boreholes()
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
        else:
            f = st.file_uploader("AGS4 file (.ags)", type=["ags", "txt"], key="up_ags")
            lf = st.selectbox("Lithology code field", ["auto", "GEOL_GEOL", "GEOL_LEG", "GEOL_GEO2", "GEOL_DESC"])
            if f is not None and st.button("Add AGS4", type="primary"):
                try:
                    p = save_upload(f)
                    b = load_ags4(p, lith_field=lf)
                    S.sources.append({"label": f"AGS4: {p.name}", "bs": b,
                                      "spec": {"type": "ags4", "path": p.name, "lith_field": lf}})
                    rebuild_boreholes()
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    if S.get("wb_warning"):
        st.warning(S.wb_warning)
    for i, s in enumerate(S.sources):
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{s['label']}** — {len(s['bs'].collars)} holes, {len(s['bs'].intervals)} intervals")
        if c2.button("Remove", key=f"rm{i}"):
            S.sources.pop(i)
            rebuild_boreholes()
            st.rerun()

    bs = S.get("bs")
    if bs is not None:
        apply_map()
        t = S.get("terrain")
        st.markdown("#### Collar elevations vs LiDAR")
        c1, c2, c3 = st.columns([2, 1, 1])
        mode = c1.radio("Use LiDAR elevation for", ["missing collar z only", "all collars", "don't drape"],
                        horizontal=True)
        tol = c2.number_input("Flag |Δz| above (m)", 0.05, 20.0, 0.5, 0.05)
        if c3.button("Apply", disabled=t is None):
            rebuild_boreholes()
            if not mode.startswith("don"):
                S.drape_report = S.bs.drape(t, mode="all" if mode.startswith("all") else "missing", tolerance=tol)
            S.drape_mode = {"mode": "all" if mode.startswith("all") else ("none" if mode.startswith("don") else "missing"),
                            "tolerance": tol}
            st.rerun()
        if S.get("drape_report") is not None:
            r = S.drape_report.copy()
            r["flag"] = np.where(r["dz_terrain"].abs() > tol, f"|Δz| > {tol} m", "")
            st.dataframe(r.round(3), hide_index=True, **WIDE, height=240)
        elif t is None:
            st.caption("Load terrain first to check/fill collar elevations from LiDAR.")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("#### Borehole summary")
            st.dataframe(bs.summary().round(2), hide_index=True, **WIDE, height=320)
        with c2:
            st.markdown("#### Log QA")
            qa = bs.qa()
            if qa.empty:
                st.success("No gaps, overlaps or depth inconsistencies found.")
            else:
                st.dataframe(qa, hide_index=True, **WIDE)
            for k, v in bs.tests.items():
                st.caption(f"Tests loaded: {k} ({len(v)} records)")
        if bs.tests:
            from geosurf.insitu import spt_by_unit

            st.markdown("#### In-situ and laboratory tests")
            ttabs = st.tabs(list(bs.tests) + (["SPT by unit"] if "SPT" in bs.tests else []))
            for tt, (k, v) in zip(ttabs, bs.tests.items()):
                with tt:
                    vv = v.copy()
                    for c in vv.columns[vv.dtypes == object]:
                        vv[c] = vv[c].map(lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x))
                    st.dataframe(vv, hide_index=True, **WIDE, height=260)
            if "SPT" in bs.tests:
                with ttabs[-1]:
                    st.caption("Refusals (R, 50R, >50) counted as N = 50. Units from the logged intervals.")
                    st.dataframe(spt_by_unit(bs).round(1), hide_index=True, **WIDE)
        with st.expander("Interval table"):
            st.dataframe(bs.intervals, hide_index=True, **WIDE)

# ============================================================================ 3 stratigraphy
with tabs[2]:
    bs = S.get("bs")
    if bs is None:
        st.info("Load borehole data first.")
    else:
        st.subheader("Map lithology codes to model units")
        S.setdefault("lith_map", {})
        codes = bs.lith_codes
        cur = pd.DataFrame({"code": codes,
                            "intervals": [int((bs.intervals.lith == c).sum()) for c in codes],
                            "example description": [bs.intervals.loc[bs.intervals.lith == c, "desc"].iloc[0] for c in codes],
                            "model unit": [S.lith_map.get(c, c) for c in codes]})
        ed = st.data_editor(cur, hide_index=True, disabled=["code", "intervals", "example description"], **WIDE,
                            column_config={"model unit": st.column_config.TextColumn(
                                help=f"Several codes can map to one unit. Use {IGNORE} for core loss, voids, etc.")},
                            key="lith_editor")
        S.lith_map = dict(zip(ed["code"], ed["model unit"].fillna(IGNORE)))
        apply_map()
        units_present = [u for u in pd.unique(bs.intervals["unit"]) if u != IGNORE]
        st.subheader("Stratigraphic column (top → bottom)")
        c1, c2 = st.columns([3, 2])
        with c1:
            mode = st.radio("Interpolation strategy",
                            ["Independent surfaces (one GemPy series per contact)", "Conformable (one series, layer-cake)",
                             "Custom groups (edit the 'group' column)"],
                            help="Independent: each contact is interpolated on its own and stacked by erosion — "
                                 "recommended for soils, fills and weathering profiles with variable thickness. "
                                 "Conformable: all contacts share one scalar field (parallel layering).")
            if st.button("Suggest order from boreholes") or S.get("units_df") is None or \
                    set(S.units_df["unit"]) != set(units_present):
                order = [u for u in suggest_stratigraphy(bs) if u in units_present]
                S.units_df = pd.DataFrame({"order": range(len(order)), "unit": order, "description": "",
                                           "group": order, "color": [DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                                                                     for i in range(len(order))]})
            udf = S.units_df.copy()
            if mode.startswith("Indep"):
                udf["group"] = udf["unit"]
            elif mode.startswith("Conf"):
                udf["group"] = "Strata"
            udf = st.data_editor(udf.sort_values("order"), hide_index=True, **WIDE, key="units_editor",
                                 column_config={"order": st.column_config.NumberColumn(help="0 = top (youngest)"),
                                                "color": st.column_config.TextColumn(help="hex colour"),
                                                "group": st.column_config.TextColumn(
                                                    help="GemPy series. Units sharing a group share a scalar field.")})
            S.units_df = udf
            strat = strat_from_df(udf)
            st.caption(f"Basement (lowest unit, no surface of its own): **{strat.basement}** · "
                       f"surfaces to model: {', '.join('base ' + u.name for u in strat.surface_units)}")
        with c2:
            # simple stratigraphic column graphic
            iv = bs.intervals[bs.intervals.unit != IGNORE]
            th = (iv.base - iv.top).groupby(iv.unit).mean()
            html = "".join(f"<div style='background:{strat.color(u.name)};padding:6px 10px;border:1px solid #444;"
                           f"min-height:{max(22, min(90, 8 * th.get(u.name, 3)))}px;color:#111'>"
                           f"<b>{u.name}</b> <small>{udf.set_index('unit').loc[u.name, 'description'] or ''} · "
                           f"mean {th.get(u.name, np.nan):.1f} m</small></div>" for u in strat.units)
            st.markdown(html, unsafe_allow_html=True)
        st.subheader("Model settings")
        c1, c2, c3 = st.columns(3)
        t = S.get("terrain")
        with c1:
            nx = st.slider("Grid cells X / Y", 20, 120, 50, 5)
            nz = st.slider("Grid cells Z", 20, 150, 60, 5)
            out_cell = st.number_input("Output surface cell (m)", 0.5, 50.0, float(t.dx) if t is not None else 5.0, 0.5)
        with c2:
            alpha = st.slider("LiDAR terrain trend (0 = elevation, 1 = depth below ground)", 0.0, 1.0,
                              1.0 if t is not None else 0.0, 0.1, disabled=t is None,
                              help="Interpolates contacts as depth below the LiDAR surface so shallow units follow "
                                   "the terrain between boreholes. Use 0 for deep/structural horizons independent of topography.")
            smooth = st.number_input("Trend smoothing (m)", 0.0, 500.0, 0.0, 5.0,
                                     help="Gaussian smoothing of the terrain used as trend (removes micro-relief, fills, buildings)")
            pad_z = st.number_input("Vertical padding (m)", 0.0, 200.0, 10.0, 5.0)
        with c3:
            pin = st.selectbox("Absent / pinched-out units", ["zero", "skip"],
                               format_func=lambda v: {"zero": "zero thickness at the hole", "skip": "no constraint"}[v])
            nug = st.select_slider("Surface point nugget (smoothing)", [1e-6, 2e-5, 1e-4, 1e-3, 1e-2], 2e-5,
                                   format_func=lambda v: f"{v:g}")
            k = st.select_slider("Local orientation fits (k neighbours, 0 = off)", [0, 4, 5, 6, 8, 10], 0)
        of = st.file_uploader("Optional structural measurements CSV (x, y, z, dip, azimuth, unit)", type=["csv"])
        S.extra_ori = None
        if of is not None:
            from geosurf.model import parse_orientations

            try:
                raw = pd.read_csv(of, sep=None, engine="python")
                ori = parse_orientations(raw, strat)
                S.extra_ori = ori
                ign = ori.attrs.get("ignored_units") or []
                st.success(f"{len(ori)} structural measurements will be used"
                           + (f" (ignored units: {', '.join(ign)})" if ign else ""))
            except Exception as e:
                st.error(f"Structural measurements not used – {e}")
        S.cfg = ModelConfig(resolution=(nx, nx, nz), pad_z=pad_z, nugget=nug, pinchout=pin,
                            local_orientations_k=k, trend_alpha=alpha if t is not None else 0.0, trend_smooth=smooth)
        S.out_cell = out_cell
        S.strat = strat

# ============================================================================ 4 model
with tabs[3]:
    bs, strat = S.get("bs"), S.get("strat")
    if bs is None or strat is None:
        st.info("Load boreholes and define the stratigraphy first (tabs 2 and 3).")
    else:
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("▶ Compute GemPy model", type="primary", **WIDE):
                t0 = time.time()
                with st.spinner("Extracting contacts, interpolating with GemPy, extracting surfaces..."):
                    try:
                        S.res = run_pipeline(bs, strat, S.get("terrain"), S.cfg,
                                             extra_orientations=S.get("extra_ori"), output_cell=S.out_cell)
                        S.res_time = time.time() - t0
                        S.pop("export_zip", None)
                        S.pop("sec_key", None)
                    except Exception as e:
                        S.res = None
                        st.exception(e)
                if S.get("res") is not None:
                    st.rerun()
        res = S.get("res")
        if res is not None:
            with c1:
                st.success(f"Computed in {S.res_time:.1f} s")
                st.metric("Interface points", len(res.points))
                st.metric("Orientations", len(res.orientations))
                mf = borehole_misfit(res)
                st.metric("Max |misfit| at contacts", f"{mf.error.abs().max():.2f} m" if len(mf) else "—")
            with c2:
                cc = st.columns(5)
                names = [k for k in res.surfaces_clipped if not k.startswith("_")]
                show = cc[0].multiselect("Surfaces", names, names[1:] if len(names) > 1 else names,
                                         help="The uppermost contact often coincides with the terrain where the unit is absent")
                vex = cc[1].slider("Vertical exaggeration", 1.0, 10.0, 3.0, 0.5)
                sop = cc[2].slider("Surface opacity", 0.1, 1.0, 0.85, 0.05)
                top = cc[3].slider("Terrain opacity", 0.0, 1.0, 0.25, 0.05)
                show_bh = cc[4].checkbox("Boreholes", True)
                st.plotly_chart(viz.fig_3d(res, bs if show_bh else None, show, terrain=top > 0, terrain_opacity=top,
                                           surface_opacity=sop, vexag=vex, height=720), **WIDE)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Misfit at borehole contacts** (model − logged)")
                st.dataframe(mf.round(3), hide_index=True, **WIDE, height=260)
            with c2:
                st.markdown("**Interface points**")
                st.dataframe(res.points.round(2), hide_index=True, **WIDE, height=260)
            with c3:
                st.markdown("**Warnings**")
                if res.issues.empty:
                    st.success("No sequence issues.")
                else:
                    st.dataframe(res.issues, hide_index=True, **WIDE, height=260)
            st.caption("Points: *contact* = logged base of unit; *absent-top* / *pinch-out* = zero-thickness "
                       "constraints where a unit is missing in a hole.")

# ============================================================================ 5 maps
with tabs[4]:
    res = S.get("res")
    if res is None:
        st.info("Compute the model first.")
    else:
        th, dp = out.thickness_maps(res), out.depth_maps(res)
        opts = {f"Elevation · base {k}": (r, "Viridis", "m") for k, r in res.surfaces_clipped.items() if not k.startswith("_")}
        opts.update({f"Thickness (isopach) · {k}": (r, "YlOrBr", "m") for k, r in th.items()})
        opts.update({f"Depth below ground · base {k}": (r, "Blues", "m") for k, r in dp.items()})
        if "_topography" in res.surfaces_clipped:
            opts["Elevation · LiDAR terrain"] = (res.surfaces_clipped["_topography"], "Earth", "m")
        opts["Distance to nearest borehole (confidence)"] = (out.distance_to_data(res), "Reds", "m")
        c1, c2 = st.columns([3, 1])
        with c2:
            sel = st.radio("Map", list(opts))
            ci = st.number_input("Contour interval (m, 0 = none)", 0.0, 50.0, 1.0, 0.5)
            st.markdown("**Volumes**")
            st.dataframe(out.volumes(res).round(1), hide_index=True, **WIDE)
        with c1:
            r, cs, lab = opts[sel]
            rr = r.resample(max(r.dx, max(r.nx, r.ny) * r.dx / 350))
            st.plotly_chart(viz.fig_map(rr, sel, colorscale=cs, collars=S.bs.collars, contour=ci or None,
                                        unit_label=lab, height=620), **WIDE)

# ============================================================================ 6 sections
with tabs[5]:
    res = S.get("res")
    if res is None:
        st.info("Compute the model first.")
    else:
        S.setdefault("sections", {})
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Section lines** (vertices, in model coordinates)")
            rows = [{"section": n, "x": p[0], "y": p[1]} for n, pts in S.sections.items() for p in pts]
            df = pd.DataFrame(rows or [{"section": "A", "x": res.extent[0], "y": (res.extent[2] + res.extent[3]) / 2},
                                       {"section": "A", "x": res.extent[1], "y": (res.extent[2] + res.extent[3]) / 2}])
            ed = st.data_editor(df, num_rows="dynamic", hide_index=True, **WIDE, key="sec_editor")
            S.sections = {n: g[["x", "y"]].to_numpy(float).tolist() for n, g in ed.dropna().groupby("section", sort=False)
                          if len(g) >= 2}
            base = res.surfaces_clipped.get("_topography") or next(iter(res.surfaces_clipped.values()))
            st.plotly_chart(viz.fig_plan_lines(base.resample(max(base.dx, base.nx * base.dx / 250)), S.bs.collars,
                                               S.sections, height=420), **WIDE)
        with c2:
            if S.sections:
                name = st.selectbox("Section", list(S.sections))
                cc = st.columns(3)
                buf = cc[0].number_input("Borehole projection buffer (m)", 0.0, 500.0, 25.0, 5.0)
                vex = cc[1].slider("Vertical exaggeration ", 1.0, 10.0, 3.0, 0.5)
                blk = cc[2].checkbox("GemPy lithology block", True, help="Evaluates the 3D model on the section plane")
                key = (name, tuple(map(tuple, S.sections[name])), buf, blk, id(res))
                if S.get("sec_key") != key:
                    with st.spinner("Evaluating section..."):
                        S.sec = out.cross_section(res, S.sections[name], buffer=buf, with_block=blk)
                        S.sec_key = key
                st.plotly_chart(viz.fig_section(S.sec, res.strat, S.bs, vexag=vex), **WIDE)
                d = pd.DataFrame({"chainage": S.sec["chainage"], "x": S.sec["xy"][:, 0], "y": S.sec["xy"][:, 1]})
                if S.sec["terrain"] is not None:
                    d["terrain"] = S.sec["terrain"]
                for k, v in S.sec["profiles"].items():
                    d[f"base_{k}"] = v
                st.download_button("Download section profile (CSV)", d.to_csv(index=False).encode(), f"section_{name}.csv")

# ============================================================================ 7 export
with tabs[6]:
    res = S.get("res")
    if res is None:
        st.info("Compute the model first.")
    else:
        st.subheader("Export surfaces and derived products")
        c1, c2 = st.columns(2)
        with c1:
            fm = st.multiselect("Formats", ["tif", "asc", "xyz", "dxf", "obj", "vtk"],
                                ["tif", "asc", "xyz", "dxf", "obj", "vtk"],
                                format_func=lambda f: {"tif": "GeoTIFF grids (GIS)", "asc": "Esri ASCII grids",
                                                       "xyz": "XYZ point CSV", "dxf": "DXF TIN + contours (CAD / Civil 3D)",
                                                       "obj": "OBJ meshes", "vtk": "VTK meshes (ParaView)"}[f])
            ci = st.number_input("DXF contour interval (m, 0 = none)", 0.0, 50.0, 1.0, 0.5)
            mf = st.number_input("Max TIN faces per surface (DXF/OBJ/VTK)", 5_000, 500_000, 40_000, 5_000)
            gm = st.checkbox("Also export raw GemPy dual-contouring meshes (OBJ)", False)
            if st.button("Build export package", type="primary"):
                folder = workdir() / f"export_{int(time.time())}"
                with st.spinner("Writing files..."):
                    files = out.export_all(res, folder, formats=tuple(fm), contour_interval=ci or None,
                                           max_faces=int(mf), sections=S.get("sections"), include_gempy_meshes=gm,
                                           boreholes=S.bs)
                    S.export_zip = out.zip_folder(folder)
                    S.export_list = [str(f.relative_to(folder)) for f in files]
            if S.get("export_zip"):
                st.download_button("⬇ Download ZIP", S.export_zip, "geosurf_export.zip", type="primary")
                st.caption(f"{len(S.export_list)} files · {len(S.export_zip) / 1e6:.1f} MB")
                st.code("\n".join(S.export_list), language=None)
        with c2:
            st.markdown("**Automation: project file**")
            st.caption("Re-run this exact setup from the command line (e.g. when new boreholes or a new LiDAR "
                       "survey arrive): place the project file next to the data files and run "
                       "`python -m geosurf.cli run project.json`.")
            specs = []
            for s in S.get("sources", []):
                sp = dict(s["spec"])
                for k in ("path", "collars", "intervals", "survey"):
                    if sp.get(k):
                        sp[k] = Path(sp[k]).name
                specs.append(sp)
            ts = dict(S.get("terrain_spec") or {})
            if ts.get("path"):
                ts["path"] = Path(ts["path"]).name
            pj = make_project_dict("geosurf_project", S.crs, ts or None, specs, S.get("lith_map", {}), res.strat,
                                   S.cfg, {"folder": "output", "cell": S.out_cell, "formats": fm,
                                           "contour_interval": ci or None, "sections": S.get("sections")},
                                   drape=S.get("drape_mode"))
            txt = json.dumps(pj, indent=2, default=float)
            st.download_button("Download project.json", txt.encode(), "project.json")
            st.code(txt[:3000] + ("\n..." if len(txt) > 3000 else ""), language="json")
