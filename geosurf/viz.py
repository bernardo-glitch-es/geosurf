"""Plotly figures for the app (3D model, maps, cross-sections)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .raster import Raster


def _down(r: Raster, max_n: int = 160):
    s = max(1, int(np.ceil(max(r.nx, r.ny) / max_n)))
    X, Y = r.mesh()
    return X[::s, ::s], Y[::s, ::s], r.z[::s, ::s]


def borehole_traces_3d(bs, strat, width: int = 8) -> list[go.Scatter3d]:
    tr = bs.traces()
    out = []
    for unit, g in tr.groupby("unit", sort=False):
        xs, ys, zs, txt = [], [], [], []
        for r in g.itertuples():
            xs += [r.xt, r.xb, None]
            ys += [r.yt, r.yb, None]
            zs += [r.zt, r.zb, None]
            label = f"{r.hole_id}<br>{r.lith} ({r.top:.1f}-{r.base:.1f} m)"
            txt += [label, label, None]
        out.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=f"BH: {unit}",
                                line=dict(color=strat.color(unit) if unit in strat.names else "#555",
                                          width=width), text=txt, hoverinfo="text", legendgroup="bh"))
    spt = bs.tests.get("SPT")
    if spt is not None and len(spt):
        from .insitu import with_xyz

        d = with_xyz(bs, spt)
        if len(d):
            out.append(go.Scatter3d(
                x=d.x, y=d.y, z=d.z, mode="markers", name="SPT N",
                marker=dict(size=5, color=d.n, colorscale="Turbo", cmin=0, cmax=50,
                            symbol=["diamond" if r else "circle" for r in d.refusal],
                            colorbar=dict(title="N SPT", x=1.02, len=0.5)),
                text=[f"{h} SPT @ {z:.2f} m<br>N = {t}" for h, z, t in zip(d.hole_id, d.depth, d.n_text)],
                hoverinfo="text"))
    c = bs.collars
    out.append(go.Scatter3d(x=c.x, y=c.y, z=c.z + 1, mode="text", text=c.hole_id, name="Hole IDs",
                            textfont=dict(size=10), legendgroup="bh"))
    return out


def fig_3d(res, bs=None, surfaces: list[str] | None = None, terrain: bool = True,
           terrain_opacity: float = 0.35, surface_opacity: float = 0.9, vexag: float = 3.0,
           height: int = 700) -> go.Figure:
    fig = go.Figure()
    sc = res.surfaces_clipped
    names = surfaces if surfaces is not None else [k for k in sc if not k.startswith("_")]
    for n in names:
        X, Y, Z = _down(sc[n])
        col = res.strat.color(n)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, name=f"base {n}", showscale=False, opacity=surface_opacity,
                                 colorscale=[[0, col], [1, col]], showlegend=True,
                                 hovertemplate=f"base {n}<br>x=%{{x:.1f}}<br>y=%{{y:.1f}}<br>z=%{{z:.2f}}<extra></extra>",
                                 lighting=dict(ambient=0.6, diffuse=0.7, roughness=0.9, specular=0.1)))
    if terrain and "_topography" in sc:
        X, Y, Z = _down(sc["_topography"])
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, name="LiDAR terrain", opacity=terrain_opacity,
                                 colorscale="Greys", showscale=False, showlegend=True,
                                 hovertemplate="terrain z=%{z:.2f}<extra></extra>"))
    if bs is not None:
        for t in borehole_traces_3d(bs, res.strat):
            fig.add_trace(t)
    xmin, xmax, ymin, ymax, *_ = res.extent
    zr = [np.nanmin([np.nanmin(sc[n].z) for n in names] or [0]), np.nanmax(sc["_topography"].z)
          if "_topography" in sc else res.extent[5]]
    dx, dy, dz = xmax - xmin, ymax - ymin, (zr[1] - zr[0]) * vexag
    m = max(dx, dy)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30, b=0),
                      scene_camera=dict(eye=dict(x=-1.1, y=-1.5, z=0.9)),
                      scene=dict(aspectmode="manual", aspectratio=dict(x=dx / m, y=dy / m, z=dz / m),
                                 xaxis_title="Easting", yaxis_title="Northing", zaxis_title="Elevation (m)"),
                      legend=dict(itemsizing="constant"))
    return fig


def fig_map(r: Raster, title: str = "", colorscale="Viridis", collars=None, contour: float | None = None,
            zmin=None, zmax=None, unit_label: str = "m", height: int = 520) -> go.Figure:
    fig = go.Figure(go.Heatmap(x=r.xc, y=r.yc, z=r.z, colorscale=colorscale, zmin=zmin, zmax=zmax,
                               colorbar=dict(title=unit_label),
                               hovertemplate="x=%{x:.1f}<br>y=%{y:.1f}<br>%{z:.2f}<extra></extra>"))
    if contour:
        fig.add_trace(go.Contour(x=r.xc, y=r.yc, z=r.z, contours=dict(coloring="none", showlabels=True,
                                 start=np.nanmin(r.z), end=np.nanmax(r.z), size=contour),
                                 line=dict(width=0.6, color="rgba(0,0,0,0.5)"), showscale=False,
                                 hoverinfo="skip"))
    if collars is not None:
        fig.add_trace(go.Scatter(x=collars.x, y=collars.y, mode="markers+text", text=collars.hole_id,
                                 textposition="top center", marker=dict(color="black", size=6, symbol="circle"),
                                 textfont=dict(size=9), name="boreholes"))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
                      yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig


def fig_plan_lines(base: Raster, collars, lines: dict[str, np.ndarray], height: int = 480) -> go.Figure:
    fig = fig_map(base, "Plan: section lines", colorscale="Greys", collars=collars, height=height,
                  unit_label="z")
    for name, p in lines.items():
        p = np.asarray(p)
        fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines+markers+text", name=name,
                                 text=[f"{name}" if i == 0 else (f"{name}'" if i == len(p) - 1 else "")
                                       for i in range(len(p))], textposition="bottom left",
                                 line=dict(color="crimson", width=3)))
    return fig


def fig_section(sec: dict, strat, bs=None, vexag: float = 2.0, height: int = 480) -> go.Figure:
    names = sec["unit_names"]
    fig = go.Figure()
    if sec["block"] is not None:
        n = len(names)
        cs = []
        for i, nm in enumerate(names):
            c = strat.color(nm)
            cs += [[i / n, c], [(i + 1) / n, c]]
        fig.add_trace(go.Heatmap(x=sec["chainage"], y=sec["z"], z=sec["block"], zmin=0.5, zmax=n + 0.5,
                                 colorscale=cs, showscale=False, opacity=0.85,
                                 customdata=np.vectorize(lambda v: names[int(v) - 1] if np.isfinite(v) else "")(sec["block"]),
                                 hovertemplate="ch=%{x:.1f} m<br>z=%{y:.2f}<br>%{customdata}<extra></extra>"))
    for k, v in sec["profiles"].items():
        fig.add_trace(go.Scatter(x=sec["chainage"], y=v, mode="lines", name=f"base {k}",
                                 line=dict(color=strat.color(k), width=2.5, dash="solid")))
    if sec["terrain"] is not None:
        fig.add_trace(go.Scatter(x=sec["chainage"], y=sec["terrain"], mode="lines", name="LiDAR terrain",
                                 line=dict(color="black", width=2)))
    if bs is not None and sec["holes"]:
        iv = bs.intervals
        for h in sec["holes"]:
            hid, ch = h["hole_id"], h["chainage"]
            g = iv[iv.hole_id == hid]
            for r in g.itertuples():
                zt, zb = bs.desurvey(hid, [r.top, r.base])[:, 2]
                fig.add_trace(go.Scatter(x=[ch, ch], y=[zt, zb], mode="lines", showlegend=False,
                                         line=dict(color=strat.color(r.unit) if r.unit in strat.names else "#555",
                                                   width=9),
                                         hovertext=f"{hid} (offset {h['offset']:.1f} m)<br>{r.lith}: {r.top:.1f}-{r.base:.1f} m",
                                         hoverinfo="text"))
            ztop = bs.collars.set_index("hole_id").loc[hid, "z"]
            fig.add_annotation(x=ch, y=ztop, text=hid, showarrow=False, yshift=12, font=dict(size=10))
            for tname, tdf in bs.tests.items():
                if tname != "SPT" or tdf is None:
                    continue
                t = tdf[tdf.hole_id == hid]
                if t.empty:
                    continue
                zt = bs.desurvey(hid, t["depth"].values)[:, 2]
                fig.add_trace(go.Scatter(x=[ch] * len(t), y=zt, mode="markers+text", showlegend=False,
                                         marker=dict(symbol="triangle-left", size=8, color="black"),
                                         text=[f"N={v}" for v in t["n_text"]], textposition="middle right",
                                         textfont=dict(size=9),
                                         hovertext=[f"{hid} SPT {d:.2f} m: N={v}" for d, v in zip(t["depth"], t["n_text"])],
                                         hoverinfo="text"))
    zlo, zhi = float(sec["z"][0]), float(np.nanmax(sec["terrain"]) if sec["terrain"] is not None else sec["z"][-1])
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Chainage (m)",
                      yaxis_title="Elevation (m)",
                      yaxis=dict(scaleanchor="x", scaleratio=vexag, range=[zlo, zhi + 0.08 * (zhi - zlo)]),
                      xaxis=dict(range=[0, float(sec["chainage"][-1])]),
                      legend=dict(orientation="h", y=-0.2))
    return fig
