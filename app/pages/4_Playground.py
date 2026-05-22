"""
4_Playground.py
MINBAR Interactive Playground — select sources, columns, and plot type
to visualise any slice of the full MINBAR catalogue live.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Playground", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,300;0,400;0,600;1,400&family=Instrument+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Instrument Sans', sans-serif;
    letter-spacing: -0.01em;
}
h1, h2, h3 {
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.04em;
    font-weight: 600 !important;
}
h1 { font-size: 1.85rem !important; }

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2rem !important;
    color: #e8d5b0;
    letter-spacing: -0.03em;
}
[data-testid="stMetricLabel"] {
    color: #7a7a7a;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
}

div[data-testid="stSidebar"] {
    background: #0a0a0d;
    border-right: 1px solid rgba(255,255,255,0.06);
}
div[data-testid="stSidebar"] * { color: #bbb !important; }

.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #555;
    padding: 0.6rem 0.3rem;
}
.stTabs [aria-selected="true"] {
    color: #e8d5b0 !important;
    border-bottom-color: #e8d5b0 !important;
}

.section-note {
    color: #666;
    font-size: 0.78rem;
    line-height: 1.55;
    border-left: 2px solid #333;
    padding-left: 12px;
    margin: 0.5rem 0 1rem 0;
}

.col-tag {
    display: inline-block;
    background: #1a1a22;
    border: 1px solid #2a2a35;
    border-radius: 3px;
    padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #c8a96e;
    margin: 2px 2px;
}

div.stAlert { background: #111116; border: 1px solid #222; }
</style>
""", unsafe_allow_html=True)

st.title(" Playground")
st.markdown(
    "<p style='color:#777; font-family:Instrument Sans; margin-top:-0.8rem; font-size:0.92rem'>"
    "Free-form exploration — select sources, pick columns, plot.</p>",
    unsafe_allow_html=True,
)

# ── Data loading ──────────────────────────────────────────────────────────────
BASE_DIR = os.environ.get("LMXB_DATA_DIR", ".")
FULL_CSV  = os.path.join(BASE_DIR, "minbar_bursts_full.csv")

@st.cache_data(show_spinner="Loading  catalogue…")
def load_full():
    try:
        with open(FULL_CSV, "r") as _f:
            skip = sum(1 for line in _f if line.startswith("#"))
        df = pd.read_csv(
            FULL_CSV,
            sep="\t",
            skiprows=skip,
            index_col=False,
            quotechar='"',
            na_values=["-", ""],
        )
    except FileNotFoundError:
        fallback = os.path.join(BASE_DIR, "data/raw/minbar/minbar_bursts_web.txt")
        with open(fallback, "r") as _f:
            skip = sum(1 for line in _f if line.startswith("#"))
        df = pd.read_csv(
            fallback,
            sep="\t",
            skiprows=skip,
            index_col=False,
            quotechar='"',
            na_values=["-", ""],
        )
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError):
            pass
    if "time" in df.columns:
        _epoch = pd.Timestamp("1858-11-17")
        df["_date"] = _epoch + pd.to_timedelta(
            pd.to_numeric(df["time"], errors="coerce"), unit="D"
        )
    return df

try:
    df_full = load_full()
except FileNotFoundError:
    st.error(
        f"Data not found at `{FULL_CSV}` or fallback path. "
        "Set the `LMXB_DATA_DIR` environment variable or update `BASE_DIR`."
    )
    st.stop()
except Exception as e:
    st.error(f"Failed to load catalogue: {type(e).__name__}: {e}")
    st.stop()

# ── Column metadata ───────────────────────────────────────────────────────────
_all_cols      = list(df_full.columns)
_numeric_cols  = [c for c in _all_cols if pd.api.types.is_numeric_dtype(df_full[c])]
_cat_cols      = [c for c in _all_cols if not pd.api.types.is_numeric_dtype(df_full[c])
                  and c not in ("_date",)]

_COL_HINTS = {
    "bpflux":  "Bolometric peak flux (10⁻⁹ erg/s/cm²)",
    "bpfluxe": "Uncertainty on peak flux",
    "bfluen":  "Bolometric fluence",
    "tau":     "Fluence / peak-flux ratio (burst duration proxy)",
    "kt":      "Blackbody temperature at peak (keV)",
    "kte":     "Uncertainty on kT",
    "rad":     "Blackbody normalisation (R² / d²)",
    "edt":     "Exponential decay constant (s)",
    "dur":     "Burst duration (s)",
    "tdel":    "Time since previous burst (s)",
    "alpha":   "Ratio persistent flux / burst fluence",
    "perflx":  "Persistent 3–25 keV flux",
    "rise":    "Burst rise time (s)",
    "rexp":    "Photospheric radius-expansion flag",
    "hc":      "Hard colour",
    "sc":      "Soft colour",
    "s_z":     "Position on colour–colour diagram (S_Z)",
    "time":    "Burst start time (MJD)",
    "gamma":   "Persistent flux / peak PRE flux ratio",
}

_all_instrs  = sorted(df_full["instr"].dropna().unique().tolist()) if "instr" in df_full.columns else []
_PALETTE = [
    "#4e79a7","#f28e2b","#e15759","#76b7b2",
    "#59a14f","#edc948","#b07aa1","#ff9da7",
    "#9c755f","#bab0ac",
]
_INSTR_COLORS = {i: _PALETTE[k % len(_PALETTE)] for k, i in enumerate(_all_instrs)}

_LAYOUT_COMMON = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Instrument Sans, sans-serif", size=12),
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Controls")

    st.markdown("##### Sources")
    all_sources = sorted(df_full["name"].dropna().unique().tolist()) if "name" in df_full.columns else []

    src_mode = st.radio(
        "Select by",
        ["All sources", "Pick sources", "Top N by burst count"],
        index=0,
        horizontal=False,
    )

    if src_mode == "Pick sources":
        selected_sources = st.multiselect(
            "Sources", all_sources,
            default=all_sources[:3] if len(all_sources) >= 3 else all_sources,
        )
    elif src_mode == "Top N by burst count":
        top_n = st.slider("N", 1, min(50, len(all_sources)), 10)
        selected_sources = (
            df_full["name"].value_counts().head(top_n).index.tolist()
            if "name" in df_full.columns else all_sources
        )
    else:
        selected_sources = all_sources

    st.markdown("##### Instruments")
    selected_instrs = st.multiselect(
        "Instruments", _all_instrs, default=_all_instrs,
    )

    st.markdown("##### Photospheric radius expansion")
    pre_filter = st.select_slider(
        "PRE filter",
        options=["All bursts", "PRE only (rexp ≥ 1.629)", "PRE only (rexp ≥ 2.0)", "Non-PRE only"],
        value="All bursts",
    )

    st.markdown("##### Numeric filter")
    with st.expander("Range filter", expanded=False):
        filter_col = st.selectbox("Column", _numeric_cols, key="filt_col")
        _col_data  = df_full[filter_col].dropna()
        _fmin, _fmax = float(_col_data.min()), float(_col_data.max())
        if _fmin < _fmax:
            filt_range = st.slider(
                f"{filter_col} range",
                _fmin, _fmax, (_fmin, _fmax),
                key="filt_range",
            )
        else:
            filt_range = (_fmin, _fmax)
        apply_numeric_filter = st.checkbox("Apply", value=False)

    st.markdown("---")
    st.caption(
        f"**Full catalogue:** {len(df_full):,} bursts · "
        f"{df_full['name'].nunique() if 'name' in df_full.columns else '?'} sources"
    )


# ── Apply filters ─────────────────────────────────────────────────────────────
mask = pd.Series(True, index=df_full.index)

if "name" in df_full.columns and selected_sources:
    mask &= df_full["name"].isin(selected_sources)

if "instr" in df_full.columns and selected_instrs:
    mask &= df_full["instr"].isin(selected_instrs)

if "rexp" in df_full.columns:
    if pre_filter == "PRE only (rexp ≥ 1.629)":
        mask &= df_full["rexp"] >= 1.629
    elif pre_filter == "PRE only (rexp ≥ 2.0)":
        mask &= df_full["rexp"] >= 2.0
    elif pre_filter == "Non-PRE only":
        mask &= df_full["rexp"] < 1.629

if apply_numeric_filter and filter_col in df_full.columns:
    mask &= df_full[filter_col].between(*filt_range)

df_sel = df_full[mask].copy()

# ── Summary metrics ───────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Selected bursts",  f"{len(df_sel):,}")
m2.metric("Unique sources",   f"{df_sel['name'].nunique() if 'name' in df_sel.columns else '?'}")
m3.metric("Instruments",      f"{df_sel['instr'].nunique() if 'instr' in df_sel.columns else '?'}")
if "rexp" in df_sel.columns:
    n_pre = int((df_sel["rexp"] >= 1.629).sum())
    m4.metric("PRE bursts", f"{n_pre:,}", delta=f"{n_pre/max(len(df_sel),1)*100:.0f}%", delta_color="off")

if len(df_sel) == 0:
    st.warning("No bursts match the current filters.")
    st.stop()

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_xy, tab_hist, tab_ts, tab_matrix, tab_table = st.tabs([
    "X vs Y",
    "Distribution",
    "Time Series",
    "Correlation Matrix",
    "Data Table",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — X vs Y Scatter
# ══════════════════════════════════════════════════════════════════════════════
with tab_xy:
    st.markdown(
        '<p class="section-note">'
        "Scatter plot of any two numeric columns. "
        "Colour by source, instrument, or PRE flag."
        "</p>",
        unsafe_allow_html=True,
    )

    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])

    _default_x = "bpflux" if "bpflux" in _numeric_cols else _numeric_cols[0]
    _default_y = "kt"     if "kt"     in _numeric_cols else (_numeric_cols[1] if len(_numeric_cols) > 1 else _numeric_cols[0])

    with ctrl1:
        x_col = st.selectbox(
            "X axis", _numeric_cols,
            index=_numeric_cols.index(_default_x),
            format_func=lambda c: f"{c}  —  {_COL_HINTS[c]}" if c in _COL_HINTS else c,
        )
        x_err_col = st.selectbox(
            "X error bar", ["none"] + _numeric_cols,
            index=0,
            key="x_err",
        )
        x_log = st.checkbox("Log X", value=False)

    with ctrl2:
        y_col = st.selectbox(
            "Y axis", _numeric_cols,
            index=_numeric_cols.index(_default_y),
            format_func=lambda c: f"{c}  —  {_COL_HINTS[c]}" if c in _COL_HINTS else c,
        )
        y_err_col = st.selectbox(
            "Y error bar", ["none"] + _numeric_cols,
            index=0,
            key="y_err",
        )
        y_log = st.checkbox("Log Y", value=False)

    with ctrl3:
        color_by = st.selectbox(
            "Colour by",
            ["source", "instrument", "PRE flag", "none"],
            index=0,
        )
        marker_size  = st.slider("Marker size", 2, 16, 5)
        marker_opacity = st.slider("Opacity", 0.05, 1.0, 0.5, step=0.05)
        max_points = st.number_input(
            "Max points (0 = all)", min_value=0, max_value=200_000,
            value=10_000, step=1000,
            help="Down-sample for performance.",
        )

    plot_df = df_sel.dropna(subset=[x_col, y_col]).copy()
    if max_points and len(plot_df) > int(max_points):
        plot_df = plot_df.sample(int(max_points), random_state=42)

    if len(plot_df) == 0:
        st.info("No valid data for the selected columns.")
    else:
        fig_xy = go.Figure()

        def _get_color_groups(cby):
            if cby == "source" and "name" in plot_df.columns:
                return "name", plot_df["name"].unique()
            elif cby == "instrument" and "instr" in plot_df.columns:
                return "instr", plot_df["instr"].unique()
            elif cby == "PRE flag" and "rexp" in plot_df.columns:
                plot_df["_pre_label"] = plot_df["rexp"].apply(
                    lambda v: "PRE" if v >= 1.629 else "Non-PRE"
                )
                return "_pre_label", ["PRE", "Non-PRE"]
            else:
                return None, None

        group_col, groups = _get_color_groups(color_by)

        if group_col is None:
            ex = plot_df[x_err_col].tolist() if x_err_col != "none" else None
            ey = plot_df[y_err_col].tolist() if y_err_col != "none" else None
            fig_xy.add_trace(go.Scattergl(
                x=plot_df[x_col], y=plot_df[y_col],
                mode="markers",
                marker=dict(size=marker_size, color="#c8a96e", opacity=marker_opacity),
                error_x=dict(type="data", array=ex, visible=ex is not None, thickness=0.7) if ex else None,
                error_y=dict(type="data", array=ey, visible=ey is not None, thickness=0.7) if ey else None,
                hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
            ))
        else:
            _palette_ext = _PALETTE * 10
            for idx, grp_val in enumerate(groups):
                sub = plot_df[plot_df[group_col] == grp_val]
                if len(sub) == 0:
                    continue
                if color_by == "instrument":
                    color = _INSTR_COLORS.get(str(grp_val), _palette_ext[idx])
                elif color_by == "PRE flag":
                    color = "#c8a96e" if grp_val == "PRE" else "#3a6186"
                else:
                    color = _palette_ext[idx % len(_palette_ext)]

                ex = sub[x_err_col].tolist() if x_err_col != "none" and x_err_col in sub else None
                ey = sub[y_err_col].tolist() if y_err_col != "none" and y_err_col in sub else None

                fig_xy.add_trace(go.Scattergl(
                    x=sub[x_col], y=sub[y_col],
                    mode="markers",
                    name=str(grp_val),
                    marker=dict(size=marker_size, color=color, opacity=marker_opacity),
                    error_x=dict(type="data", array=ex, visible=True, thickness=0.6, color=color) if ex else None,
                    error_y=dict(type="data", array=ey, visible=True, thickness=0.6, color=color) if ey else None,
                    hovertemplate=(
                        f"<b>{grp_val}</b><br>"
                        f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>"
                    ),
                ))

        fig_xy.update_layout(
            **_LAYOUT_COMMON,
            xaxis=dict(
                title=f"{x_col}" + (f"  —  {_COL_HINTS[x_col]}" if x_col in _COL_HINTS else ""),
                type="log" if x_log else "linear",
            ),
            yaxis=dict(
                title=f"{y_col}" + (f"  —  {_COL_HINTS[y_col]}" if y_col in _COL_HINTS else ""),
                type="log" if y_log else "linear",
            ),
            legend=dict(orientation="v", x=1.01, font_size=10, itemsizing="constant"),
            margin=dict(t=20, b=20, r=200 if group_col else 20),
            height=520,
        )
        st.plotly_chart(fig_xy, use_container_width=True)

        with st.expander("Statistics"):
            stat_cols = st.columns(2)
            for i, col_name in enumerate([x_col, y_col]):
                vals = plot_df[col_name].dropna()
                with stat_cols[i]:
                    st.markdown(f"**{col_name}**")
                    st.dataframe(
                        pd.DataFrame({
                            "stat": ["N", "mean", "median", "std", "min", "max"],
                            "value": [
                                f"{len(vals):,}",
                                f"{vals.mean():.4g}",
                                f"{vals.median():.4g}",
                                f"{vals.std():.4g}",
                                f"{vals.min():.4g}",
                                f"{vals.max():.4g}",
                            ]
                        }),
                        hide_index=True,
                        use_container_width=True,
                    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Distribution
# ══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.markdown(
        '<p class="section-note">'
        "Histogram with optional KDE overlay. Split by source, instrument, or PRE flag."
        "</p>",
        unsafe_allow_html=True,
    )

    hc1, hc2, hc3 = st.columns([2, 2, 2])

    with hc1:
        hist_col = st.selectbox(
            "Column", _numeric_cols,
            index=_numeric_cols.index("bpflux") if "bpflux" in _numeric_cols else 0,
            format_func=lambda c: f"{c}  —  {_COL_HINTS[c]}" if c in _COL_HINTS else c,
            key="hist_col",
        )
        hist_log_x = st.checkbox("Log X", value=False, key="hist_lx")
        hist_log_y = st.checkbox("Log Y", value=False, key="hist_ly")

    with hc2:
        hist_split = st.selectbox(
            "Split by",
            ["none", "instrument", "source", "PRE flag"],
            index=0,
            key="hist_split",
        )
        n_bins = st.slider("Bins", 10, 200, 50, key="hist_bins")

    with hc3:
        show_kde = st.checkbox("Overlay KDE", value=True)
        hist_opacity = st.slider("Opacity", 0.1, 1.0, 0.6, step=0.05, key="hist_op")
        hist_cap_pct = st.slider(
            "Clip upper percentile", 90, 100, 99,
            help="Exclude extreme high-value outliers.",
            key="hist_cap",
        )

    hist_data = df_sel[hist_col].dropna()
    _hcap = float(np.percentile(hist_data, hist_cap_pct))
    hist_data_clipped = hist_data[hist_data <= _hcap]
    bin_size = (_hcap - float(hist_data_clipped.min())) / n_bins if n_bins > 0 else 1.0

    fig_hist = go.Figure()

    def _add_hist_trace(sub_data, name, color):
        sub_clipped = sub_data[sub_data <= _hcap]
        if len(sub_clipped) < 2:
            return
        fig_hist.add_trace(go.Histogram(
            x=sub_clipped,
            xbins=dict(start=float(sub_clipped.min()), end=_hcap, size=bin_size),
            name=name,
            marker=dict(color=color, opacity=hist_opacity,
                        line=dict(color="#111", width=0.4)),
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y}}<extra></extra>",
        ))
        if show_kde and len(sub_clipped) >= 10:
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(sub_clipped, bw_method=0.2)
                x_grid = np.linspace(float(sub_clipped.min()), _hcap, 300)
                kde_vals = kde(x_grid) * len(sub_clipped) * bin_size
                fig_hist.add_trace(go.Scatter(
                    x=x_grid, y=kde_vals,
                    mode="lines",
                    name=f"{name} KDE",
                    line=dict(color=color, width=2),
                    hoverinfo="skip",
                    showlegend=False,
                ))
            except Exception:
                pass

    if hist_split == "none":
        _add_hist_trace(hist_data, "All selected", "#c8a96e")
    elif hist_split == "instrument" and "instr" in df_sel.columns:
        for instr in sorted(df_sel["instr"].dropna().unique()):
            sub = df_sel[df_sel["instr"] == instr][hist_col].dropna()
            _add_hist_trace(sub, instr, _INSTR_COLORS.get(instr, "#aaa"))
    elif hist_split == "source" and "name" in df_sel.columns:
        _srcs = df_sel["name"].value_counts().head(12).index.tolist()
        for i, src in enumerate(_srcs):
            sub = df_sel[df_sel["name"] == src][hist_col].dropna()
            _add_hist_trace(sub, src, _PALETTE[i % len(_PALETTE)])
    elif hist_split == "PRE flag" and "rexp" in df_sel.columns:
        for label, mask_pre, color in [
            ("PRE (rexp≥1.629)", df_sel["rexp"] >= 1.629, "#c8a96e"),
            ("Non-PRE",          df_sel["rexp"] <  1.629, "#3a6186"),
        ]:
            sub = df_sel[mask_pre][hist_col].dropna()
            _add_hist_trace(sub, label, color)

    fig_hist.update_layout(
        **_LAYOUT_COMMON,
        barmode="overlay",
        xaxis=dict(
            title=f"{hist_col}" + (f"  —  {_COL_HINTS[hist_col]}" if hist_col in _COL_HINTS else ""),
            type="log" if hist_log_x else "linear",
        ),
        yaxis=dict(title="Count", type="log" if hist_log_y else "linear"),
        legend=dict(orientation="h", y=-0.2, font_size=11),
        margin=dict(t=20, b=60),
        height=460,
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Time Series
# ══════════════════════════════════════════════════════════════════════════════
with tab_ts:
    st.markdown(
        '<p class="section-note">'
        "Any numeric column as a function of burst time (MJD or calendar date), "
        "split by source or instrument."
        "</p>",
        unsafe_allow_html=True,
    )

    tsc1, tsc2, tsc3 = st.columns([2, 2, 2])

    with tsc1:
        ts_y_col = st.selectbox(
            "Y axis", _numeric_cols,
            index=_numeric_cols.index("bpflux") if "bpflux" in _numeric_cols else 0,
            format_func=lambda c: f"{c}  —  {_COL_HINTS[c]}" if c in _COL_HINTS else c,
            key="ts_y",
        )
        ts_y_err_col = st.selectbox("Error bar", ["none"] + _numeric_cols,
                                    index=0, key="ts_yerr")
        ts_y_log = st.checkbox("Log Y", value=False, key="ts_yl")

    with tsc2:
        ts_time_col = st.selectbox(
            "Time axis",
            (["_date"] if "_date" in df_sel.columns else []) + (["time"] if "time" in df_sel.columns else []),
            index=0,
            key="ts_tcol",
            format_func=lambda c: "Calendar date" if c == "_date" else "MJD",
        )
        ts_color_by = st.selectbox(
            "Colour by", ["instrument", "source", "PRE flag", "none"],
            index=0, key="ts_color",
        )

    with tsc3:
        ts_max_pts = st.number_input("Max points", 0, 200_000, 15_000,
                                     step=1000, key="ts_max")
        ts_opacity = st.slider("Opacity", 0.05, 1.0, 0.5, step=0.05, key="ts_op")
        ts_marker_sz = st.slider("Marker size", 2, 14, 4, key="ts_ms")

    ts_df = df_sel.dropna(subset=[ts_time_col, ts_y_col]).copy()
    ts_df = ts_df.sort_values(ts_time_col)
    if ts_max_pts and len(ts_df) > int(ts_max_pts):
        ts_df = ts_df.sample(int(ts_max_pts), random_state=42).sort_values(ts_time_col)

    if len(ts_df) == 0:
        st.info("No data with valid time and y values.")
    else:
        fig_ts2 = go.Figure()

        def _get_ts_groups(cby):
            if cby == "instrument" and "instr" in ts_df.columns:
                return "instr", ts_df["instr"].dropna().unique()
            elif cby == "source" and "name" in ts_df.columns:
                return "name", ts_df["name"].dropna().unique()
            elif cby == "PRE flag" and "rexp" in ts_df.columns:
                ts_df["_pre_ts"] = ts_df["rexp"].apply(
                    lambda v: "PRE" if v >= 1.629 else "Non-PRE"
                )
                return "_pre_ts", ["PRE", "Non-PRE"]
            else:
                return None, None

        ts_grp_col, ts_groups = _get_ts_groups(ts_color_by)

        def _add_ts(sub, name, color):
            ey = sub[ts_y_err_col].tolist() if ts_y_err_col != "none" and ts_y_err_col in sub else None
            fig_ts2.add_trace(go.Scattergl(
                x=sub[ts_time_col], y=sub[ts_y_col],
                mode="markers",
                name=name,
                marker=dict(size=ts_marker_sz, color=color, opacity=ts_opacity),
                error_y=dict(type="data", array=ey, visible=True,
                             thickness=0.7, color=color) if ey else None,
                hovertemplate=f"<b>{name}</b><br>%{{x}}<br>{ts_y_col}: %{{y:.3g}}<extra></extra>",
            ))

        if ts_grp_col is None:
            _add_ts(ts_df, "All selected", "#c8a96e")
        else:
            for idx, grp_val in enumerate(ts_groups):
                sub = ts_df[ts_df[ts_grp_col] == grp_val]
                if len(sub) == 0:
                    continue
                if ts_color_by == "instrument":
                    color = _INSTR_COLORS.get(str(grp_val), _PALETTE[idx % len(_PALETTE)])
                elif ts_color_by == "PRE flag":
                    color = "#c8a96e" if grp_val == "PRE" else "#3a6186"
                else:
                    color = _PALETTE[idx % len(_PALETTE)]
                _add_ts(sub, str(grp_val), color)

        fig_ts2.update_layout(
            **_LAYOUT_COMMON,
            xaxis_title="Date" if ts_time_col == "_date" else "Time (MJD)",
            yaxis=dict(
                title=f"{ts_y_col}" + (f"  —  {_COL_HINTS[ts_y_col]}" if ts_y_col in _COL_HINTS else ""),
                type="log" if ts_y_log else "linear",
            ),
            legend=dict(orientation="h", y=-0.2, font_size=10),
            margin=dict(t=20, b=70),
            height=500,
        )
        st.plotly_chart(fig_ts2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Correlation Matrix
# ══════════════════════════════════════════════════════════════════════════════
with tab_matrix:
    st.markdown(
        '<p class="section-note">'
        "Pearson / Spearman correlation matrix for a selected subset of numeric columns."
        "</p>",
        unsafe_allow_html=True,
    )

    _default_matrix_cols = [c for c in ["bpflux","kt","tau","edt","dur","rexp","alpha","perflx","hc","sc"]
                             if c in _numeric_cols]

    cm_cols = st.multiselect(
        "Columns",
        _numeric_cols,
        default=_default_matrix_cols[:8],
        help="Select 2–15 columns.",
    )
    cm_method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
    cm_min_n  = st.slider("Min shared observations", 10, 500, 50,
                          help="Column pairs with fewer shared valid rows are masked.")

    if len(cm_cols) < 2:
        st.info("Select at least 2 columns.")
    else:
        cm_df = df_sel[cm_cols].copy()

        n = len(cm_cols)
        corr_vals = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_vals[i, j] = 1.0
                    continue
                pair = cm_df[[cm_cols[i], cm_cols[j]]].dropna()
                if len(pair) < cm_min_n:
                    continue
                if cm_method == "pearson":
                    from scipy.stats import pearsonr
                    r, _ = pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
                else:
                    from scipy.stats import spearmanr
                    r, _ = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                corr_vals[i, j] = r

        _cs = [
            [0.0,  "#3a6186"],
            [0.25, "#5a8ab0"],
            [0.5,  "#1a1a22"],
            [0.75, "#a88050"],
            [1.0,  "#e8d5b0"],
        ]

        fig_corr = go.Figure(go.Heatmap(
            z=corr_vals,
            x=cm_cols, y=cm_cols,
            colorscale=_cs,
            zmid=0, zmin=-1, zmax=1,
            text=np.where(np.isnan(corr_vals), "N/A",
                          np.round(corr_vals, 2).astype(str)),
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorbar=dict(title="r", thickness=14),
            hovertemplate="%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>",
        ))
        fig_corr.update_layout(
            **_LAYOUT_COMMON,
            xaxis=dict(tickangle=-40, tickfont_size=11),
            yaxis=dict(tickfont_size=11),
            margin=dict(t=20, b=100, l=100),
            height=max(380, n * 42 + 120),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown(
            '<p class="section-note">'
            'Masked cells indicate fewer than the minimum shared observations.</p>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Data Table
# ══════════════════════════════════════════════════════════════════════════════
with tab_table:
    st.markdown(
        '<p class="section-note">'
        "Browse and export the filtered catalogue. Select columns and download as CSV."
        "</p>",
        unsafe_allow_html=True,
    )

    tbl_c1, tbl_c2 = st.columns([3, 1])

    _default_tbl_cols = [c for c in
        ["name", "time", "instr", "bpflux", "bpfluxe", "kt", "kte",
         "rexp", "tau", "dur", "perflx", "alpha"]
        if c in df_sel.columns]

    with tbl_c1:
        tbl_cols = st.multiselect(
            "Columns", _all_cols,
            default=_default_tbl_cols,
            key="tbl_cols",
        )
    with tbl_c2:
        tbl_max_rows = st.number_input("Max rows", 50, 50_000, 1_000, step=500)
        round_decimals = st.slider("Round decimals", 0, 6, 3)

    if not tbl_cols:
        st.info("Select at least one column.")
    else:
        display_df = df_sel[tbl_cols].head(int(tbl_max_rows))
        for c in display_df.select_dtypes(include="number").columns:
            display_df[c] = display_df[c].round(round_decimals)

        st.dataframe(display_df, use_container_width=True, height=500)

        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="minbar_playground_export.csv",
            mime="text/csv",
        )

        st.caption(
            f"Showing {len(display_df):,} of {len(df_sel):,} filtered bursts · "
            f"{len(tbl_cols)} columns"
        )
