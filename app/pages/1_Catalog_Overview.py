"""
1_Catalog_Overview.py
MINBAR + LMXB catalog overview, with burst-flux bimodality analysis,
temporal coverage, cross-calibration diagnostics, and GC standard candles.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats as sp_stats
import streamlit as st

from pages.utils.physics_helpers import (
    compute_kde_peaks,
    sources_with_enough_bursts,
    compute_gc_fluxes,
    compute_gc_luminosities,
    eddington_band,
    classify_chi2,
    KUULKERS_TO_MINBAR,
    GC_PAIRS,
    CLUSTER_DISTANCES,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Catalog Overview", layout="wide")

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
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem;
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

.bimodal-badge {
    display: inline-block;
    background: linear-gradient(135deg, #c8a96e22, #e8d5b011);
    border: 1px solid #c8a96e44;
    border-radius: 3px;
    padding: 3px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #c8a96e;
    margin-left: 6px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.82rem;
}
.stat-row .label { color: #888; }
.stat-row .value { color: #e8d5b0; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }

.section-note {
    color: #666;
    font-size: 0.78rem;
    line-height: 1.55;
    border-left: 2px solid #333;
    padding-left: 12px;
    margin: 0.5rem 0 1rem 0;
}

div.stAlert { background: #111116; border: 1px solid #222; }
</style>
""", unsafe_allow_html=True)

st.title("LMXB & MINBAR Catalog Overview")
st.markdown(
    "<p style='color:#777; font-family:Instrument Sans; margin-top:-0.8rem; font-size:0.92rem'>"
    "Burst population statistics · flux bimodality · temporal coverage · "
    "cross-calibration · GC standard candles</p>",
    unsafe_allow_html=True,
)

# ── Data loading ──────────────────────────────────────────────────────────────
BASE_DIR = os.environ.get("LMXB_DATA_DIR", ".")

@st.cache_data(show_spinner="Loading MINBAR & LMXB catalogs…")
def load_data():
    bursts = pd.read_csv(
        os.path.join(BASE_DIR, "data/raw/minbar/minbar_bursts_web.txt"),
        sep="\t", skiprows=33, index_col=False,
    )
    lmxb = pd.read_csv(
        os.path.join(BASE_DIR, "data/raw/fortin_lmxb/LMXBwebcat_latest.csv"),
    )
    return bursts, lmxb

try:
    bursts, lmxb_cat = load_data()
except FileNotFoundError:
    st.error(
        f"Data directory not found at `{BASE_DIR}`. "
        "Set the `LMXB_DATA_DIR` environment variable or update `BASE_DIR` in the script."
    )
    st.stop()

# MJD → Python datetime (MJD epoch = 1858-11-17)
_MJD_EPOCH = pd.Timestamp("1858-11-17")
bursts["date"] = _MJD_EPOCH + pd.to_timedelta(
    pd.to_numeric(bursts["time"], errors="coerce"), unit="D"
)

valid_bursts = bursts[(bursts["bpflux"] > 0) & (bursts["bpfluxe"] > 0)].copy()

# Stable instrument colour map
_all_instrs = sorted(bursts["instr"].dropna().unique().tolist())
_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]
_INSTR_COLORS = {
    instr: _PALETTE[i % len(_PALETTE)]
    for i, instr in enumerate(_all_instrs)
}

# ── Helper: hex → rgba ────────────────────────────────────────────────────────
def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Common Plotly layout ──────────────────────────────────────────────────────
_LAYOUT_COMMON = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Instrument Sans, sans-serif", size=12),
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Controls")

    st.markdown("##### Burst selection")
    min_bursts_bio = st.slider(
        "Min bursts for bimodality", 20, 200, 50, step=10,
        help="Minimum valid flux measurements required for a source to appear in bimodality analysis.",
    )
    rexp_threshold = st.select_slider(
        "PRE threshold (rexp ≥)",
        options=[1.629, 1.8, 2.0],
        value=1.629,
        help="Radius-expansion flag threshold. 1.629 = loose, 2.0 = strict.",
    )

    st.markdown("---")
    st.markdown("##### Eddington band")
    X_hydrogen = st.slider("Hydrogen fraction X", 0.0, 0.7, 0.7, step=0.05)
    M_ns_solar = st.slider("M_ns (M☉)", 1.0, 2.5, 1.4, step=0.05)

    st.markdown("---")
    st.markdown("##### Data quality")
    flux_snr_min = st.slider(
        "Min flux S/N", 1.0, 20.0, 3.0, step=0.5,
        help="Exclude bursts with bpflux/bpfluxe below this ratio.",
    )

    st.markdown("---")
    st.caption(
        f"**Catalog:** {len(bursts):,} bursts · {bursts['name'].nunique()} sources  \n"
        f"**Instruments:** {', '.join(_all_instrs)}"
    )


# ── Derived masks ─────────────────────────────────────────────────────────────
high_snr = valid_bursts[valid_bursts["bpflux"] / valid_bursts["bpfluxe"] >= flux_snr_min].copy()
pre_mask = bursts["rexp"] >= rexp_threshold


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_metrics, tab_temporal, tab_bio, tab_calib, tab_gc = st.tabs([
    "Catalog Metrics",
    "Temporal Coverage",
    "Burst Flux Bimodality",
    "Cross-Calibration",
    "GC Standard Candles",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Catalog Metrics
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Bursts", f"{len(bursts):,}")
    c2.metric("Unique Sources", f"{bursts['name'].nunique()}")
    c3.metric("Valid Flux", f"{len(valid_bursts):,}",
              delta=f"{len(valid_bursts)/len(bursts)*100:.0f}%", delta_color="off")
    c4.metric("High S/N", f"{len(high_snr):,}",
              delta=f"S/N≥{flux_snr_min:.0f}", delta_color="off")
    c5.metric("PRE Bursts", f"{pre_mask.sum():,}",
              delta=f"rexp≥{rexp_threshold}", delta_color="off")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Most Observed Sources")
        n_top = st.slider("Top N sources", 5, 40, 20, key="top_n")
        top_src = (
            valid_bursts["name"].value_counts().head(n_top).reset_index()
        )
        top_src.columns = ["Source", "N_bursts"]

        _pre_ct = bursts[pre_mask]["name"].value_counts()
        top_src["N_PRE"] = top_src["Source"].map(_pre_ct).fillna(0).astype(int)
        top_src["PRE_frac"] = top_src["N_PRE"] / top_src["N_bursts"]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=top_src["Source"],
            y=top_src["N_bursts"] - top_src["N_PRE"],
            name="Non-PRE",
            marker=dict(color="#3a6186", opacity=0.85),
            hovertemplate="<b>%{x}</b><br>Non-PRE: %{y}<extra></extra>",
        ))
        fig_bar.add_trace(go.Bar(
            x=top_src["Source"],
            y=top_src["N_PRE"],
            name="PRE",
            marker=dict(color="#c8a96e", opacity=0.9),
            hovertemplate="<b>%{x}</b><br>PRE: %{y}<extra></extra>",
        ))
        fig_bar.update_layout(
            **_LAYOUT_COMMON,
            barmode="stack",
            xaxis_tickangle=-45,
            margin=dict(t=20, b=10),
            height=380,
            legend=dict(orientation="h", y=1.05, font_size=11),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown("#### Instrument Coverage")
        instr_counts = valid_bursts["instr"].value_counts().reset_index()
        instr_counts.columns = ["Instrument", "Count"]
        fig_pie = go.Figure(go.Pie(
            labels=instr_counts["Instrument"],
            values=instr_counts["Count"],
            hole=0.52,
            textinfo="label+percent",
            textfont_size=11,
            marker=dict(
                colors=[_INSTR_COLORS.get(i, "#777") for i in instr_counts["Instrument"]],
                line=dict(color="#111", width=1),
            ),
        ))
        fig_pie.update_layout(
            **_LAYOUT_COMMON,
            showlegend=False,
            margin=dict(t=20, b=10),
            height=380,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    col_pre, col_flux = st.columns(2)

    with col_pre:
        st.markdown("#### PRE Burst Fraction by Source (top 20)")
        st.markdown(
            '<p class="section-note">'
            "Sources with a high PRE fraction are strong standard-candle candidates."
            "</p>",
            unsafe_allow_html=True,
        )
        pre_counts = bursts[pre_mask]["name"].value_counts().rename("N_PRE")
        all_counts = valid_bursts["name"].value_counts().rename("N_all")
        frac_df = pd.concat([pre_counts, all_counts], axis=1).dropna()
        frac_df["frac"] = frac_df["N_PRE"] / frac_df["N_all"]
        frac_df = frac_df.sort_values("frac", ascending=False).head(20).reset_index()
        frac_df.columns = ["Source", "N_PRE", "N_all", "frac"]

        fig_frac = go.Figure(go.Bar(
            x=frac_df["Source"], y=frac_df["frac"],
            text=(frac_df["frac"] * 100).round(0).astype(int).astype(str) + "%",
            textposition="outside",
            textfont=dict(size=10),
            marker=dict(
                color=frac_df["frac"],
                colorscale=[[0, "#1c1c22"], [0.5, "#7a6040"], [1, "#e8d5b0"]],
                showscale=False,
            ),
            hovertemplate="<b>%{x}</b><br>PRE fraction: %{y:.1%}<br>%{customdata[0]} PRE / %{customdata[1]} total<extra></extra>",
            customdata=frac_df[["N_PRE", "N_all"]].values,
        ))
        fig_frac.update_layout(
            **_LAYOUT_COMMON,
            yaxis=dict(tickformat=".0%", range=[0, 1.12]),
            xaxis_tickangle=-45,
            margin=dict(t=20, b=10),
            height=360,
        )
        st.plotly_chart(fig_frac, use_container_width=True)

    with col_flux:
        st.markdown("#### Peak Flux Distribution (all sources)")
        st.markdown(
            '<p class="section-note">'
            "The high-flux tail is enriched with PRE events."
            "</p>",
            unsafe_allow_html=True,
        )
        _flux_all = valid_bursts["bpflux"].to_numpy()
        _flux_pre = valid_bursts.loc[valid_bursts["rexp"] >= rexp_threshold, "bpflux"].to_numpy()
        _bmax = float(np.percentile(_flux_all, 99))

        fig_flux_dist = go.Figure()
        fig_flux_dist.add_trace(go.Histogram(
            x=_flux_all, xbins=dict(start=0, end=_bmax, size=_bmax / 60),
            name="All bursts", marker=dict(color="#3a6186", opacity=0.6),
        ))
        if len(_flux_pre) > 5:
            fig_flux_dist.add_trace(go.Histogram(
                x=_flux_pre, xbins=dict(start=0, end=_bmax, size=_bmax / 60),
                name="PRE bursts", marker=dict(color="#c8a96e", opacity=0.75),
            ))
        fig_flux_dist.update_layout(
            **_LAYOUT_COMMON,
            barmode="overlay",
            xaxis_title="Peak Flux (10⁻⁹ erg s⁻¹ cm⁻²)",
            yaxis_title="Count",
            legend=dict(orientation="h", y=1.05, font_size=11),
            margin=dict(t=20, b=10),
            height=360,
        )
        st.plotly_chart(fig_flux_dist, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Radius-Expansion Flag (rexp) Distribution")
    st.markdown(
        '<p class="section-note">'
        "Natural clustering near 1 and 2 validates the binary PRE classification. "
        "The intermediate region is where threshold choice matters most."
        "</p>",
        unsafe_allow_html=True,
    )

    rexp_col, rexp_ctrl = st.columns([3, 1])

    with rexp_ctrl:
        rexp_source_opts = ["All sources"] + sorted(bursts["name"].dropna().unique().tolist())
        rexp_src = st.selectbox("Filter by source", rexp_source_opts, key="rexp_src")
        rexp_bin_w = st.slider("Bin width", 0.02, 0.2, 0.05, step=0.01, key="rexp_bw")
        rexp_log_y = st.toggle("Log y-axis", value=False, key="rexp_logy")
        rexp_show_instr = st.toggle("Colour by instrument", value=False, key="rexp_instr")

    rexp_df = bursts.dropna(subset=["rexp"]).copy()
    if rexp_src != "All sources":
        rexp_df = rexp_df[rexp_df["name"] == rexp_src]

    with rexp_col:
        if rexp_show_instr:
            fig_rexp = go.Figure()
            for instr, grp in rexp_df.groupby("instr"):
                fig_rexp.add_trace(go.Histogram(
                    x=grp["rexp"],
                    xbins=dict(start=0, end=rexp_df["rexp"].max() + rexp_bin_w,
                               size=rexp_bin_w),
                    name=instr,
                    marker_color=_INSTR_COLORS.get(instr, "#aaa"),
                    opacity=0.7,
                    hovertemplate=f"<b>{instr}</b><br>rexp: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
                ))
            fig_rexp.update_layout(barmode="stack")
        else:
            fig_rexp = go.Figure(go.Histogram(
                x=rexp_df["rexp"],
                xbins=dict(start=0, end=rexp_df["rexp"].max() + rexp_bin_w,
                           size=rexp_bin_w),
                marker=dict(color="#3a6186", opacity=0.8,
                            line=dict(color="#111", width=0.4)),
                hovertemplate="rexp: %{x:.2f}<br>Count: %{y}<extra></extra>",
                name="All instruments",
            ))

        for thresh, label, color in [
            (1.629, "1.629 (loose)",  "#c8a96e"),
            (1.8,   "1.8",           "#e0a0a0"),
            (2.0,   "2.0 (strict)",  "#e05252"),
        ]:
            fig_rexp.add_vline(
                x=thresh,
                line=dict(color=color, width=1.8, dash="dash"),
                annotation_text=label,
                annotation_position="top right",
                annotation_font_size=10,
                annotation_font_color=color,
            )

        fig_rexp.update_layout(
            **_LAYOUT_COMMON,
            barmode="stack" if rexp_show_instr else "overlay",
            xaxis_title="rexp",
            yaxis_title="Number of bursts",
            yaxis_type="log" if rexp_log_y else "linear",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=20, b=60),
            height=380,
        )
        st.plotly_chart(fig_rexp, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Flux vs. Duration")
    st.markdown(
        '<p class="section-note">'
        "Short, bright bursts tend to be hydrogen-poor (pure He ignition). "
        "PRE bursts cluster at the bright end."
        "</p>",
        unsafe_allow_html=True,
    )

    _dur_col = None
    for c in ["e_b.dur", "dur", "tdel"]:
        if c in valid_bursts.columns:
            _dur_col = c
            break

    if _dur_col is not None:
        _scatter_df = valid_bursts.dropna(subset=[_dur_col]).copy()
        _scatter_df = _scatter_df[(_scatter_df[_dur_col] > 0) & (_scatter_df[_dur_col] < 500)]
        _scatter_df["is_pre"] = _scatter_df["rexp"] >= rexp_threshold

        fig_scatter = go.Figure()
        for is_pre, label, color, sz in [(False, "Non-PRE", "#3a6186", 3), (True, "PRE", "#c8a96e", 5)]:
            sub = _scatter_df[_scatter_df["is_pre"] == is_pre]
            fig_scatter.add_trace(go.Scattergl(
                x=sub[_dur_col], y=sub["bpflux"],
                mode="markers",
                marker=dict(size=sz, color=color, opacity=0.35),
                name=label,
                hovertemplate=f"<b>{label}</b><br>Duration: %{{x:.1f}} s<br>Flux: %{{y:.1f}}<extra></extra>",
            ))

        fig_scatter.update_layout(
            **_LAYOUT_COMMON,
            xaxis=dict(title="Burst Duration (s)", type="log"),
            yaxis=dict(title="Peak Flux (10⁻⁹ erg s⁻¹ cm⁻²)", type="log"),
            legend=dict(orientation="h", y=1.05, font_size=11),
            margin=dict(t=20, b=10),
            height=400,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Duration column not found — scatter plot skipped.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Temporal Coverage
# ══════════════════════════════════════════════════════════════════════════════
with tab_temporal:
    st.markdown(
        "Burst detection cadence, cumulative discovery timeline, "
        "and instrument era coverage."
    )

    dated = bursts.dropna(subset=["date"]).copy()
    dated["year"] = dated["date"].dt.year
    dated["month"] = dated["date"].dt.to_period("M").astype(str)

    st.markdown("#### Cumulative Burst Detections")
    cumul_ctrl_col, cumul_plot_col = st.columns([1, 3])
    with cumul_ctrl_col:
        cumul_by_instr = st.toggle("Break down by instrument", value=True, key="cumul_instr")

    with cumul_plot_col:
        fig_cumul = go.Figure()
        if cumul_by_instr:
            for instr in _all_instrs:
                sub = dated[dated["instr"] == instr].sort_values("date")
                if len(sub) == 0:
                    continue
                sub = sub.reset_index(drop=True)
                sub["cumcount"] = range(1, len(sub) + 1)
                fig_cumul.add_trace(go.Scatter(
                    x=sub["date"], y=sub["cumcount"],
                    mode="lines",
                    name=instr,
                    line=dict(color=_INSTR_COLORS.get(instr, "#aaa"), width=2),
                    hovertemplate=f"<b>{instr}</b><br>%{{x|%Y-%m}}<br>%{{y}} bursts<extra></extra>",
                ))
        else:
            all_sorted = dated.sort_values("date").reset_index(drop=True)
            all_sorted["cumcount"] = range(1, len(all_sorted) + 1)
            fig_cumul.add_trace(go.Scatter(
                x=all_sorted["date"], y=all_sorted["cumcount"],
                mode="lines", name="All",
                line=dict(color="#e8d5b0", width=2.5),
            ))

        fig_cumul.update_layout(
            **_LAYOUT_COMMON,
            xaxis_title="Date",
            yaxis_title="Cumulative Bursts Detected",
            legend=dict(orientation="h", y=-0.15, font_size=11),
            margin=dict(t=20, b=60),
            height=400,
        )
        st.plotly_chart(fig_cumul, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Burst Rate Heatmap (year × source)")
    st.markdown(
        '<p class="section-note">'
        "Rows = sources ranked by total burst count. Columns = years. "
        "Intensity = number of bursts detected."
        "</p>",
        unsafe_allow_html=True,
    )

    n_heatmap = st.slider("Top N sources", 5, 50, 20, key="hm_n")
    _top_names = dated["name"].value_counts().head(n_heatmap).index.tolist()
    _hm_df = dated[dated["name"].isin(_top_names)]
    _pivot = _hm_df.groupby(["name", "year"]).size().unstack(fill_value=0)
    _pivot = _pivot.loc[_pivot.sum(axis=1).sort_values(ascending=True).index]

    fig_hm = go.Figure(go.Heatmap(
        z=_pivot.values,
        x=_pivot.columns.astype(str),
        y=_pivot.index,
        colorscale=[
            [0, "#0d0d10"], [0.01, "#1a1a25"],
            [0.15, "#3a6186"], [0.5, "#c8a96e"],
            [1.0, "#f5e6c8"],
        ],
        hovertemplate="<b>%{y}</b><br>Year: %{x}<br>Bursts: %{z}<extra></extra>",
        colorbar=dict(title="N", thickness=12),
    ))
    fig_hm.update_layout(
        **_LAYOUT_COMMON,
        xaxis=dict(title="Year", tickangle=-45, dtick=2),
        yaxis=dict(title="", tickfont_size=10),
        margin=dict(t=20, b=60, l=160),
        height=max(300, n_heatmap * 22 + 80),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Instrument Observation Eras")

    instr_eras = (
        dated.groupby("instr")["date"]
        .agg(["min", "max", "count"])
        .sort_values("min")
        .reset_index()
    )

    fig_era = go.Figure()
    for i, row in instr_eras.iterrows():
        color = _INSTR_COLORS.get(row["instr"], "#888")
        fig_era.add_trace(go.Scatter(
            x=[row["min"], row["max"]],
            y=[row["instr"], row["instr"]],
            mode="lines+markers",
            line=dict(color=color, width=6),
            marker=dict(color=color, size=8),
            name=row["instr"],
            hovertemplate=(
                f"<b>{row['instr']}</b><br>"
                f"{row['min'].strftime('%Y-%m')} → {row['max'].strftime('%Y-%m')}<br>"
                f"{row['count']:,} bursts"
                "<extra></extra>"
            ),
            showlegend=False,
        ))
        mid = row["min"] + (row["max"] - row["min"]) / 2
        fig_era.add_annotation(
            x=mid, y=row["instr"],
            text=f"{row['count']:,}",
            font=dict(size=10, color=color),
            showarrow=False, yshift=14,
        )

    fig_era.update_layout(
        **_LAYOUT_COMMON,
        xaxis_title="Date",
        yaxis=dict(title=""),
        margin=dict(t=20, b=40, l=120),
        height=max(200, len(instr_eras) * 50 + 60),
    )
    st.plotly_chart(fig_era, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Burst Flux Bimodality
# ══════════════════════════════════════════════════════════════════════════════
with tab_bio:
    st.markdown(
        "Sources with a **bimodal peak-flux distribution** likely alternate between "
        "**hydrogen-rich** (lower flux) and **pure-helium** (higher flux) ignition. "
        "The ratio of the two peaks encodes the fuel composition.",
    )

    bio_sources = sources_with_enough_bursts(bursts, min_bursts=min_bursts_bio)

    col_sel, col_opts = st.columns([2, 3])
    with col_sel:
        selected = st.selectbox(
            "Source", bio_sources,
            index=bio_sources.index("4U 1636-536") if "4U 1636-536" in bio_sources else 0,
        )
    with col_opts:
        bw = st.slider("KDE bandwidth", 0.05, 0.5, 0.15, step=0.01,
                        help="Gaussian KDE bandwidth. Lower values are more sensitive to narrow peaks.")
        flux_cap = st.slider("Flux cap (10⁻⁹ erg/s/cm²)", 10, 300, 150, step=10,
                             help="Upper flux limit to exclude outliers.")
        show_pre_overlay = st.toggle("Overlay PRE bursts", value=True)
        show_compare = st.toggle("Compare a second source", value=False)
        if show_compare:
            compare_src = st.selectbox("Second source", bio_sources,
                                       index=1 if len(bio_sources) > 1 else 0,
                                       key="compare_src")

    def flux_traces(source, color_hist, color_kde, color_pre,
                    name_suffix="", show_pre=True, legendgroup="g1"):
        df = valid_bursts[
            (valid_bursts["name"] == source) &
            (valid_bursts["bpflux"] <= flux_cap)
        ].copy()
        fluxes = df["bpflux"].to_numpy()
        pre_df = df[df["rexp"] >= rexp_threshold]
        pre_fluxes = pre_df["bpflux"].to_numpy()

        traces = []
        bin_w = flux_cap / 40

        traces.append(go.Histogram(
            x=fluxes,
            xbins=dict(start=0, end=flux_cap, size=bin_w),
            name=f"All bursts{name_suffix}",
            marker=dict(color=color_hist, opacity=0.55,
                        line=dict(color="#111", width=0.5)),
            legendgroup=legendgroup,
            hovertemplate="Flux bin: %{x}<br>Count: %{y}<extra></extra>",
        ))

        kde_result = compute_kde_peaks(fluxes, bw=bw, x_max=float(flux_cap))
        if kde_result:
            x_g = kde_result["x_grid"]
            k_v = kde_result["kde_vals"]
            scale = len(fluxes) * bin_w
            traces.append(go.Scatter(
                x=x_g, y=k_v * scale,
                mode="lines",
                name=f"KDE{name_suffix}",
                line=dict(color=color_kde, width=2.5),
                legendgroup=legendgroup,
                hoverinfo="skip",
            ))
            pf = kde_result["peak_fluxes"]
            ph = kde_result["peak_heights"]
            for i, (px_val, py_val) in enumerate(zip(pf, ph)):
                traces.append(go.Scatter(
                    x=[px_val], y=[py_val * scale],
                    mode="markers",
                    marker=dict(symbol="line-ns", size=18,
                                color=color_kde,
                                line=dict(color=color_kde, width=2.5)),
                    name=f"Peak {i+1}{name_suffix} ({px_val:.1f})",
                    legendgroup=legendgroup,
                    hovertemplate=f"Peak {i+1}: %{{x:.1f}}<extra></extra>",
                    showlegend=True,
                ))

        if show_pre and len(pre_fluxes) >= 3:
            traces.append(go.Histogram(
                x=pre_fluxes,
                xbins=dict(start=0, end=flux_cap, size=bin_w),
                name=f"PRE bursts{name_suffix}",
                marker=dict(color=color_pre, opacity=0.7,
                            line=dict(color="#111", width=0.5)),
                legendgroup=legendgroup,
                hovertemplate="Flux bin: %{x}<br>PRE count: %{y}<extra></extra>",
            ))

        return traces, kde_result, len(fluxes), len(pre_fluxes)

    colors_1 = ("#3a6186", "#82b4d9", "#c8a96e")
    colors_2 = ("#5a3560", "#b47ec8", "#80c896")

    if show_compare:
        fig_bio = make_subplots(rows=1, cols=2,
                                subplot_titles=[selected, compare_src],
                                shared_yaxes=False,
                                horizontal_spacing=0.08)
        t1, kr1, n1, npre1 = flux_traces(selected,    *colors_1, " (1)", show_pre_overlay, "g1")
        t2, kr2, n2, npre2 = flux_traces(compare_src, *colors_2, " (2)", show_pre_overlay, "g2")
        for t in t1:
            fig_bio.add_trace(t, row=1, col=1)
        for t in t2:
            fig_bio.add_trace(t, row=1, col=2)
        fig_bio.update_xaxes(title_text="Peak Flux (10⁻⁹ erg s⁻¹ cm⁻²)", row=1, col=1)
        fig_bio.update_xaxes(title_text="Peak Flux (10⁻⁹ erg s⁻¹ cm⁻²)", row=1, col=2)
        fig_bio.update_yaxes(title_text="Number of Bursts", row=1, col=1)
        height_bio = 480
    else:
        fig_bio = go.Figure()
        t1, kr1, n1, npre1 = flux_traces(selected, *colors_1, "", show_pre_overlay, "g1")
        for t in t1:
            fig_bio.add_trace(t)
        kr2, n2, npre2 = None, 0, 0
        fig_bio.update_xaxes(title_text="Peak Flux (10⁻⁹ erg s⁻¹ cm⁻²)")
        fig_bio.update_yaxes(title_text="Number of Bursts")
        height_bio = 440

    fig_bio.update_layout(
        **_LAYOUT_COMMON,
        barmode="overlay",
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=40, b=60),
        height=height_bio,
    )
    st.plotly_chart(fig_bio, use_container_width=True)

    def bio_card(source, kde_result, n_all, n_pre, col):
        with col:
            st.markdown(f"**{source}**")
            st.caption(f"{n_all} total bursts · {n_pre} PRE")

            src_fluxes = valid_bursts.loc[
                (valid_bursts["name"] == source) &
                (valid_bursts["bpflux"] <= flux_cap),
                "bpflux"
            ].to_numpy()
            if len(src_fluxes) >= 15:
                try:
                    from diptest import diptest as dip_test
                    dip_stat, dip_p = dip_test(src_fluxes)
                    st.metric("Dip statistic", f"{dip_stat:.4f}",
                              delta=f"p={dip_p:.3f}", delta_color="off",
                              help="Hartigan's dip test: small p rejects unimodality")
                except ImportError:
                    pass

            if kde_result and kde_result["ratio"] is not None:
                pf = kde_result["peak_fluxes"]
                ratio = kde_result["ratio"]
                st.metric("Peak 1 flux", f"{pf[0]:.1f}",
                          help="Lower flux peak (10⁻⁹ erg/s/cm²)")
                st.metric("Peak 2 flux", f"{pf[1]:.1f}",
                          help="Upper flux peak (10⁻⁹ erg/s/cm²)")
                st.metric("Peak ratio P2/P1", f"{ratio:.3f}",
                          help="Ratio > 1.2 consistent with He vs H ignition")
                if ratio > 1.15:
                    st.markdown(
                        '<span class="bimodal-badge">bimodal candidate</span>',
                        unsafe_allow_html=True,
                    )
            elif kde_result and len(kde_result["peak_fluxes"]) == 1:
                st.metric("Single peak", f"{kde_result['peak_fluxes'][0]:.1f}")
                st.caption("Unimodal")
            else:
                st.caption("No peaks detected — try lowering the bandwidth.")

    st.markdown("---")
    if show_compare:
        mc1, mc2 = st.columns(2)
        bio_card(selected,    kr1, n1, npre1, mc1)
        bio_card(compare_src, kr2, n2, npre2, mc2)
    else:
        _, mc, _ = st.columns([1, 2, 1])
        bio_card(selected, kr1, n1, npre1, mc)

    st.markdown("---")
    st.markdown("#### Burst Flux Time Series")
    st.markdown(
        '<p class="section-note">'
        "Time-ordered peak fluxes coloured by instrument. "
        "Flux jumps coinciding with instrument transitions reveal cross-calibration offsets."
        "</p>",
        unsafe_allow_html=True,
    )

    ts_df = bursts[
        (bursts["name"] == selected) &
        (bursts["bpflux"] > 0) &
        (bursts["bpfluxe"] > 0) &
        bursts["date"].notna()
    ].copy().sort_values("date")

    if len(ts_df) < 2:
        st.info("Not enough dated bursts for this source.")
    else:
        ts_ctrl_l, ts_ctrl_r = st.columns([2, 1])
        with ts_ctrl_r:
            show_mean_band = st.toggle("Show mean lines", value=True, key="ts_mean")
            show_pre_dots = st.toggle("Highlight PRE bursts", value=True, key="ts_pre")
            ts_flux_cap = st.slider(
                "Flux cap", 0.0,
                float(ts_df["bpflux"].quantile(0.995)) * 1.5,
                float(ts_df["bpflux"].quantile(0.995)),
                key="ts_cap",
            )

        ts_df = ts_df[ts_df["bpflux"] <= ts_flux_cap]

        _sm = float(ts_df["bpflux"].mean())
        _se = float(ts_df["bpflux"].std() / np.sqrt(len(ts_df)))
        _w = 1.0 / ts_df["bpfluxe"] ** 2
        _wm = float((_w * ts_df["bpflux"]).sum() / _w.sum())
        _we = float(np.sqrt(1.0 / _w.sum()))

        fig_ts = go.Figure()

        for instr, grp in ts_df.groupby("instr"):
            color = _INSTR_COLORS.get(instr, "#aaaaaa")
            is_pre = grp["rexp"] >= rexp_threshold

            non_pre = grp[~is_pre]
            if len(non_pre):
                fig_ts.add_trace(go.Scatter(
                    x=non_pre["date"], y=non_pre["bpflux"],
                    error_y=dict(type="data", array=non_pre["bpfluxe"].tolist(),
                                 visible=True, thickness=0.8,
                                 color=_hex_to_rgba(color, 0.4)),
                    mode="markers",
                    marker=dict(size=5, color=color, opacity=0.55, line=dict(width=0)),
                    name=instr, legendgroup=instr,
                    hovertemplate=f"<b>{instr}</b><br>%{{x|%Y-%m-%d}}<br>Flux: %{{y:.1f}}<extra></extra>",
                ))

            if show_pre_dots:
                pre = grp[is_pre]
                if len(pre):
                    fig_ts.add_trace(go.Scatter(
                        x=pre["date"], y=pre["bpflux"],
                        error_y=dict(type="data", array=pre["bpfluxe"].tolist(),
                                     visible=True, thickness=0.8,
                                     color=_hex_to_rgba(color, 0.7)),
                        mode="markers",
                        marker=dict(
                            size=9, color=color, opacity=0.9,
                            line=dict(color="#fff", width=1.2), symbol="circle",
                        ),
                        name=f"{instr} (PRE)", legendgroup=instr, showlegend=False,
                        hovertemplate=f"<b>{instr} — PRE</b><br>%{{x|%Y-%m-%d}}<br>Flux: %{{y:.1f}}<extra></extra>",
                    ))

        if show_mean_band:
            fig_ts.add_hrect(y0=_sm - _se, y1=_sm + _se,
                             fillcolor="#e8d5b0", opacity=0.07, line_width=0)
            fig_ts.add_hline(
                y=_sm,
                line=dict(color="#e8d5b0", width=2.0, dash="solid"),
                annotation_text=f"mean = {_sm:.1f} ± {_se:.1f}",
                annotation_font_color="#e8d5b0",
                annotation_font_size=11,
                annotation_position="top left",
            )
            fig_ts.add_hline(
                y=_wm,
                line=dict(color="#a89070", width=1.5, dash="dash"),
                annotation_text=f"wtd = {_wm:.1f} ± {_we:.1f}",
                annotation_font_color="#a89070",
                annotation_font_size=10,
                annotation_position="bottom left",
            )

        fig_ts.update_layout(
            **_LAYOUT_COMMON,
            xaxis_title="Date",
            yaxis_title="Peak Flux (10⁻⁹ erg s⁻¹ cm⁻²)",
            legend=dict(orientation="h", y=-0.2, font_size=11),
            margin=dict(t=20, b=70),
            height=400,
        )

        with ts_ctrl_l:
            st.markdown(
                f"**{selected}** · {len(ts_df)} dated bursts · "
                f"{ts_df['instr'].nunique()} instruments · "
                f"span: {ts_df['date'].min().strftime('%Y')}–{ts_df['date'].max().strftime('%Y')}"
            )

        st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")
    with st.expander("Bimodality scan — all sources", expanded=False):
        st.caption(
            "KDE peak detection across all qualifying sources. "
            "Sources with peak ratio > 1.15 flagged as bimodal candidates."
        )
        if st.button("Run scan"):
            scan_results = []
            prog = st.progress(0)
            for i, src in enumerate(bio_sources):
                df = valid_bursts[
                    (valid_bursts["name"] == src) &
                    (valid_bursts["bpflux"] <= float(flux_cap))
                ]
                kr = compute_kde_peaks(df["bpflux"].to_numpy(), bw=bw,
                                       x_max=float(flux_cap))
                n_pre = len(df[df["rexp"] >= rexp_threshold])
                if kr and kr["ratio"] is not None:
                    scan_results.append({
                        "Source": src,
                        "N_bursts": len(df),
                        "N_PRE": n_pre,
                        "Peak 1": round(kr["peak_fluxes"][0], 1),
                        "Peak 2": round(kr["peak_fluxes"][1], 1),
                        "Ratio P2/P1": round(kr["ratio"], 3),
                        "Bimodal?": "Y" if kr["ratio"] > 1.15 else "—",
                    })
                prog.progress((i + 1) / len(bio_sources))

            scan_df = pd.DataFrame(scan_results).sort_values(
                "Ratio P2/P1", ascending=False
            )
            st.dataframe(
                scan_df.style.apply(
                    lambda col: ["background-color:#1a2a1a" if v == "Y" else "" for v in col]
                    if col.name == "Bimodal?" else [""] * len(col),
                    axis=0,
                ),
                use_container_width=True,
                height=400,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Cross-Calibration
# ══════════════════════════════════════════════════════════════════════════════
with tab_calib:
    st.markdown(
        "Systematic instrument-to-instrument flux offsets can masquerade as astrophysical "
        "variability. This tab quantifies inter-instrument consistency for heavily-observed sources."
    )

    _calib_cts = (
        valid_bursts.groupby(["name", "instr"]).size()
        .unstack(fill_value=0)
    )
    _multi_instr = _calib_cts[(_calib_cts >= 10).sum(axis=1) >= 2].index.tolist()

    if not _multi_instr:
        st.info("No sources with ≥ 10 bursts in at least 2 instruments.")
    else:
        calib_src = st.selectbox("Source", sorted(_multi_instr), key="calib_src")

        src_df = valid_bursts[
            (valid_bursts["name"] == calib_src) &
            (valid_bursts["bpflux"] / valid_bursts["bpfluxe"] >= flux_snr_min)
        ].copy()

        instr_stats = []
        for instr, grp in src_df.groupby("instr"):
            if len(grp) < 5:
                continue
            f = grp["bpflux"]
            w = 1.0 / grp["bpfluxe"] ** 2
            wm = float((w * f).sum() / w.sum())
            we = float(np.sqrt(1.0 / w.sum()))
            instr_stats.append({
                "Instrument": instr,
                "N": len(grp),
                "Mean Flux": float(f.mean()),
                "Std": float(f.std()),
                "Wtd Mean": wm,
                "Wtd Err": we,
                "Median": float(f.median()),
            })
        idf = pd.DataFrame(instr_stats)

        if len(idf) < 2:
            st.info(f"Only one instrument has enough high-S/N bursts for {calib_src}.")
        else:
            st.markdown(f"#### Instrument flux comparison — {calib_src}")

            fig_box = go.Figure()
            for _, row in idf.iterrows():
                instr = row["Instrument"]
                grp = src_df[src_df["instr"] == instr]["bpflux"]
                color = _INSTR_COLORS.get(instr, "#aaa")
                fig_box.add_trace(go.Box(
                    y=grp, name=instr,
                    marker_color=color,
                    boxmean="sd",
                    hoverinfo="y",
                ))

            fig_box.update_layout(
                **_LAYOUT_COMMON,
                yaxis_title="Peak Flux (10⁻⁹ erg s⁻¹ cm⁻²)",
                showlegend=False,
                margin=dict(t=20, b=10),
                height=380,
            )
            st.plotly_chart(fig_box, use_container_width=True)

            st.markdown("##### Kolmogorov–Smirnov pairwise tests")
            st.markdown(
                '<p class="section-note">'
                "p < 0.05 suggests the two instruments sample different flux distributions, "
                "possibly indicating a calibration offset."
                "</p>",
                unsafe_allow_html=True,
            )
            ks_rows = []
            instruments = idf["Instrument"].tolist()
            for i in range(len(instruments)):
                for j in range(i + 1, len(instruments)):
                    a = src_df[src_df["instr"] == instruments[i]]["bpflux"].to_numpy()
                    b = src_df[src_df["instr"] == instruments[j]]["bpflux"].to_numpy()
                    ks_stat, ks_p = sp_stats.ks_2samp(a, b)
                    ks_rows.append({
                        "Pair": f"{instruments[i]}  vs  {instruments[j]}",
                        "KS statistic": round(ks_stat, 4),
                        "p-value": round(ks_p, 4),
                        "Significant?": "Y" if ks_p < 0.05 else "—",
                    })
            st.dataframe(pd.DataFrame(ks_rows), use_container_width=True, hide_index=True)

            st.markdown("##### Per-instrument summary")
            st.dataframe(idf.round(2), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — GC Standard Candles
# ══════════════════════════════════════════════════════════════════════════════
with tab_gc:
    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        st.markdown("#### Controls")
        dominant_only = st.toggle("Dominant instrument only", False,
                                  help="Use only the most-represented instrument per source.")
        exclude_outliers = st.multiselect(
            "Exclude sources",
            options=list(KUULKERS_TO_MINBAR.keys()),
            default=["4U 1746-37", "GRS 1747-312"],
            help="Sources with contaminated distances or unusual physics.",
        )
        st.markdown("---")
        st.markdown("##### Time window")
        _gc_date_min = bursts["date"].dropna().min().date()
        _gc_date_max = bursts["date"].dropna().max().date()
        _kuulkers_cutoff = pd.Timestamp("2002-01-01").date()
        gc_use_window = st.toggle("Filter by date range", value=False)
        if gc_use_window:
            gc_date_range = st.date_input(
                "Date range",
                value=(_gc_date_min, _kuulkers_cutoff),
                min_value=_gc_date_min,
                max_value=_gc_date_max,
                key="gc_dates",
            )
            gc_t0 = pd.Timestamp(gc_date_range[0])
            gc_t1 = pd.Timestamp(gc_date_range[1]) if len(gc_date_range) > 1 else pd.Timestamp(_gc_date_max)
        else:
            gc_t0, gc_t1 = None, None
        st.markdown("---")
        st.markdown("##### χ² classification")
        thresh_consistent = st.slider("Consistent ≤", 0.5, 3.0, 1.5, step=0.1)
        thresh_borderline = st.slider("Borderline ≤", 2.0, 10.0, 5.0, step=0.5)

    if gc_use_window and gc_t0 is not None:
        bursts_gc = bursts[
            bursts["date"].notna() &
            (bursts["date"] >= gc_t0) &
            (bursts["date"] <= gc_t1)
        ].copy()
        _window_label = f"{gc_t0.date()} → {gc_t1.date()}"
    else:
        bursts_gc = bursts
        _window_label = "All epochs"

    df_flux = compute_gc_fluxes(bursts_gc, rexp_threshold=rexp_threshold,
                                dominant_instr_only=dominant_only)
    df_flux["class"] = df_flux["chi2_red"].apply(
        lambda x: classify_chi2(x, thresh_consistent, thresh_borderline)
    )
    df_lum = compute_gc_luminosities(df_flux)

    L_lo_07, L_hi_07 = eddington_band(X_hydrogen, M_ns_solar)
    L_lo_00, L_hi_00 = eddington_band(0.0, M_ns_solar)

    mask = (df_lum["N_PRE"] >= 1) & (~df_lum["system"].isin(exclude_outliers))
    L_crit_mean = df_lum.loc[mask, "L_1e38"].mean()
    L_crit_std = df_lum.loc[mask, "L_1e38"].std()
    coeff_var = L_crit_std / L_crit_mean if L_crit_mean > 0 else np.nan

    with col_ctrl:
        st.markdown("---")
        st.markdown("##### Derived L_crit")
        st.metric("Mean", f"{L_crit_mean:.2f} × 10³⁸", help="erg/s")
        st.metric("σ", f"{L_crit_std:.2f} × 10³⁸")
        st.metric("CV", f"{coeff_var:.1%}")

    with col_main:
        st.markdown(
            f"#### Peak Luminosity vs. GC Burster"
            f"<span style='font-size:0.78rem; color:#666; font-family:JetBrains Mono; "
            f"margin-left:1rem'>{_window_label}</span>",
            unsafe_allow_html=True,
        )

        palette = {"consistent": "#4caf72", "borderline": "#f5a623", "variable": "#e05252"}
        df_sorted = df_lum.sort_values("L_1e38").reset_index(drop=True)

        fig_gc = go.Figure()

        for (lo, hi, label, alpha) in [
            (L_lo_07, L_hi_07, f"L_Edd (X={X_hydrogen:.1f})", 0.18),
            (L_lo_00, L_hi_00, "L_Edd (X=0)", 0.09),
        ]:
            fig_gc.add_hrect(y0=lo, y1=hi,
                             fillcolor="#aaaaaa", opacity=alpha,
                             layer="below", line_width=0,
                             annotation_text=label,
                             annotation_position="right",
                             annotation_font_size=11)

        fig_gc.add_hline(
            y=L_crit_mean, line=dict(color="#e8d5b0", width=1.5, dash="dash"),
            annotation_text=f"L_crit = {L_crit_mean:.2f}",
            annotation_font_color="#e8d5b0",
        )
        fig_gc.add_hrect(
            y0=L_crit_mean - L_crit_std, y1=L_crit_mean + L_crit_std,
            fillcolor="#e8d5b0", opacity=0.06, line_width=0,
        )

        for _, row in df_sorted.iterrows():
            excluded = row["system"] in exclude_outliers
            symbol = "circle" if "PRE" in row["method"] else "triangle-up"
            color = palette[row["class"]]
            fig_gc.add_trace(go.Scatter(
                x=[row["plot_label"]],
                y=[row["L_1e38"]],
                error_y=dict(type="data", array=[row["Lerr_1e38"]], visible=True,
                             thickness=1.5, color=color),
                mode="markers",
                marker=dict(
                    symbol=symbol, size=13,
                    color=color if not excluded else "#444",
                    opacity=0.4 if excluded else 1.0,
                    line=dict(color="#111", width=1),
                ),
                name=row["system"],
                hovertemplate=(
                    f"<b>{row['system']}</b> / {row['cluster']}<br>"
                    f"L = {row['L_1e38']:.2f} ± {row['Lerr_1e38']:.2f} × 10³⁸ erg/s<br>"
                    f"Method: {row['method']}<br>"
                    f"χ²_red = {row['chi2_red']:.2f}<br>"
                    f"N_PRE = {row['N_PRE']} / {row['N_all']}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

        for cls, col in palette.items():
            fig_gc.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=col, symbol="circle"),
                name=cls.capitalize(), showlegend=True,
            ))
        for sym, nm in [("circle", "PRE"), ("triangle-up", "Non-PRE")]:
            fig_gc.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color="#aaa", symbol=sym),
                name=nm, showlegend=True,
            ))

        fig_gc.update_layout(
            **_LAYOUT_COMMON,
            yaxis=dict(title="L_X (10³⁸ erg s⁻¹)", range=[0, 11]),
            xaxis=dict(title="", tickangle=-40),
            legend=dict(orientation="h", y=-0.3),
            margin=dict(t=20, b=120),
            height=520,
        )
        st.plotly_chart(fig_gc, use_container_width=True)

        st.markdown("#### Flux Stability (χ²_red per source)")
        df_chi = df_sorted.sort_values("chi2_red").reset_index(drop=True)

        fig_chi = go.Figure()
        for _, row in df_chi.iterrows():
            color = palette[row["class"]]
            excluded = row["system"] in exclude_outliers
            fig_chi.add_trace(go.Scatter(
                x=[row["chi2_red"]],
                y=[row["plot_label"]],
                mode="markers",
                marker=dict(size=14, color=color,
                            opacity=0.45 if excluded else 1.0,
                            line=dict(color="#111", width=1)),
                hovertemplate=(
                    f"<b>{row['system']}</b><br>"
                    f"χ²_red = {row['chi2_red']:.2f}  (p={row['p']:.3f})<br>"
                    f"DOF = {row['dof']}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))
            fig_chi.add_shape(
                type="line",
                x0=0, x1=row["chi2_red"],
                y0=row["plot_label"], y1=row["plot_label"],
                line=dict(color=color, width=1.5,
                          dash="dot" if excluded else "solid"),
            )

        for thresh, label in [(thresh_consistent, "consistent"), (thresh_borderline, "borderline")]:
            fig_chi.add_vline(
                x=thresh, line=dict(color="#555", width=1, dash="dash"),
                annotation_text=label, annotation_position="top right",
                annotation_font_size=10,
            )

        fig_chi.update_layout(
            **_LAYOUT_COMMON,
            xaxis_title="Reduced χ²",
            yaxis_title="",
            margin=dict(t=20, b=20, l=220),
            height=420,
        )
        st.plotly_chart(fig_chi, use_container_width=True)

        st.markdown("#### Luminosity Residuals vs. L_crit")
        st.markdown(
            '<p class="section-note">'
            "Fractional deviation (L − L_crit) / L_crit per GC source. "
            "Large outliers indicate non-standard physics or observational issues."
            "</p>",
            unsafe_allow_html=True,
        )

        if L_crit_mean > 0:
            df_resid = df_sorted.copy()
            df_resid["resid"] = (df_resid["L_1e38"] - L_crit_mean) / L_crit_mean
            df_resid["resid_err"] = df_resid["Lerr_1e38"] / L_crit_mean

            fig_resid = go.Figure()
            fig_resid.add_hline(y=0, line=dict(color="#555", width=1))
            fig_resid.add_hrect(
                y0=-coeff_var, y1=coeff_var,
                fillcolor="#e8d5b0", opacity=0.08, line_width=0,
            )

            for _, row in df_resid.iterrows():
                excluded = row["system"] in exclude_outliers
                color = palette[row["class"]]
                fig_resid.add_trace(go.Scatter(
                    x=[row["plot_label"]],
                    y=[row["resid"]],
                    error_y=dict(type="data", array=[row["resid_err"]], visible=True,
                                 thickness=1.5, color=color),
                    mode="markers",
                    marker=dict(size=11, color=color,
                                opacity=0.4 if excluded else 1.0,
                                line=dict(color="#111", width=1)),
                    hovertemplate=(
                        f"<b>{row['system']}</b><br>"
                        f"Δ = {row['resid']:+.2f} ({row['resid']*100:+.0f}%)"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ))

            fig_resid.update_layout(
                **_LAYOUT_COMMON,
                yaxis=dict(title="(L − L_crit) / L_crit", tickformat=".0%"),
                xaxis=dict(title="", tickangle=-40),
                margin=dict(t=20, b=100),
                height=380,
            )
            st.plotly_chart(fig_resid, use_container_width=True)

        with st.expander("Raw GC flux / luminosity table"):
            cols_show = [
                "system", "cluster", "mean", "err", "chi2_red", "p", "class",
                "N_PRE", "N_all", "method", "d_kpc", "L_1e38", "Lerr_1e38",
            ]
            _display_cols = [c for c in cols_show if c in df_lum.columns]
            st.dataframe(
                df_lum[_display_cols].round(3).sort_values("L_1e38", ascending=False),
                use_container_width=True,
            )
