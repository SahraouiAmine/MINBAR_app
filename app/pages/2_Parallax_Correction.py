"""
3_Quasars.py
Quasar field selection, parallax zero-point calibration, and distance inference
for LMXB targets via the Bailer-Jones (2015) exponentially decreasing space density prior.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from astropy.coordinates import SkyCoord
import astropy.units as u
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quasar Calibration", layout="wide")

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

.section-note {
    color: #666;
    font-size: 0.78rem;
    line-height: 1.55;
    border-left: 2px solid #333;
    padding-left: 12px;
    margin: 0.5rem 0 1rem 0;
}
.result-box {
    background: #0e0e14;
    border: 1px solid #2a2a38;
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    margin: 0.8rem 0;
}
div.stAlert { background: #111116; border: 1px solid #222; }
</style>
""", unsafe_allow_html=True)

st.title("Quasar Parallax Calibration")
st.markdown(
    "<p style='color:#777; font-family:Instrument Sans; margin-top:-0.8rem; font-size:0.92rem'>"
    "Quasar field selection · parallax zero-point · Bailer-Jones distance posterior</p>",
    unsafe_allow_html=True,
)

# ── Data loading ──────────────────────────────────────────────────────────────
BASE_DIR    = os.environ.get("LMXB_DATA_DIR", ".")
QUASAR_DIR  = Path(BASE_DIR) / "data" / "raw" / "qusars"   # note: matches your folder name

AVAILABLE = {
    p.stem.replace("_", " ").replace("-", " "): p
    for p in sorted(QUASAR_DIR.glob("*.parquet"))
}

# Canonical display names matching your filenames
STEM_TO_NAME = {
    "Cen X 4":   "Cen X-4",
    "Cyg X 2":   "Cyg X-2",
    "4U 0919 54": "4U 0919-54",
    "XB 2129+47": "XB 2129+47",
}

_display_names = []
_path_map      = {}
for stem, path in AVAILABLE.items():
    label = STEM_TO_NAME.get(stem, stem)
    _display_names.append(label)
    _path_map[label] = path

if not _display_names:
    st.error(f"No parquet files found in `{QUASAR_DIR}`. Run the fetch notebook cells first.")
    st.stop()

_LAYOUT_COMMON = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Instrument Sans, sans-serif", size=12),
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Controls")

    st.markdown("##### Target field")
    target_name = st.selectbox("Source", _display_names)

    @st.cache_data(show_spinner="Loading quasar field…")
    def load_field(path_str):
        df = pd.read_parquet(path_str)
        for col in ["parallax", "parallax_error", "phot_g_mean_mag", "bp_rp",
                    "pmra", "pmdec", "pmra_error", "pmdec_error"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Drop rows missing the columns we need
        df = df.dropna(subset=["parallax", "parallax_error",
                                "phot_g_mean_mag", "bp_rp", "ra", "dec"])
        # Compute ecliptic latitude
        coords = SkyCoord(ra=df["ra"].to_numpy() * u.deg,
                          dec=df["dec"].to_numpy() * u.deg, frame="icrs")
        ecl = coords.transform_to("barycentrictrueecliptic")
        df["sin_beta"] = np.sin(ecl.lat.rad)
        return df

    df_raw = load_field(str(_path_map[target_name]))

    # Target sky coordinates (centre of field)
    try:
        _tc = SkyCoord.from_name(target_name)
        target_ra  = _tc.ra.deg
        target_dec = _tc.dec.deg
    except Exception:
        target_ra  = df_raw["ra"].median()
        target_dec = df_raw["dec"].median()

    st.markdown("---")
    st.markdown("##### Ecliptic latitude filter")
    _sb_med = float(df_raw["sin_beta"].median())
    sin_beta_target = st.slider(
        "sin β target", -1.0, 1.0, round(_sb_med, 2), step=0.01,
        help="Centre of the sin(ecliptic latitude) window.",
    )
    delta_sin_beta = st.slider(
        "Δ sin β (half-width)", 0.01, 0.3, 0.02, step=0.01,
    )

    st.markdown("##### Magnitude filter")
    _mg_med = float(df_raw["phot_g_mean_mag"].median())
    mg_target = st.slider(
        "G mag target", 10.0, 22.0, round(_mg_med, 1), step=0.1,
    )
    delta_mg_rel = st.slider(
        "Δ G (relative half-width)", 0.05, 0.6, 0.316, step=0.01,
        help="Filter half-width = target × this value.",
    )

    st.markdown("##### Colour filter")
    _mbr_med = float(df_raw["bp_rp"].median())
    mbr_target = st.slider(
        "BP−RP target", -1.0, 4.0, round(_mbr_med, 2), step=0.01,
    )
    delta_mbr = st.slider(
        "Δ BP−RP (half-width)", 0.01, 1.0, 0.09, step=0.01,
    )

    st.markdown("---")
    st.caption(
        f"**Field:** {len(df_raw):,} quasars loaded  \n"
        f"RA = {target_ra:.4f}°  Dec = {target_dec:.4f}°"
    )


# ── Apply filters ─────────────────────────────────────────────────────────────
mask_beta = np.abs(df_raw["sin_beta"] - sin_beta_target) <= delta_sin_beta
mask_mg   = np.abs(df_raw["phot_g_mean_mag"] - mg_target) <= (mg_target * delta_mg_rel)
mask_mbr  = np.abs(df_raw["bp_rp"] - mbr_target) <= delta_mbr
final_mask = mask_beta & mask_mg & mask_mbr

df_filtered = df_raw[final_mask].copy()
n_all      = len(df_raw)
n_filtered = len(df_filtered)

# ── Zero-point computation ────────────────────────────────────────────────────
if n_filtered >= 2:
    parallaxes = df_filtered["parallax"].to_numpy()
    errors     = df_filtered["parallax_error"].to_numpy()
    weights    = 1.0 / (errors ** 2)
    w0         = float(np.sum(parallaxes * weights) / np.sum(weights))
    sigma_w0   = float(np.sqrt(1.0 / np.sum(weights)))
else:
    w0, sigma_w0 = np.nan, np.nan

# ── Summary bar ───────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("All quasars",     f"{n_all:,}")
m2.metric("After filters",   f"{n_filtered:,}",
          delta=f"{n_filtered/max(n_all,1)*100:.1f}%", delta_color="off")
m3.metric("Zero point w₀",  f"{w0:.4f} mas" if not np.isnan(w0) else "—")
m4.metric("σ(w₀)",           f"{sigma_w0:.4f} mas" if not np.isnan(sigma_w0) else "—")

if n_filtered < 2:
    st.warning("Fewer than 2 quasars pass the current filters — widen the selection windows.")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_sky, tab_diag, tab_dist = st.tabs([
    "Sky Map",
    "Filter Diagnostics",
    "Distance Posterior",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Sky Map
# ══════════════════════════════════════════════════════════════════════════════
with tab_sky:
    st.markdown(
        '<p class="section-note">'
        "All quasars in the field (grey) with the filtered subset overlaid (gold). "
        "The target LMXB is marked in green."
        "</p>",
        unsafe_allow_html=True,
    )

    fig_sky = go.Figure()

    # All quasars — background
    df_bg = df_raw[~final_mask]
    fig_sky.add_trace(go.Scattergl(
        x=df_bg["ra"], y=df_bg["dec"],
        mode="markers",
        marker=dict(size=4, color="#2a2a3a", opacity=0.6,
                    line=dict(width=0)),
        name=f"Excluded ({len(df_bg):,})",
        hovertemplate="RA: %{x:.4f}<br>Dec: %{y:.4f}<extra>excluded</extra>",
    ))

    # Filtered quasars
    if n_filtered > 0:
        fig_sky.add_trace(go.Scattergl(
            x=df_filtered["ra"], y=df_filtered["dec"],
            mode="markers",
            marker=dict(size=7, color="#c8a96e", opacity=0.85,
                        line=dict(color="#111", width=0.5)),
            name=f"Selected ({n_filtered:,})",
            hovertemplate=(
                "RA: %{x:.4f}<br>Dec: %{y:.4f}<br>"
                "G: %{customdata[0]:.2f}<br>BP−RP: %{customdata[1]:.2f}<br>"
                "ϖ: %{customdata[2]:.4f} mas"
                "<extra>selected</extra>"
            ),
            customdata=df_filtered[["phot_g_mean_mag", "bp_rp", "parallax"]].to_numpy(),
        ))

    # Target marker
    fig_sky.add_trace(go.Scatter(
        x=[target_ra], y=[target_dec],
        mode="markers",
        marker=dict(symbol="cross", size=18, color="#00ff88",
                    line=dict(color="#00ff88", width=2.5)),
        name=target_name,
        hovertemplate=f"<b>{target_name}</b><br>RA: {target_ra:.4f}<br>Dec: {target_dec:.4f}<extra></extra>",
    ))

    fig_sky.update_layout(
        **_LAYOUT_COMMON,
        xaxis=dict(title="RA (deg)", autorange="reversed"),
        yaxis=dict(title="Dec (deg)", scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", y=-0.15, font_size=11),
        margin=dict(t=20, b=60),
        height=540,
    )
    st.plotly_chart(fig_sky, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Filter Diagnostics
# ══════════════════════════════════════════════════════════════════════════════
with tab_diag:
    st.markdown(
        '<p class="section-note">'
        "Distributions of the three filter variables. "
        "Gold = selected subset, grey = full field."
        "</p>",
        unsafe_allow_html=True,
    )

    fig_diag = make_subplots(
        rows=1, cols=3,
        subplot_titles=["sin β (ecliptic latitude)", "G magnitude", "BP−RP colour"],
        horizontal_spacing=0.08,
    )

    def _add_hist_pair(col, row, bins, x_sel, x_all):
        fig_diag.add_trace(go.Histogram(
            x=x_all, nbinsx=bins,
            marker=dict(color="#2a3a4a", opacity=0.7),
            name="All", showlegend=(row == 1),
            hovertemplate="%{x:.3f}: %{y}<extra>all</extra>",
        ), row=1, col=col)
        fig_diag.add_trace(go.Histogram(
            x=x_sel, nbinsx=bins,
            marker=dict(color="#c8a96e", opacity=0.85),
            name="Selected", showlegend=(row == 1),
            hovertemplate="%{x:.3f}: %{y}<extra>selected</extra>",
        ), row=1, col=col)

    _add_hist_pair(1, 1, 50, df_filtered["sin_beta"].to_numpy(),   df_raw["sin_beta"].to_numpy())
    _add_hist_pair(2, 1, 50, df_filtered["phot_g_mean_mag"].to_numpy(), df_raw["phot_g_mean_mag"].to_numpy())
    _add_hist_pair(3, 1, 50, df_filtered["bp_rp"].to_numpy(),       df_raw["bp_rp"].to_numpy())

    # Filter window annotations
    for col, lo, hi in [
        (1, sin_beta_target - delta_sin_beta,    sin_beta_target + delta_sin_beta),
        (2, mg_target - mg_target * delta_mg_rel, mg_target + mg_target * delta_mg_rel),
        (3, mbr_target - delta_mbr,              mbr_target + delta_mbr),
    ]:
        fig_diag.add_vrect(
            x0=lo, x1=hi,
            fillcolor="#c8a96e", opacity=0.08,
            line=dict(color="#c8a96e", width=1, dash="dash"),
            row=1, col=col,
        )

    fig_diag.update_layout(
        **_LAYOUT_COMMON,
        barmode="overlay",
        legend=dict(orientation="h", y=-0.2, font_size=11),
        margin=dict(t=40, b=60),
        height=380,
    )
    st.plotly_chart(fig_diag, use_container_width=True)

    # Parallax distribution of selected quasars
    if n_filtered >= 2:
        st.markdown("#### Parallax distribution — selected quasars")
        st.markdown(
            '<p class="section-note">'
            "Ideally centred on zero with a tight spread. "
            "The weighted mean (dashed line) is the zero-point correction applied to the target."
            "</p>",
            unsafe_allow_html=True,
        )

        _plx = df_filtered["parallax"].to_numpy()
        _plx_clip = np.clip(_plx, np.percentile(_plx, 1), np.percentile(_plx, 99))

        fig_plx = go.Figure()
        fig_plx.add_trace(go.Histogram(
            x=_plx_clip, nbinsx=40,
            marker=dict(color="#3a6186", opacity=0.75,
                        line=dict(color="#111", width=0.4)),
            name="Parallax",
            hovertemplate="ϖ: %{x:.4f} mas<br>Count: %{y}<extra></extra>",
        ))
        fig_plx.add_vline(
            x=0, line=dict(color="#555", width=1, dash="dot"),
            annotation_text="ϖ = 0", annotation_font_size=10,
        )
        if not np.isnan(w0):
            fig_plx.add_vline(
                x=w0,
                line=dict(color="#c8a96e", width=2, dash="dash"),
                annotation_text=f"w₀ = {w0:.4f}",
                annotation_font_color="#c8a96e",
                annotation_font_size=11,
                annotation_position="top right",
            )
        fig_plx.update_layout(
            **_LAYOUT_COMMON,
            xaxis_title="Parallax (mas)",
            yaxis_title="Count",
            margin=dict(t=20, b=20),
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig_plx, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Distance Posterior
# ══════════════════════════════════════════════════════════════════════════════
with tab_dist:
    st.markdown(
        '<p class="section-note">'
        "Enter the target's raw Gaia DR3 parallax and uncertainty. "
        "The zero-point from the selected quasar subset is subtracted automatically. "
        "Distance is inferred using the Bailer-Jones (2015) exponentially decreasing "
        "space density prior."
        "</p>",
        unsafe_allow_html=True,
    )

    inp1, inp2, inp3 = st.columns(3)

    with inp1:
        st.markdown("##### Target parallax")
        raw_parallax = st.number_input(
            "Raw parallax (mas)", value=0.48, step=0.01, format="%.4f",
            help="From gaiadr3.gaia_source for the target source_id.",
        )
        parallax_error = st.number_input(
            "Parallax error (mas)", value=0.10, min_value=0.001, step=0.01, format="%.4f",
        )
        source_id_label = st.text_input(
            "Source ID (reference only)", value="",
            help="Gaia DR3 source_id — not used in computation, kept for record.",
        )

    with inp2:
        st.markdown("##### Zero-point")
        zp_auto = round(w0, 4) if not np.isnan(w0) else 0.0
        use_auto_zp = st.toggle("Use computed zero-point", value=True)
        if use_auto_zp and not np.isnan(w0):
            zero_point = zp_auto
            st.markdown(
                f'<div class="result-box">'
                f'w₀ = <span style="color:#c8a96e; font-family:JetBrains Mono">'
                f'{zero_point:.4f} ± {sigma_w0:.4f}</span> mas'
                f'<br><span style="color:#555; font-size:0.75rem">'
                f'from {n_filtered} quasars</span></div>',
                unsafe_allow_html=True,
            )
        else:
            zero_point = st.number_input(
                "Manual zero-point (mas)", value=zp_auto, step=0.001, format="%.4f",
            )

    with inp3:
        st.markdown("##### Prior")
        L_pc = st.number_input(
            "Scale length L (pc)", value=1350, min_value=100, max_value=10000, step=50,
            help="Exponential prior length scale. Bailer-Jones (2015) use ~1350 pc for the Galaxy.",
        )
        r_min = st.number_input("r min (pc)", value=100,  min_value=1,     step=100)
        r_max = st.number_input("r max (pc)", value=25000, min_value=1000, step=500)
        n_grid = 3000

    # ── Computation ───────────────────────────────────────────────────────────
    calibrated_parallax = raw_parallax - zero_point
    omega = calibrated_parallax / 1000.0   # mas → arcsec
    sigma = parallax_error    / 1000.0

    # Posterior mode via cubic roots (Bailer-Jones 2015, eq. 18)
    # r^3/L - 2r^2 + (omega/sigma^2)*r - 1/sigma^2 = 0
    coeff = np.array([1.0 / L_pc, -2.0, omega / sigma**2, -1.0 / sigma**2])
    roots = np.roots(coeff)
    real_roots = roots[np.isreal(roots)].real
    positive_roots = real_roots[real_roots > 0]

    if len(positive_roots) == 1:
        mode_distance = float(positive_roots[0])
    elif len(positive_roots) > 1:
        mode_distance = float(np.min(positive_roots)) if omega >= 0 else float(positive_roots[0])
    else:
        mode_distance = np.nan

    # Posterior grid
    r = np.linspace(max(1.0, r_min), r_max, n_grid)
    log_posterior = (
        2 * np.log(r)
        - r / L_pc
        - (1.0 / (2 * sigma**2)) * (omega - 1.0 / r) ** 2
    )
    log_posterior -= log_posterior.max()   # numerical stability
    posterior = np.exp(log_posterior)
    posterior /= np.trapezoid(posterior, r)    # normalise

    # Credible interval
    cdf = np.cumsum(posterior) * (r[1] - r[0])
    cdf /= cdf[-1]
    r_lo = float(r[np.searchsorted(cdf, 0.159)])
    r_hi = float(r[np.searchsorted(cdf, 0.841)])
    r_med = float(r[np.searchsorted(cdf, 0.500)])

    # ── Result cards ─────────────────────────────────────────────────────────
    st.markdown("---")
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Calibrated parallax",
               f"{calibrated_parallax:.4f} mas",
               delta=f"raw {raw_parallax:.4f}", delta_color="off")
    rc2.metric("Posterior mode",
               f"{mode_distance:.0f} pc" if not np.isnan(mode_distance) else "—")
    rc3.metric("Posterior median",   f"{r_med:.0f} pc")
    rc4.metric("68% interval",
               f"{r_lo:.0f} – {r_hi:.0f} pc",
               delta=f"±{(r_hi-r_lo)/2:.0f}", delta_color="off")

    # ── Posterior plot ────────────────────────────────────────────────────────
    fig_post = go.Figure()

    # 68% band
    mask_ci = (r >= r_lo) & (r <= r_hi)
    fig_post.add_trace(go.Scatter(
        x=np.concatenate([r[mask_ci], r[mask_ci][::-1]]),
        y=np.concatenate([posterior[mask_ci], np.zeros(mask_ci.sum())]),
        fill="toself",
        fillcolor="rgba(200,169,110,0.12)",
        line=dict(width=0),
        name="68% interval",
        hoverinfo="skip",
    ))

    # Posterior curve
    fig_post.add_trace(go.Scatter(
        x=r, y=posterior,
        mode="lines",
        line=dict(color="#9b7fc8", width=2.5),
        name="Posterior P(r | ω, σ)",
        hovertemplate="r = %{x:.0f} pc<br>P = %{y:.2e}<extra></extra>",
    ))

    # Mode
    if not np.isnan(mode_distance):
        fig_post.add_vline(
            x=mode_distance,
            line=dict(color="#e05252", width=2, dash="dash"),
            annotation_text=f"mode = {mode_distance:.0f} pc",
            annotation_font_color="#e05252",
            annotation_font_size=11,
            annotation_position="top right",
        )

    # Median
    fig_post.add_vline(
        x=r_med,
        line=dict(color="#c8a96e", width=1.5, dash="dot"),
        annotation_text=f"median = {r_med:.0f} pc",
        annotation_font_color="#c8a96e",
        annotation_font_size=10,
        annotation_position="top left",
    )

    fig_post.update_layout(
        **_LAYOUT_COMMON,
        xaxis_title="Distance r (pc)",
        yaxis_title="P(r | ω, σ_ω)  [normalised]",
        legend=dict(orientation="h", y=-0.18, font_size=11),
        margin=dict(t=20, b=60),
        height=440,
        annotations=[dict(
            x=0.01, y=0.97, xref="paper", yref="paper",
            text=(
                f"ω_cal = {calibrated_parallax:.4f} mas  |  "
                f"σ = {parallax_error:.4f} mas  |  "
                f"w₀ = {zero_point:.4f} mas  |  "
                f"L = {L_pc} pc"
            ),
            showarrow=False,
            font=dict(size=10, color="#555", family="JetBrains Mono"),
            align="left",
        )],
    )
    st.plotly_chart(fig_post, use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────────────
    with st.expander("Export calibration result"):
        result_dict = {
            "target":                target_name,
            "source_id":             source_id_label or None,
            "raw_parallax_mas":      raw_parallax,
            "parallax_error_mas":    parallax_error,
            "zero_point_mas":        zero_point,
            "sigma_zero_point_mas":  sigma_w0 if not np.isnan(sigma_w0) else None,
            "n_quasars_used":        n_filtered,
            "calibrated_parallax_mas": calibrated_parallax,
            "L_pc":                  L_pc,
            "mode_distance_pc":      round(mode_distance, 1) if not np.isnan(mode_distance) else None,
            "median_distance_pc":    round(r_med, 1),
            "ci68_lo_pc":            round(r_lo, 1),
            "ci68_hi_pc":            round(r_hi, 1),
        }
        import json
        st.code(json.dumps(result_dict, indent=2), language="json")

        out_path = (
            Path(BASE_DIR) / "data" / "results"
            / f"{target_name.replace(' ', '_')}_parallax_calibration.json"
        )
        if st.button("Save to disk"):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result_dict, f, indent=2)
            st.success(f"Saved → {out_path}")
