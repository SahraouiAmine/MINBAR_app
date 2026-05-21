"""
physics_helpers.py
Shared physics computations for the LMXB / PRE burst dashboard.
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, chi2
from scipy.signal import find_peaks
from pathlib import Path

# ── Physical constants ────────────────────────────────────────────────────────
G    = 6.674e-8          # cm³ g⁻¹ s⁻²
c    = 2.998e10          # cm s⁻¹
Msun = 1.989e33          # g
kpc_to_cm = 3.08567758128e21
F_unit_to_cgs = 1e-9    # minbar flux unit → erg s⁻¹ cm⁻²

# ── Name cross-match tables ───────────────────────────────────────────────────

# MINBAR name  →  Kuulkers (2003) name
MINBAR_TO_KUULKERS = {
    "4U 0513-40":      "MX 0513-40",
    "4U 1722-30":      "4U 1722-30",
    "MXB 1730-335":    "MXB 1730-335",
    "SLX 1732-304":    "XB 1733-30",
    "EXO 1745-248":    "XB 1745-25",
    "SAX J1748.9-2021":"MX 1746-20",
    "4U 1746-37":      "4U 1746-37",
    "GRS 1747-312":    "GRS 1747-312",
    "4U 1820-303":     "4U 1820-30",
    "XB 1832-330":     "H 1825-331",
    "4U 1850-086":     "A 1850-08",
    "M15 X-2":         "4U 2129+12",
}
KUULKERS_TO_MINBAR = {v: k for k, v in MINBAR_TO_KUULKERS.items()}

# MINBAR name  →  Fortin LMXB-cat name  (for the full PRE distance sample)
MINBAR_TO_FORTIN = {
    "4U 1246-588": "4U 1246-58",
    "4U 1702-429": "X Ara X-1",
    "4U 1608-522": "V* QX Nor",
    "4U 1820-303": "X Sgr X-4",
    "4U 1636-536": "V* V801 Ara",
    "4U 1735-444": "V* V926 Sco",
    "4U 0614+09":  "V* V1055 Ori",
    "4U 1254-69":  "V* GR Mus",
}

# Globular cluster hosting each Kuulkers source
GC_PAIRS = {
    "MX 0513-40":   "NGC 1851",
    "4U 1722-30":   "Terzan 2",
    "MXB 1730-335": "Liller 1",
    "XB 1733-30":   "Terzan 1",
    "XB 1745-25":   "Terzan 5",
    "MX 1746-20":   "NGC 6440",
    "4U 1746-37":   "NGC 6441",
    "GRS 1747-312": "Terzan 6",
    "4U 1820-30":   "NGC 6624",
    "H 1825-331":   "NGC 6652",
    "A 1850-08":    "NGC 6712",
    "4U 2129+12":   "NGC 7078",
}

# Globular cluster distances (kpc, err_kpc)
CLUSTER_DISTANCES = {
    "NGC 1851": (11.951, 0.134),
    "Terzan 2": (7.753,  0.332),
    "Liller 1": (8.061,  0.353),
    "Terzan 1": (5.673,  0.175),
    "Terzan 5": (6.617,  0.150),
    "NGC 6440": (8.248,  0.248),
    "NGC 6441": (12.728, 0.163),
    "Terzan 6": (7.271,  0.360),
    "NGC 6624": (8.019,  0.108),
    "NGC 6652": (9.464,  0.139),
    "NGC 6712": (7.382,  0.240),
    "NGC 7078": (10.709, 0.096),
}

# Manual Kuulkers flux rows for sources missing from MINBAR
KUULKERS_MANUAL_ROWS = pd.DataFrame([
    {"name": "SLX 1732-304", "instr": "Kuulkers", "time": np.nan,
     "bpflux": 74.0, "bpfluxe": 10.0, "rexp": 1.0},
    {"name": "SLX 1732-304", "instr": "Kuulkers", "time": np.nan,
     "bpflux": 63.0, "bpfluxe":  8.0, "rexp": 1.0},
    {"name": "4U 1850-086",  "instr": "Kuulkers", "time": np.nan,
     "bpflux": 52.0, "bpfluxe":  5.0, "rexp": 2.0},
    {"name": "4U 1850-086",  "instr": "Kuulkers", "time": np.nan,
     "bpflux": 60.0, "bpfluxe": 20.0, "rexp": 2.0},
    {"name": "4U 1850-086",  "instr": "Kuulkers", "time": np.nan,
     "bpflux": 60.0, "bpfluxe": 10.0, "rexp": 1.0},
])


# ── Eddington luminosity ──────────────────────────────────────────────────────

def L_edd(X: float, R_ns: float = 1e6, M_ns: float = 1.4 * Msun) -> float:
    """
    Eddington luminosity (erg/s) with gravitational redshift correction.

    Parameters
    ----------
    X     : hydrogen mass fraction (0 = pure He, 0.7 = solar)
    R_ns  : neutron star radius in cm  (default 10 km)
    M_ns  : neutron star mass in g     (default 1.4 Msun)
    """
    redshift = np.sqrt(1.0 - (2.0 * G * M_ns) / (c**2 * R_ns))
    kappa    = 0.2 * (1.0 + X)
    return (4.0 * np.pi * c * G * M_ns / kappa) * redshift


def eddington_band(X: float, M_ns_solar: float = 1.4) -> tuple[float, float]:
    """
    Returns (L_low, L_high) in units of 10^38 erg/s for the Eddington band,
    spanning R_ns = 10 km (low) to 100 km (high).
    """
    M_ns = M_ns_solar * Msun
    lo = L_edd(X, R_ns=1e6, M_ns=M_ns) / 1e38
    hi = L_edd(X, R_ns=1e7, M_ns=M_ns) / 1e38
    return lo, hi


# ── χ² flux-stability helpers ─────────────────────────────────────────────────

def classify_chi2(chi2_red: float,
                  threshold_consistent: float = 1.5,
                  threshold_borderline: float  = 5.0) -> str:
    if chi2_red <= threshold_consistent:
        return "consistent"
    elif chi2_red <= threshold_borderline:
        return "borderline"
    return "variable"


def weighted_mean_flux(fluxes: np.ndarray,
                       errors: np.ndarray) -> dict:
    """
    Weighted mean, reduced χ², and p-value for a set of flux measurements.
    Returns a dict with keys: mean, err, chi2_red, p, dof.
    """
    w       = 1.0 / errors**2
    mean    = np.sum(w * fluxes) / np.sum(w)
    err     = np.sqrt(1.0 / np.sum(w))
    chi2_v  = np.sum((fluxes - mean)**2 / errors**2)
    dof     = len(fluxes) - 1
    chi2_r  = chi2_v / dof if dof > 0 else np.nan
    p       = 1.0 - chi2.cdf(chi2_v, dof) if dof > 0 else np.nan
    return dict(mean=mean, err=err, chi2_red=chi2_r, p=p, dof=dof)


def compute_gc_fluxes(bursts: pd.DataFrame,
                      rexp_threshold: float = 1.629,
                      dominant_instr_only: bool = False) -> pd.DataFrame:
    """
    For each Kuulkers GC source compute weighted-mean PRE peak flux,
    χ², and classification.

    Returns a DataFrame with columns:
      system, cluster, mean, err, chi2_red, p, dof, class,
      N_PRE, N_all, method
    """
    clean = bursts[(bursts["bpflux"] > 0) & (bursts["bpfluxe"] > 0)].copy()
    clean = pd.concat([clean, KUULKERS_MANUAL_ROWS], ignore_index=True)

    rows = []
    for kuu, minb in KUULKERS_TO_MINBAR.items():
        df_sys = clean[clean["name"] == minb].copy()

        if dominant_instr_only and len(df_sys) > 0:
            best = df_sys["instr"].value_counts().idxmax()
            df_sys = df_sys[df_sys["instr"] == best]

        all_f  = df_sys["bpflux"].to_numpy()
        all_e  = df_sys["bpfluxe"].to_numpy()
        df_pre = df_sys[df_sys["rexp"] >= rexp_threshold]
        pre_f  = df_pre["bpflux"].to_numpy()
        pre_e  = df_pre["bpfluxe"].to_numpy()

        # pick fluxes to use
        if len(pre_f) >= 2:
            fluxes, errors, method = pre_f, pre_e, "PRE weighted mean"
        elif len(pre_f) == 1:
            fluxes, errors, method = pre_f, pre_e, "Single PRE"
        elif len(all_f) >= 2:
            fluxes, errors, method = all_f, all_e, "max flux"
        else:
            continue

        if len(fluxes) < 2:
            # single measurement — can't do χ²
            rows.append(dict(
                system=kuu, cluster=GC_PAIRS[kuu],
                mean=fluxes[0], err=errors[0],
                chi2_red=np.nan, p=np.nan, dof=0,
                **{"class": "consistent"},
                N_PRE=len(pre_f), N_all=len(all_f), method=method,
            ))
            continue

        stats = weighted_mean_flux(fluxes, errors)
        rows.append(dict(
            system=kuu, cluster=GC_PAIRS[kuu],
            mean=stats["mean"], err=stats["err"],
            chi2_red=stats["chi2_red"], p=stats["p"], dof=stats["dof"],
            **{"class": classify_chi2(stats["chi2_red"])},
            N_PRE=len(pre_f), N_all=len(all_f), method=method,
        ))

    return pd.DataFrame(rows)


def compute_gc_luminosities(df_fluxes: pd.DataFrame) -> pd.DataFrame:
    """Add luminosity columns to the GC flux DataFrame."""
    df = df_fluxes.copy()
    df["d_kpc"]     = df["cluster"].map(lambda c: CLUSTER_DISTANCES[c][0])
    df["d_kpc_err"] = df["cluster"].map(lambda c: CLUSTER_DISTANCES[c][1])
    d_cm     = df["d_kpc"]     * kpc_to_cm
    d_cm_err = df["d_kpc_err"] * kpc_to_cm
    F        = df["mean"] * F_unit_to_cgs
    F_err    = df["err"]  * F_unit_to_cgs
    L        = 4.0 * np.pi * d_cm**2 * F
    dL_dF    = 4.0 * np.pi * d_cm**2
    dL_dd    = 8.0 * np.pi * d_cm * F
    L_err    = np.sqrt((dL_dF * F_err)**2 + (dL_dd * d_cm_err)**2)
    df["L_1e38"]    = L    / 1e38
    df["Lerr_1e38"] = L_err/ 1e38
    df["L_frac_err"]= df["Lerr_1e38"] / df["L_1e38"]
    df["plot_label"]= df["system"] + " / " + df["cluster"]
    return df


# ── Distance inference ────────────────────────────────────────────────────────

def dist_from_flux(F_cgs: float, L_crit: float) -> float:
    """Return distance in kpc given flux (erg/s/cm²) and luminosity (erg/s)."""
    d_cm = np.sqrt(L_crit / (4.0 * np.pi * F_cgs))
    return d_cm / kpc_to_cm


def compute_pre_distances(bursts: pd.DataFrame,
                          L_crit: float = 3.86e38,
                          L_err:  float = 0.98e38,
                          rexp_val: float = 2.0) -> dict:
    """
    For every MINBAR source with rexp == rexp_val PRE bursts,
    compute inferred distances (mean / max / min flux).

    Returns dist_dict[system] = ([d_mean, d_max, d_min], [err_mean, err_max, err_min])
    """
    pre = bursts[bursts["rexp"] == rexp_val].copy()
    frac_err = 0.5 * (L_err / L_crit)
    dist_dict = {}
    for system in pre["name"].unique():
        df = pre[pre["name"] == system]
        f  = df["bpflux"].to_numpy()
        e  = df["bpfluxe"].to_numpy()
        valid = (e > 0) & (f > 0) & ~np.isnan(e) & ~np.isnan(f)
        if not valid.any():
            continue
        vf, ve = f[valid], e[valid]
        w      = 1.0 / ve**2
        mean_f = np.sum(w * vf) / np.sum(w)
        max_f  = np.max(vf)
        min_f  = np.min(vf)
        d_vals = [dist_from_flux(x * F_unit_to_cgs, L_crit)
                  for x in [mean_f, max_f, min_f]]
        d_errs = [d * frac_err for d in d_vals]
        dist_dict[system] = (d_vals, d_errs)
    return dist_dict


# ── Bimodality helpers ────────────────────────────────────────────────────────

def compute_kde_peaks(fluxes: np.ndarray,
                      bw: float = 0.15,
                      x_min: float = 0.0,
                      x_max: float = 150.0,
                      n_grid: int = 2000) -> dict:
    """
    Fit a KDE to flux data and detect peaks.

    Returns
    -------
    dict with keys:
        x_grid, kde_vals,
        peak_fluxes  (sorted ascending),
        peak_heights,
        ratio        (peak2 / peak1, or None if < 2 peaks)
    """
    if len(fluxes) < 5:
        return None # type: ignore
    kde      = gaussian_kde(fluxes, bw_method=bw)
    x_grid   = np.linspace(x_min, x_max, n_grid)
    kde_vals = kde(x_grid)
    peak_idx, _ = find_peaks(kde_vals)
    if len(peak_idx) == 0:
        return dict(x_grid=x_grid, kde_vals=kde_vals,
                    peak_fluxes=np.array([]), peak_heights=np.array([]),
                    ratio=None)
    # sort by height, keep top-2
    order     = np.argsort(kde_vals[peak_idx])[::-1]
    top_idx   = peak_idx[order[:2]]
    pf        = np.sort(x_grid[top_idx])
    ph        = kde_vals[top_idx[np.argsort(x_grid[top_idx])]]
    ratio     = pf[1] / pf[0] if len(pf) == 2 else None
    return dict(x_grid=x_grid, kde_vals=kde_vals,
                peak_fluxes=pf, peak_heights=ph, ratio=ratio)


def sources_with_enough_bursts(bursts: pd.DataFrame,
                               min_bursts: int = 30) -> list[str]:
    """Return source names (sorted by burst count desc) with >= min_bursts valid flux rows."""
    valid = bursts[(bursts["bpflux"] > 0) & (bursts["bpfluxe"] > 0)]
    counts = valid["name"].value_counts()
    return counts[counts >= min_bursts].index.tolist()


# ── Inclination bias simulation ───────────────────────────────────────────────

def xi(cosi: np.ndarray) -> np.ndarray:
    """Fujimoto (1988) geometric correction factor."""
    return 1.0 / (0.5 + np.abs(cosi))


def simulate_inclination_scatter(n_samples: int = 500_000,
                                 d_max_kpc: float = 30.0,
                                 frac_err_L: float = 0.125,
                                 seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the d_true vs d_inferred scatter arising from random inclinations
    and L_crit uncertainty.

    Returns (d_true, d_inferred) both in kpc.
    """
    rng     = np.random.default_rng(seed)
    d_true  = rng.uniform(0, d_max_kpc, n_samples)
    cosi    = rng.uniform(0, 1, n_samples)
    d_inf   = d_true * np.sqrt(xi(cosi))
    noise   = rng.normal(0.0, frac_err_L * d_inf, n_samples)
    return d_true, d_inf + noise


# ── Parallax zero-point helpers (Ding 2021) ───────────────────────────────────

# Ding (2021) Table 2/3: the 4 confirmed PRE bursters with Gaia EDR3 counterparts.
# Used as priors for filter initialisation when these sources are selected.
DING_TARGETS = {
    "Cyg X-2": {
        "source_id":        1952859683185470208,
        "ra":               326.17146667,
        "dec":               38.32140556,
        "parallax_raw":      0.068,   # mas  (π₁, uncalibrated)
        "parallax_error":    0.019,
        # Ding Table 3 optimal filter values
        "r_opt":             5.5,     # deg
        "sin_beta_target":   0.74,
        "delta_sin_beta":    0.02,
        "mg_target":        14.70,
        "delta_mg_rel":      0.316,
        "mbr_target":        0.71,
        "delta_mbr":         0.09,
        # Published zero-point and calibrated parallax
        "pi0_ding":          0.019,   # mas
        "pi0_ding_err":      0.032,
        # Nominal PRE distance at X=0 (Galloway 2020)
        "pre_dist_x0_kpc":  11.6,
        "pre_dist_x0_err":   0.9,
    },
    "Cen X-4": {
        "source_id":        6205715168442046592,
        "ra":               224.59139167,
        "dec":              -31.66900000,
        "parallax_raw":      0.53,
        "parallax_error":    0.13,
        "r_opt":            10.0,
        "sin_beta_target":  -0.24,
        "delta_sin_beta":    0.14,
        "mg_target":        17.85,
        "delta_mg_rel":      0.050,
        "mbr_target":        1.59,
        "delta_mbr":         1.20,
        "pi0_ding":         -0.022,
        "pi0_ding_err":      0.006,
        "pre_dist_x0_kpc":   1.2,
        "pre_dist_x0_err":   0.3,
    },
    "4U 0919-54": {
        "source_id":        5310395631798303104,
        "ra":               140.11029583,
        "dec":              -55.20679722,
        "parallax_raw":      0.24,
        "parallax_error":    0.06,
        "r_opt":             7.2,
        "sin_beta_target":  -0.90,
        "delta_sin_beta":    0.05,
        "mg_target":        17.15,
        "delta_mg_rel":      0.120,
        "mbr_target":        1.19,
        "delta_mbr":         0.58,
        "pi0_ding":         -0.009,
        "pi0_ding_err":      0.013,
        "pre_dist_x0_kpc":   3.9,
        "pre_dist_x0_err":   0.2,
    },
    "XB 2129+47": {
        "source_id":        1978241050130301312,
        "ra":               322.85920667,
        "dec":               47.29012333,
        "parallax_raw":      0.50,
        "parallax_error":    0.08,
        "r_opt":             7.6,
        "sin_beta_target":   0.84,
        "delta_sin_beta":    0.08,
        "mg_target":        17.58,
        "delta_mg_rel":      0.087,
        "mbr_target":        1.29,
        "delta_mbr":         0.76,
        "pi0_ding":          0.004,
        "pi0_ding_err":      0.018,
        "pre_dist_x0_kpc":   None,
        "pre_dist_x0_err":   None,
    },
}

# Mean Ding filter widths — sensible defaults for non-Ding sources
DING_MEAN_FILTERS = {
    "r_deg":          7.6,     # mean search radius
    "delta_sin_beta": 0.0725,
    "delta_mg_rel":   0.143,
    "delta_mbr":      0.6175,
}

# Fortin → Ding name map (for pre-filling filter priors)
FORTIN_TO_DING = {
    "V* V1229 Aql":  "Cyg X-2",     # Fortin Main_ID for Cyg X-2
    "Cyg X-2":       "Cyg X-2",
    "Cen X-4":       "Cen X-4",
    "4U 0919-54":    "4U 0919-54",
    "XB 2129+47":    "XB 2129+47",
    "X Cyg X-2":     "Cyg X-2",
}


def get_ding_priors(source_name: str) -> dict | None:
    """Return Ding filter priors for a source if it is one of the 4 Ding targets."""
    return DING_TARGETS.get(FORTIN_TO_DING.get(source_name, source_name), None)


def gaia_adql_query_quasars(ra: float, dec: float, radius_deg: float) -> str:
    """ADQL to fetch quasars from gaiadr3.agn_cross_id around (ra, dec)."""
    return f"""
SELECT
    s.source_id, s.ra, s.dec,
    s.parallax, s.parallax_error,
    s.pmra, s.pmdec, s.pmra_error, s.pmdec_error,
    s.phot_g_mean_mag, s.bp_rp
FROM
    gaiadr3.gaia_source AS s
JOIN
    gaiadr3.agn_cross_id AS agn ON s.source_id = agn.source_id
WHERE
    DISTANCE({ra}, {dec}, s.ra, s.dec) < {radius_deg}
    AND s.bp_rp IS NOT NULL
    AND s.parallax IS NOT NULL
    AND s.parallax_error IS NOT NULL
    AND ABS(s.pmra  / NULLIF(s.pmra_error,  0)) <= 3
    AND ABS(s.pmdec / NULLIF(s.pmdec_error, 0)) <= 3
"""


def gaia_adql_target_astrometry(source_id: int) -> str:
    """ADQL to fetch astrometry + photometry for a single source_id."""
    return f"""
SELECT source_id, ra, dec, parallax, parallax_error,
       phot_g_mean_mag, bp_rp
FROM gaiadr3.gaia_source
WHERE source_id = {source_id}
"""


def weighted_parallax_zero_point(
    parallaxes: np.ndarray,
    errors: np.ndarray,
) -> dict:
    """Inverse-variance weighted mean parallax and formal uncertainty."""
    w    = 1.0 / errors**2
    pi0  = float(np.sum(w * parallaxes) / np.sum(w))
    sig  = float(np.sqrt(1.0 / np.sum(w)))
    spi0 = float(np.std(parallaxes))
    return dict(pi0=pi0, sigma=sig, scatter=spi0, n=len(parallaxes))


def apply_ding_filters(
    quasars_df: pd.DataFrame,
    sin_beta:         np.ndarray,
    mg_target:        float,
    mbr_target:       float,
    sin_beta_target:  float,
    delta_sin_beta:   float,
    delta_mg_rel:     float,
    delta_mbr:        float,
) -> np.ndarray:
    """Apply Ding-style (sin β, mG, mBR) filters. Returns boolean mask."""
    mg  = quasars_df["phot_g_mean_mag"].to_numpy()
    mbr = quasars_df["bp_rp"].to_numpy()
    mask_beta = np.abs(sin_beta - sin_beta_target) <= delta_sin_beta
    mask_mg   = np.abs(mg  - mg_target)            <= mg_target * delta_mg_rel
    mask_mbr  = np.abs(mbr - mbr_target)           <= delta_mbr
    return mask_beta & mask_mg & mask_mbr


def w_index(parallaxes: np.ndarray, errors: np.ndarray,
            s_glb: float = 0.32) -> float:
    """Ding (2021) w-index for filter optimisation."""
    if len(parallaxes) < 50:
        return -np.inf
    w_arr = 1.0 / errors**2
    mean  = np.sum(w_arr * parallaxes) / np.sum(w_arr)
    var_w = np.sum(w_arr * (parallaxes - mean)**2) / np.sum(w_arr)
    s_star = np.sqrt(var_w) / s_glb
    return float(np.log10(len(parallaxes)) - np.log10(max(s_star, 1e-9)))


def bailer_jones_posterior(
    parallax_mas:     float,
    parallax_err_mas: float,
    L_pc:             float = 1350.0,
    r_min_pc:         float = 100.0,
    r_max_pc:         float = 50000.0,
    n_grid:           int   = 4000,
) -> dict:
    """
    Bailer-Jones (2015) exponentially decreasing space density posterior.
    Returns r (pc), normalised posterior, mode, and 68% credible interval.
    """
    omega = parallax_mas     / 1000.0
    sigma = parallax_err_mas / 1000.0
    r     = np.linspace(r_min_pc, r_max_pc, n_grid)
    log_post = (
        2.0 * np.log(r) - r / L_pc
        - 0.5 * ((omega - 1.0 / r) / sigma) ** 2
    )
    log_post -= log_post.max()
    post  = np.exp(log_post)
    post /= np.trapezoid(post, r)
    mode  = float(r[np.argmax(post)])
    cdf   = np.cumsum(post) * (r[1] - r[0])
    cdf  /= cdf[-1]
    lo    = float(r[np.searchsorted(cdf, 0.159)])
    hi    = float(r[np.searchsorted(cdf, 0.841)])
    return dict(r=r, posterior=post, mode=mode, lo=lo, hi=hi)
