"""
fetch_quasar_fields.py
----------------------
Query Gaia DR3 agn_cross_id for quasars around each LMXB target and
write one parquet file per source plus a combined manifest JSON.

Run once (or whenever you want to refresh the data):
    python fetch_quasar_fields.py

Output layout (under DATA_DIR/quasars/):
    manifest.json               — target metadata (coords, N_quasars, …)
    Cen_X-4.parquet
    Cyg_X-2.parquet
    4U_0919-54.parquet
    XB_2129+47.parquet
"""

import os
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.environ.get("LMXB_DATA_DIR", "/Users/aminesahraoui/lmxb-pre-gaia")
OUT_DIR    = Path(BASE_DIR) / "data" / "quasars"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    {"name": "Cen X-4",     "search_radius_deg": 5.5},
    {"name": "Cyg X-2",     "search_radius_deg": 5.5},
    {"name": "4U 0919-54",  "search_radius_deg": 5.5},
    {"name": "XB 2129+47",  "search_radius_deg": 5.5},
]

# Gaia query: only sources with well-constrained proper motion and colour
ADQL_TEMPLATE = """
SELECT
    s.source_id,
    s.ra,
    s.dec,
    s.parallax,
    s.parallax_error,
    s.pmra,
    s.pmdec,
    s.pmra_error,
    s.pmdec_error,
    s.phot_g_mean_mag,
    s.phot_bp_mean_mag,
    s.phot_rp_mean_mag,
    s.bp_rp,
    s.ruwe,
    agn.clean_quasar_catalogue,
    agn.table_name AS agn_catalog
FROM
    gaiadr3.gaia_source AS s
JOIN
    gaiadr3.agn_cross_id AS agn ON s.source_id = agn.source_id
WHERE
    DISTANCE({ra:.6f}, {dec:.6f}, s.ra, s.dec) < {radius:.4f}
    AND s.bp_rp IS NOT NULL
    AND s.phot_g_mean_mag IS NOT NULL
    AND ABS(s.pmra  / NULLIF(s.pmra_error,  0)) <= 3
    AND ABS(s.pmdec / NULLIF(s.pmdec_error, 0)) <= 3
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_filename(name: str) -> str:
    """Turn 'Cyg X-2' → 'Cyg_X-2' (safe for filesystems)."""
    return re.sub(r"\s+", "_", name.strip())


def angular_separation_deg(ra1, dec1, ra_arr, dec_arr):
    """Vectorised angular separation in degrees (small-angle safe via astropy)."""
    c1   = SkyCoord(ra=ra1, dec=dec1, unit="deg")
    cref = SkyCoord(ra=ra_arr, dec=dec_arr, unit="deg")
    return c1.separation(cref).deg


# ── Main fetch loop ───────────────────────────────────────────────────────────
Gaia.ROW_LIMIT = -1          # no row cap
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

manifest = []

for target in TARGETS:
    name   = target["name"]
    radius = target["search_radius_deg"]
    fname  = safe_filename(name)

    print(f"\n{'─'*60}")
    print(f"  {name}  (radius = {radius}°)")
    print(f"{'─'*60}")

    # Resolve coordinates
    try:
        coord = SkyCoord.from_name(name)
    except Exception as e:
        print(f"  [ERROR] Could not resolve coordinates: {e}")
        continue

    ra_deg  = coord.ra.deg
    dec_deg = coord.dec.deg
    print(f"  Coords  :  RA={ra_deg:.5f}  Dec={dec_deg:.5f}")

    # Launch async Gaia query
    query = ADQL_TEMPLATE.format(ra=ra_deg, dec=dec_deg, radius=radius)
    try:
        job     = Gaia.launch_job_async(query, verbose=False)
        results = job.get_results()
    except Exception as e:
        print(f"  [ERROR] Gaia query failed: {e}")
        continue

    n = len(results)
    print(f"  Quasars found : {n}")

    if n == 0:
        print("  Skipping — no quasars in field.")
        manifest.append({
            "name":       name,
            "ra_deg":     round(ra_deg,  5),
            "dec_deg":    round(dec_deg, 5),
            "radius_deg": radius,
            "n_quasars":  0,
            "file":       None,
        })
        continue

    # Convert to pandas, add derived columns
    df = results.to_pandas()

    # Angular separation from target
    df["sep_deg"] = angular_separation_deg(
        ra_deg, dec_deg,
        df["ra"].to_numpy(), df["dec"].to_numpy()
    )

    # Magnitude columns: ensure float
    for col in ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
                "bp_rp", "parallax", "parallax_error", "ruwe"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Tag target
    df["target"] = name

    # Sort by angular separation
    df = df.sort_values("sep_deg").reset_index(drop=True)

    # Write parquet
    out_path = OUT_DIR / f"{fname}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved   : {out_path}  ({df.shape[1]} cols)")

    # Summary stats for manifest
    manifest.append({
        "name":           name,
        "ra_deg":         round(ra_deg,  5),
        "dec_deg":        round(dec_deg, 5),
        "radius_deg":     radius,
        "n_quasars":      n,
        "g_mag_min":      round(float(df["phot_g_mean_mag"].min()), 3),
        "g_mag_max":      round(float(df["phot_g_mean_mag"].max()), 3),
        "g_mag_median":   round(float(df["phot_g_mean_mag"].median()), 3),
        "bp_rp_median":   round(float(df["bp_rp"].median()), 3),
        "sep_max_deg":    round(float(df["sep_deg"].max()), 4),
        "file":           f"{fname}.parquet",
    })

# ── Write manifest ────────────────────────────────────────────────────────────
manifest_path = OUT_DIR / "manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n{'='*60}")
print(f"  Manifest written : {manifest_path}")
print(f"  Targets processed: {len(manifest)}")
total_q = sum(t["n_quasars"] for t in manifest)
print(f"  Total quasars    : {total_q:,}")
print(f"{'='*60}\n")
