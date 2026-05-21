"""
download_gaia_cache.py
----------------------
Run once from the project root:

    python download_gaia_cache.py

Fetches Gaia DR3 quasar fields and target astrometry for every source in the
Fortin LMXB catalog that has a Gaia counterpart, then writes:

    data/gaia_cache/quasars.parquet   — all quasar rows, with a 'source' column
    data/gaia_cache/targets.parquet   — one astrometry row per source

Re-running is safe: sources already present in the cache are skipped unless
you pass --force.
"""

import argparse
import io
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = "/Users/aminesahraoui/lmxb-pre-gaia"
CACHE_DIR   = BASE_DIR + "/data" + "/gaia_cache"
LMXB_PATH   = BASE_DIR + "/data" + "/raw"  + "/fortin_lmxb" +"/LMXBwebcat_latest.csv"
QUASAR_FILE = CACHE_DIR + "/quasars.parquet"
TARGET_FILE = CACHE_DIR + "/targets.parquet"

TAP_ENDPOINTS = [
    "https://gea.esac.esa.int/tap-server/tap/sync",
    "https://gaiatap.esac.esa.int/tap-server/tap/sync",
]

# Search radius used for quasar cone — generous enough to cover any Ding-optimal
# radius; the page can filter further in software.
DEFAULT_RADIUS_DEG = 11.0

RETRY_DELAY  = 5   # seconds between retries
MAX_RETRIES  = 3


# ── TAP helper ────────────────────────────────────────────────────────────────
def tap_query(adql: str, timeout: int = 180) -> pd.DataFrame:
    """POST an ADQL query to the Gaia TAP and return a DataFrame."""
    from astropy.io.votable import parse_single_table

    params = {
        "REQUEST": "doQuery",
        "LANG":    "ADQL",
        "FORMAT":  "votable",
        "QUERY":   adql,
    }

    last_err = None
    for attempt in range(MAX_RETRIES):
        for url in TAP_ENDPOINTS:
            try:
                resp = requests.post(url, data=params, timeout=timeout, verify=False)
                resp.raise_for_status()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tbl = parse_single_table(io.BytesIO(resp.content))
                df = tbl.to_table().to_pandas()
                # unmask masked arrays
                for col in df.columns:
                    if hasattr(df[col], "filled"):
                        df[col] = df[col].filled(np.nan)
                return df
            except Exception as e:
                last_err = e
                print(f"    ↳ {url} failed ({e}), trying next…")
        if attempt < MAX_RETRIES - 1:
            print(f"  All endpoints failed, retrying in {RETRY_DELAY}s…")
            time.sleep(RETRY_DELAY)

    raise RuntimeError(f"All TAP endpoints failed after {MAX_RETRIES} attempts. "
                       f"Last error: {last_err}")


# ── ADQL templates ────────────────────────────────────────────────────────────
def adql_quasars(ra: float, dec: float, radius_deg: float) -> str:
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


def adql_target(source_id: int) -> str:
    return f"""
SELECT source_id, ra, dec, parallax, parallax_error,
       phot_g_mean_mag, bp_rp
FROM gaiadr3.gaia_source
WHERE source_id = {source_id}
"""


# ── Main ──────────────────────────────────────────────────────────────────────
def main(force: bool = False) -> None:
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    # Load Fortin catalog
    lmxb = pd.read_csv(LMXB_PATH)
    gaia_sources = lmxb[
        lmxb["Gaia_Distance"].notna() &
        lmxb["Gaia_ID"].notna()
    ].copy()
    gaia_sources["label"] = (
        gaia_sources["Popular_ID"].fillna(gaia_sources["Main_ID"])
    )

    print(f"Found {len(gaia_sources)} Fortin sources with Gaia counterparts.\n")

    # Load existing cache if present
    if Path(QUASAR_FILE).exists() and not force:
        q_existing = pd.read_parquet(QUASAR_FILE)
        already_done_q = set(q_existing["source"].unique())
    else:
        q_existing = pd.DataFrame()
        already_done_q = set()

    if Path(TARGET_FILE).exists() and not force:
        t_existing = pd.read_parquet(TARGET_FILE)
        already_done_t = set(t_existing["source"].unique())
    else:
        t_existing = pd.DataFrame()
        already_done_t = set()

    new_quasar_frames = []
    new_target_frames = []

    for _, row in gaia_sources.iterrows():
        label     = row["label"]
        ra        = float(row["RAdeg"])
        dec       = float(row["DEdeg"])
        try:
            gaia_id = int(float(row["Gaia_ID"]))
        except (ValueError, TypeError):
            gaia_id = None

        # ── Quasar field ──────────────────────────────────────────────────
        if label in already_done_q and not force:
            print(f"[SKIP quasars] {label} — already cached")
        else:
            print(f"[FETCH quasars] {label}  (ra={ra:.3f}, dec={dec:.3f}, r={DEFAULT_RADIUS_DEG}°)")
            try:
                df_q = tap_query(adql_quasars(ra, dec, DEFAULT_RADIUS_DEG))
                df_q = df_q.dropna(subset=["parallax","parallax_error",
                                            "phot_g_mean_mag","bp_rp"])
                df_q["source"]    = label
                df_q["src_ra"]    = ra
                df_q["src_dec"]   = dec
                new_quasar_frames.append(df_q)
                print(f"  → {len(df_q)} quasars fetched")
            except Exception as e:
                print(f"  ✗ FAILED: {e}")

        # ── Target astrometry ─────────────────────────────────────────────
        if label in already_done_t and not force:
            print(f"[SKIP target]  {label} — already cached")
        else:
            if gaia_id is None:
                print(f"[SKIP target]  {label} — no Gaia source_id")
                continue
            print(f"[FETCH target] {label}  (source_id={gaia_id})")
            try:
                df_t = tap_query(adql_target(gaia_id))
                if len(df_t) == 0:
                    print(f"  ✗ No rows returned for source_id={gaia_id}")
                    continue
                df_t["source"] = label
                new_target_frames.append(df_t)
                row_t = df_t.iloc[0]
                print(f"  → π = {row_t.get('parallax','?'):.3f} ± "
                      f"{row_t.get('parallax_error','?'):.3f} mas")
            except Exception as e:
                print(f"  ✗ FAILED: {e}")

        # Brief pause to avoid hammering the archive
        time.sleep(1.0)

    # ── Write / merge cache ───────────────────────────────────────────────────
    if new_quasar_frames:
        new_q = pd.concat(new_quasar_frames, ignore_index=True)
        combined_q = pd.concat([q_existing, new_q], ignore_index=True) \
                     if not q_existing.empty else new_q
        combined_q.to_parquet(QUASAR_FILE, index=False)
        print(f"\n✓ Quasar cache: {len(combined_q)} rows → {QUASAR_FILE}")
    else:
        print("\nNo new quasar data to write.")

    if new_target_frames:
        new_t = pd.concat(new_target_frames, ignore_index=True)
        combined_t = pd.concat([t_existing, new_t], ignore_index=True) \
                     if not t_existing.empty else new_t
        combined_t.to_parquet(TARGET_FILE, index=False)
        print(f"✓ Target cache: {len(combined_t)} rows → {TARGET_FILE}")
    else:
        print("No new target data to write.")

    print("\nDone. Re-run with --force to refresh all sources.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Gaia quasar cache.")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch all sources even if already cached.")
    args = parser.parse_args()
    main(force=args.force)