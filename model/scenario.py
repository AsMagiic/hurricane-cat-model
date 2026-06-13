"""
Deterministic named-storm runner (v4 calibration substrate).

Given a storm (name, year), this module:
  1. Loads the real HURDAT2 track and observed intensities.
  2. Selects the Florida landfall fix and derives deterministic storm parameters
     via the Vickery-Wadhera mean functions (no RNG).
  3. Densifies the track to ~hourly spacing via linear interpolation.
  4. Runs the Holland wind field over the synthetic FL portfolio.
  5. Applies the exposure-agnostic loss kernel to produce a per-location damage vector.

Zero RNG draws anywhere in this module. The stochastic simulation baseline
(model/loss.py, model/hazard.py) is unaffected.

Track array column convention (matches wind_at_locations):
  col 0: lat        (degrees N)
  col 1: lon        (degrees W, negative float)
  col 2: vmax_step  (mph, 1-min sustained, observed NHC intensity — NOT K-D decay)
  col 3: cum_dist_km (km, cumulative haversine along the polyline, first = 0)
"""

import math
import os

import numpy as np
import pandas as pd

# Reused from existing modules (verified against HEAD)
from calibration.parse_hurdat2 import parse
from model.units            import kt_to_mph
from model.geo_utils        import haversine, bearing
from model.wind_field       import StormParams, wind_at_locations
from model.vulnerability    import GUST_FACTOR, build_event_kernel
from model.loss             import compute_event_loss
from model.exposure_io      import load_oed_exposure, OED_LOC_PATH, OED_ACC_PATH
from model                  import hazard as _hazard
from model_config           import load_model_cfg   # root-level (not model.model_config)
from calibration.wind_pressure import predict_dp

# ---------------------------------------------------------------------------
# Module-level config (loaded once at import; zero RNG)
# ---------------------------------------------------------------------------
_mcfg           = load_model_cfg()
_fl_bbox        = _mcfg.scenario.florida_bbox       # .lat_min/.lat_max/.lon_min/.lon_max
_interp_step_hr = float(_mcfg.scenario.track_interp_step_hr)
_p_env_mb       = float(_mcfg.wind_pressure.p_env_mb)
_wpr_a          = float(_mcfg.wind_pressure.a)
_wpr_b_exp      = float(_mcfg.wind_pressure.b)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_SCENARIO_DIR = os.path.join(_ROOT, "results", "scenarios")


# ---------------------------------------------------------------------------
# Track builder — Commit A
# ---------------------------------------------------------------------------

def load_storm(name: str, year: int, hurdat2_path=None) -> pd.DataFrame:
    """
    Return all HURDAT2 fixes for the storm matching (name, year).

    Parameters
    ----------
    name         : str -- storm name, case-insensitive (e.g. 'ANDREW' or 'andrew')
    year         : int -- calendar year (e.g. 1992)
    hurdat2_path : str | None -- path to HURDAT2 file; None uses the configured default

    Returns
    -------
    pd.DataFrame -- one row per fix; columns: storm_id, name, datetime, record_id,
                    status, lat (degrees N), lon (degrees W, negative), vmax_kt (kt),
                    pmin_mb (mb)

    Raises
    ------
    ValueError if 0 or more than 1 storm_id matches (name, year).
    """
    df = parse() if hurdat2_path is None else parse(path=hurdat2_path)

    mask = (
        df["name"].str.strip().str.upper() == name.strip().upper()
    ) & (
        df["datetime"].dt.year == year
    )
    storm_df = df[mask].copy()

    n_ids = storm_df["storm_id"].nunique()
    if n_ids == 0:
        raise ValueError(f"No storm found: name={name!r}, year={year}")
    if n_ids > 1:
        ids = storm_df["storm_id"].unique().tolist()
        raise ValueError(
            f"Ambiguous: name={name!r}, year={year} matches {n_ids} storm_ids: {ids}"
        )

    return storm_df.reset_index(drop=True)


def _pick_landfall(storm_df, fl_bbox, lats=None, lons=None):
    """
    Select the Florida landfall fix.

    Primary  : among record_id == 'L' fixes inside fl_bbox, return the one with
               max vmax_kt. Multiple FL landfalls (e.g. Ian 2022 — Cayo Costa and
               a second fix) are handled by picking the peak-intensity fix.
    Fallback : if no 'L' fix exists inside fl_bbox, return the fix of closest
               haversine approach to the portfolio centroid (mean of lats/lons).

    Parameters
    ----------
    storm_df : pd.DataFrame -- filtered storm fixes from load_storm
    fl_bbox  : _NS          -- bounding box (.lat_min, .lat_max, .lon_min, .lon_max)
    lats     : array-like | None -- portfolio latitudes (degrees N) for fallback centroid
    lons     : array-like | None -- portfolio longitudes (degrees W) for fallback centroid

    Returns
    -------
    pd.Series -- one row from storm_df

    Raises
    ------
    ValueError if storm_df is empty.
    """
    if len(storm_df) == 0:
        raise ValueError("storm_df is empty; cannot pick a landfall fix")

    in_box = (
        (storm_df["record_id"] == "L") &
        (storm_df["lat"] >= fl_bbox.lat_min) &
        (storm_df["lat"] <= fl_bbox.lat_max) &
        (storm_df["lon"] >= fl_bbox.lon_min) &
        (storm_df["lon"] <= fl_bbox.lon_max)
    )
    candidates = storm_df[in_box]
    if len(candidates) > 0:
        return candidates.loc[candidates["vmax_kt"].idxmax()]

    # Fallback: closest approach to portfolio centroid
    if lats is not None and lons is not None:
        centroid_lat = float(np.mean(lats))
        centroid_lon = float(np.mean(lons))
    else:
        centroid_lat = float(storm_df["lat"].mean())
        centroid_lon = float(storm_df["lon"].mean())

    dists = storm_df.apply(
        lambda row: haversine(row["lat"], row["lon"], centroid_lat, centroid_lon),
        axis=1,
    )
    return storm_df.loc[dists.idxmin()]


def _prev_fix(storm_df, landfall_fix):
    """
    Return the fix immediately preceding landfall_fix in chronological order.

    If landfall_fix is the first chronological fix, returns the next fix instead
    (heading and vt are then computed in the forward direction in _build_storm_params).

    Parameters
    ----------
    storm_df     : pd.DataFrame -- full storm fix table from load_storm
    landfall_fix : pd.Series    -- the selected landfall fix

    Returns
    -------
    pd.Series -- adjacent fix

    Raises
    ------
    ValueError if the storm has fewer than 2 distinct-timestamp fixes.
    """
    df = storm_df.sort_values("datetime").reset_index(drop=True)
    lf_time = landfall_fix["datetime"]

    earlier = df[df["datetime"] < lf_time]
    if len(earlier) > 0:
        return earlier.iloc[-1]

    later = df[df["datetime"] > lf_time]
    if len(later) == 0:
        raise ValueError("Storm has only one distinct-timestamp fix; cannot compute heading/vt")
    return later.iloc[0]


def _interpolate_track(storm_df, step_hr=None) -> pd.DataFrame:
    """
    Linearly interpolate lat (degrees N), lon (degrees W), vmax_kt (kt),
    pmin_mb (mb) to a superset of the original fix times plus interior hourly
    grid points.

    Every original fix datetime is preserved exactly in the output, guaranteeing
    that interpolated values at those timestamps equal the observed values to
    floating-point precision.

    Guard: if two consecutive fixes share the same timestamp (dt_hr == 0), that
    zero-duration segment is skipped without error.

    Parameters
    ----------
    storm_df : pd.DataFrame -- storm fixes (any order; sorted internally by datetime)
    step_hr  : float | None -- interior grid step in hours; defaults to config value

    Returns
    -------
    pd.DataFrame -- columns: [datetime, lat, lon, vmax_kt, pmin_mb], sorted by datetime
    """
    if step_hr is None:
        step_hr = _interp_step_hr

    df = storm_df.sort_values("datetime").reset_index(drop=True)
    all_rows = []

    for i in range(len(df)):
        row_i = df.iloc[i]
        all_rows.append({
            "datetime": row_i["datetime"],
            "lat":      row_i["lat"],
            "lon":      row_i["lon"],
            "vmax_kt":  row_i["vmax_kt"],
            "pmin_mb":  row_i["pmin_mb"],
        })

        if i < len(df) - 1:
            row_j = df.iloc[i + 1]
            t0 = row_i["datetime"]
            t1 = row_j["datetime"]
            dt_hr = (t1 - t0).total_seconds() / 3600.0

            if dt_hr == 0.0:  # zero-duration segment guard
                continue

            # Interior points at step_hr, 2*step_hr, ..., strictly less than dt_hr
            for delta in np.arange(step_hr, dt_hr, step_hr):
                frac = delta / dt_hr
                all_rows.append({
                    "datetime": t0 + pd.Timedelta(hours=float(delta)),
                    "lat":      row_i["lat"]     + frac * (row_j["lat"]     - row_i["lat"]),
                    "lon":      row_i["lon"]     + frac * (row_j["lon"]     - row_i["lon"]),
                    "vmax_kt":  row_i["vmax_kt"] + frac * (row_j["vmax_kt"] - row_i["vmax_kt"]),
                    "pmin_mb":  row_i["pmin_mb"] + frac * (row_j["pmin_mb"] - row_i["pmin_mb"]),
                })

    return pd.DataFrame(all_rows).sort_values("datetime").reset_index(drop=True)


def _build_track_array(interp_df) -> np.ndarray:
    """
    Build the (N, 4) track array consumed by wind_at_locations.

    Columns:
      0: lat        (degrees N)
      1: lon        (degrees W, negative float)
      2: vmax_step  (mph, 1-min sustained) = kt_to_mph(interp_df.vmax_kt)
         Observed NHC intensity, NOT Kaplan-DeMaria model decay.
         The max-over-track envelope in wind_at_locations naturally attenuates
         distant fixes, so the full track with real intensities is correct.
      3: cum_dist_km (km) -- cumulative haversine along the interpolated polyline;
         first entry is 0.

    Parameters
    ----------
    interp_df : pd.DataFrame -- output of _interpolate_track

    Returns
    -------
    np.ndarray -- shape (N, 4), dtype float64
    """
    lats     = interp_df["lat"].to_numpy(dtype=float)
    lons     = interp_df["lon"].to_numpy(dtype=float)
    vmax_mph = kt_to_mph(interp_df["vmax_kt"].to_numpy(dtype=float))

    cum_dist = np.zeros(len(interp_df))
    for i in range(1, len(interp_df)):
        cum_dist[i] = cum_dist[i - 1] + haversine(
            lats[i - 1], lons[i - 1], lats[i], lons[i]
        )

    return np.column_stack([lats, lons, vmax_mph, cum_dist])


def _build_storm_params(landfall_fix, prev_fix) -> StormParams:
    """
    Construct StormParams from the landfall fix and the adjacent preceding fix.
    Zero RNG draws; all quantities derived deterministically.

    Parameters
    ----------
    landfall_fix : pd.Series -- the selected FL landfall fix
    prev_fix     : pd.Series -- fix immediately before landfall in time
                               (or after, if landfall was the first fix)

    Returns
    -------
    StormParams -- rmax (km), heading_deg (degrees), vt_kmh (km/h), b, dp_mb (mb),
                   lat (degrees N)

    Derived quantities
    ------------------
    heading_deg : bearing(prev → landfall) or (landfall → next), meteorological
                  convention (0 = N, 90 = E, clockwise)
    vt_kmh      : haversine distance / |time delta| in km/h;
                  fallback: 1.0 h if delta is zero (guarded)
    dp_mb       : p_env_mb - pmin_landfall (mb);
                  if pmin_mb is NaN → predict_dp(kt_to_mph(vmax_kt), a, b_exp)
    rmax_km     : exp(_hazard._vw_rmax_mean(dp_mb, lat)) [km];
                  floor applied if _hazard._RMAX_FLOOR == 'on'
    b           : clip(_hazard._vw_b_mean(rmax_km, lat),
                       _hazard._VW_B_MIN, _hazard._VW_B_MAX)
    lat         : landfall_fix.lat (degrees N, Coriolis input to StormParams)
    """
    lat = float(landfall_fix["lat"])
    lf_time = landfall_fix["datetime"]
    pv_time = prev_fix["datetime"]

    if pv_time < lf_time:
        # Normal case: prev is before landfall — bearing from prev toward landfall
        heading_deg = bearing(
            float(prev_fix["lat"]), float(prev_fix["lon"]),
            lat, float(landfall_fix["lon"]),
        )
        dt_hr = (lf_time - pv_time).total_seconds() / 3600.0
    else:
        # Edge case: landfall was first fix — bearing from landfall toward next fix
        heading_deg = bearing(
            lat, float(landfall_fix["lon"]),
            float(prev_fix["lat"]), float(prev_fix["lon"]),
        )
        dt_hr = (pv_time - lf_time).total_seconds() / 3600.0

    if dt_hr <= 0.0:  # guard against coincident timestamps
        dt_hr = 1.0

    dist_km = haversine(
        float(prev_fix["lat"]), float(prev_fix["lon"]),
        lat, float(landfall_fix["lon"]),
    )
    vt_kmh = dist_km / dt_hr

    # Pressure deficit
    pmin = float(landfall_fix["pmin_mb"])
    vmax_kt = float(landfall_fix["vmax_kt"])
    if math.isnan(pmin):
        vmax_mph_lf = float(kt_to_mph(vmax_kt))
        dp_mb = float(predict_dp(vmax_mph_lf, _wpr_a, _wpr_b_exp))
    else:
        dp_mb = float(_p_env_mb - pmin)

    # Rmax (km): deterministic V&W mean + optional physical floor
    ln_rmax = _hazard._vw_rmax_mean(dp_mb, lat)
    rmax_km = float(np.exp(ln_rmax))
    if _hazard._RMAX_FLOOR == "on":
        rmax_km = max(rmax_km, float(_hazard._RMAX_FLOOR_KM))

    # Holland B: deterministic V&W mean, censored to physical bounds
    b_raw = _hazard._vw_b_mean(rmax_km, lat)
    b = float(np.clip(b_raw, _hazard._VW_B_MIN, _hazard._VW_B_MAX))

    return StormParams(
        rmax=rmax_km,
        heading_deg=float(heading_deg),
        vt_kmh=vt_kmh,
        b=b,
        dp_mb=dp_mb,
        lat=lat,
    )


# ---------------------------------------------------------------------------
# run_scenario — Commit B
# ---------------------------------------------------------------------------

def run_scenario(
    name: str,
    year: int,
    hurdat2_path=None,
    loc_path=None,
    acc_path=None,
):
    """
    Deterministic named-storm runner. Zero RNG draws anywhere in this function.

    Pipeline
    --------
    1.  load_storm           → storm_df (HURDAT2 fixes)
    2.  _pick_landfall       → landfall_fix (max-vmax FL 'L' fix)
    3.  _prev_fix            → prev_fix (for heading / vt)
    4.  _interpolate_track   → interp_df (~hourly dense track)
    5.  _build_track_array   → track  shape (N, 4)
    6.  _build_storm_params  → StormParams (V&W mean, no RNG)
    7.  load_oed_exposure    → exp_df (from loc_path / acc_path)
    8.  build_event_kernel   → vuln_kernel (ndarray constructions)
    9.  wind_at_locations    → footprint  shape (n_loc,)  [mph, 1-min sustained]
    10. compute_event_loss   → ground_up, gross, dr  each shape (n_loc,)  [USD, USD, fraction]
    11. _write_outputs       → results/scenarios/{NAME}_{YEAR}_*.csv

    Parameters
    ----------
    name         : str -- storm name, case-insensitive
    year         : int -- calendar year
    hurdat2_path : str | None -- HURDAT2 file; None → configured default
    loc_path     : str | None -- OED location CSV; None → canonical OED_LOC_PATH
    acc_path     : str | None -- OED account CSV;  None → canonical OED_ACC_PATH

    Returns
    -------
    footprint  : (n_loc,) float64 -- max 1-min sustained wind at each location (mph)
    ground_up  : (n_loc,) float64 -- ground-up loss (USD)
    gross      : (n_loc,) float64 -- gross loss after deductible / limit (USD)
    dr         : (n_loc,) float64 -- realized damage ratio in [0, 1]

    Notes
    -----
    Footprint is 1-min sustained mph (the gust factor 1.3 lives inside
    compute_event_loss, not in the footprint) — apples-to-apples with NHC for
    Step 3 footprint validation.
    """
    # Track and storm parameters (HURDAT2-sourced, deterministic)
    storm_df     = load_storm(name, year, hurdat2_path)
    landfall_fix = _pick_landfall(storm_df, _fl_bbox)
    prev         = _prev_fix(storm_df, landfall_fix)
    interp_df    = _interpolate_track(storm_df)
    track        = _build_track_array(interp_df)
    storm_params = _build_storm_params(landfall_fix, prev)

    # Exposure (load from paths, NOT from loss.py module globals)
    exp_df = load_oed_exposure(
        loc_path if loc_path is not None else OED_LOC_PATH,
        acc_path if acc_path is not None else OED_ACC_PATH,
    )
    tivs         = exp_df["tiv"].to_numpy(dtype=float)
    deductibles  = exp_df["deductible"].to_numpy(dtype=float)
    pol_limits   = exp_df["limit"].to_numpy(dtype=float)
    lats         = exp_df["lat"].to_numpy(dtype=float)
    lons         = exp_df["lon"].to_numpy(dtype=float)
    n_loc        = len(exp_df)
    vuln_kernel  = build_event_kernel(exp_df["construction"].to_numpy())

    # Wind field (1-min sustained mph per location, max over all track steps)
    footprint = wind_at_locations(track, storm_params, lats, lons)

    # Loss (deterministic; dmg_rng=None means mean vulnerability curve)
    ground_up, gross, dr = compute_event_loss(
        footprint,
        tivs=tivs,
        deductibles=deductibles,
        pol_limits=pol_limits,
        gust_factors=np.full(n_loc, GUST_FACTOR),
        vuln_kernel=vuln_kernel,
        dmg_rng=None,
    )

    _write_outputs(name, year, exp_df, footprint, ground_up, gross, dr)
    return footprint, ground_up, gross, dr


def _write_outputs(name, year, exp_df, footprint, ground_up, gross, dr):
    """Write per-location and portfolio-level CSVs to results/scenarios/."""
    os.makedirs(_RESULTS_SCENARIO_DIR, exist_ok=True)
    stem = f"{name.upper()}_{year}"

    pd.DataFrame({
        "location_id":      exp_df["location_id"],
        "lat":              exp_df["lat"],
        "lon":              exp_df["lon"],
        "wind_sustained_mph": footprint,
    }).to_csv(
        os.path.join(_RESULTS_SCENARIO_DIR, f"{stem}_footprint.csv"), index=False
    )

    pd.DataFrame({
        "location_id": exp_df["location_id"],
        "tiv":         exp_df["tiv"],
        "ground_up":   ground_up,
        "gross":       gross,
        "dr":          dr,
    }).to_csv(
        os.path.join(_RESULTS_SCENARIO_DIR, f"{stem}_damage.csv"), index=False
    )

    tivs = exp_df["tiv"].to_numpy(dtype=float)
    rows = [{"scope": "portfolio", "tiv": tivs.sum(),
             "ground_up": ground_up.sum(), "gross": gross.sum(),
             "dr_mean": float(dr.mean())}]
    for ctype in sorted(exp_df["construction"].unique()):
        mask = (exp_df["construction"] == ctype).to_numpy()
        rows.append({
            "scope":      ctype,
            "tiv":        tivs[mask].sum(),
            "ground_up":  ground_up[mask].sum(),
            "gross":      gross[mask].sum(),
            "dr_mean":    float(dr[mask].mean()) if mask.sum() > 0 else 0.0,
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(_RESULTS_SCENARIO_DIR, f"{stem}_summary.csv"), index=False
    )
