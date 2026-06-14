"""
Spatial footprint validation: modelled wind radii vs NHC best-track observations.

Two wind fields are computed for each storm:

  Instantaneous landfall footprint — single-step track at the landfall fix only
  (lat/lon/vmax from that fix, cum_dist=0). Used for R34/R50/R64 quadrant-radii
  comparison. NHC best-track radii are also instantaneous (snapshot at a single
  analysis time), so this is an apples-to-apples comparison.

  Max-over-track envelope — full interpolated track, max wind at each grid point.
  Used for Cat-3+ swath metrics. This is the damage-relevant footprint.

The instantaneous vs envelope contrast diagnoses the source of any size bias:
  - Instantaneous compact, envelope broad  → along-track translation smearing
    (the storm traverses many grid points as it moves; geometry, not physics)
  - Instantaneous already broad            → V&W Rmax physics over-estimates size

Wind speeds: 1-min sustained (mph) — NHC best-track convention.
NO gust factor applied. The gust factor (1.3 in config) is a structural load factor
applied inside the vulnerability kernel; it has no place in a meteorological
wind-radii comparison.

Quadrant convention: GEOGRAPHIC NE/SE/SW/NW (absolute bearing from the landfall
centre). NHC stores R34/R50/R64 in geographic quadrants. Storm-relative quadrants
(forward-right, forward-left, etc.) would produce an apples-to-oranges comparison.

Distance units: nm reported externally (NHC convention); km used internally.
Grid quantisation: ~step_deg × 60 nm/deg ≈ 3 nm at default 0.05°. Differences
vs observed < 5 nm are within grid quantisation noise and should not be
over-interpreted.
"""

import os

import numpy as np
import pandas as pd

from model.scenario import prepare_storm
from model.units    import kt_to_mph
from model.geo_utils import haversine, bearing
from model.wind_field import wind_at_locations
from model_config   import load_model_cfg

# Private HURDAT2 constants — imported as constants, parser is NOT called here
from calibration.parse_hurdat2 import _RAW as _HURDAT2_DEFAULT, _MISS as _HURDAT2_MISS

_mcfg      = load_model_cfg()
_grid_step = float(_mcfg.scenario.grid_step_deg)
_grid_pad  = float(_mcfg.scenario.grid_pad_deg)

_KM_PER_NM = 1.852
_QUADRANTS = ["NE", "SE", "SW", "NW"]

# Standard NHC wind-radii thresholds (kt) and the Cat-3+ threshold
_THRESHOLDS_KT   = [64, 50, 34]
_CAT3PLUS_KT     = 96
_THRESH_LABELS   = ["R64", "R50", "R34", "Cat3plus"]
_THRESH_KT_ALL   = [64, 50, 34, _CAT3PLUS_KT]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def build_grid(track_arr, *, step_deg=None, pad_deg=None):
    """
    Build a regular lat/lon grid covering the storm track.

    The grid bounds are derived from the track's lat/lon extremes, expanded by
    pad_deg in each direction to ensure the outermost wind threshold ring (34 kt,
    typically 100-200 nm radius) is fully captured.

    Parameters
    ----------
    track_arr : (N, 4) ndarray  -- columns [lat °N, lon °W, vmax mph, cum_dist km]
    step_deg  : float | None    -- grid step in degrees; None → config value (~0.05°)
    pad_deg   : float | None    -- bbox padding in degrees; None → config value (2°)

    Returns
    -------
    grid_lats : (M,) float64   -- flattened grid latitudes (°N)
    grid_lons : (M,) float64   -- flattened grid longitudes (°W, negative)
    """
    step = step_deg if step_deg is not None else _grid_step
    pad  = pad_deg  if pad_deg  is not None else _grid_pad

    lat_min = track_arr[:, 0].min() - pad
    lat_max = track_arr[:, 0].max() + pad
    lon_min = track_arr[:, 1].min() - pad
    lon_max = track_arr[:, 1].max() + pad

    lat_vals = np.arange(lat_min, lat_max + step * 0.5, step)
    lon_vals = np.arange(lon_min, lon_max + step * 0.5, step)

    GLat, GLon = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    return GLat.ravel(), GLon.ravel()


# ---------------------------------------------------------------------------
# Quadrant classifier
# ---------------------------------------------------------------------------

def _geo_quadrant(bearing_arr):
    """
    Classify bearings into geographic NHC quadrants.

    NHC stores R34/R50/R64 in geographic quadrants (absolute bearing):
      NE = [  0°,  90°)   SE = [ 90°, 180°)
      SW = [180°, 270°)   NW = [270°, 360°)

    Parameters
    ----------
    bearing_arr : array-like -- bearings in degrees (any range; mod 360 applied)

    Returns
    -------
    ndarray of str ('NE'|'SE'|'SW'|'NW') -- one label per input bearing
    """
    b = np.asarray(bearing_arr, dtype=float) % 360.0
    q = np.empty(b.shape, dtype=object)
    q[(b >= 0)   & (b < 90)]  = "NE"
    q[(b >= 90)  & (b < 180)] = "SE"
    q[(b >= 180) & (b < 270)] = "SW"
    q[(b >= 270)]              = "NW"
    return q


# ---------------------------------------------------------------------------
# Wind radii and swath extent computation
# ---------------------------------------------------------------------------

def compute_radii(grid_lats, grid_lons, wind_mph, landfall_fix, heading_deg):
    """
    Compute wind radii by geographic quadrant and cross/along-track swath metrics.

    For each threshold (64 kt, 50 kt, 34 kt, 96 kt Cat-3+):
      Per-quadrant max radius: the greatest distance from the landfall centre at
        which wind >= threshold, measured separately in each geographic quadrant.
      Max radius (all quadrants): max of the four quadrant values.
      Cross-track width: max – min of the signed cross-track displacement (km/nm)
        of all points satisfying wind >= threshold. Translation-insensitive.
      Along-track extent: max – min of the signed along-track displacement.
        For the envelope field this quantifies along-track smearing.

    All radii are grid-quantised to ~step_deg × 60 nm/deg ≈ 3 nm at 0.05°.

    Coordinate system (meteorological heading θ, 0=N clockwise):
      Along-track unit vector in (x=east, y=north): (sin θ, cos θ)
      Cross-track unit vector (right of track):      (cos θ, –sin θ)
    Positive cross-track = right of track; positive along-track = ahead of storm.

    Parameters
    ----------
    grid_lats    : (M,) float64  -- grid latitudes (°N)
    grid_lons    : (M,) float64  -- grid longitudes (°W, negative)
    wind_mph     : (M,) float64  -- wind speed (mph, 1-min sustained)
    landfall_fix : pd.Series     -- HURDAT2 row; must have 'lat' and 'lon' fields
    heading_deg  : float         -- storm heading (meteorological degrees)

    Returns
    -------
    dict -- keys: 'R64', 'R50', 'R34', 'Cat3plus'
      Each value is a dict:
        'NE', 'SE', 'SW', 'NW' : float (nm) or nan  -- quadrant max radii
        'max_radius_nm'         : float (nm) or nan  -- max across all quadrants
        'cross_track_nm'        : float (nm) or nan  -- cross-track width
        'along_track_nm'        : float (nm) or nan  -- along-track extent
    """
    lf_lat = float(landfall_fix["lat"])
    lf_lon = float(landfall_fix["lon"])

    dist_nm = haversine(lf_lat, lf_lon, grid_lats, grid_lons) / _KM_PER_NM
    brg     = bearing(lf_lat, lf_lon, grid_lats, grid_lons)
    quad    = _geo_quadrant(brg)

    # Cross/along-track coordinates (physical km, then converted to nm)
    cos_lat = np.cos(np.radians(lf_lat))
    dlat_km = (grid_lats - lf_lat) * 111.32
    dlon_km = (grid_lons - lf_lon) * 111.32 * cos_lat
    theta   = np.radians(heading_deg)
    along_km = dlon_km * np.sin(theta) + dlat_km * np.cos(theta)
    cross_km = dlon_km * np.cos(theta) - dlat_km * np.sin(theta)
    along_nm = along_km / _KM_PER_NM
    cross_nm = cross_km / _KM_PER_NM

    result = {}
    for label, thresh_kt in zip(_THRESH_LABELS, _THRESH_KT_ALL):
        thresh_mph = float(kt_to_mph(thresh_kt))
        mask       = wind_mph >= thresh_mph

        entry = {}
        for q in _QUADRANTS:
            qmask = mask & (quad == q)
            entry[q] = float(dist_nm[qmask].max()) if qmask.any() else np.nan

        entry["max_radius_nm"] = (
            float(dist_nm[mask].max()) if mask.any() else np.nan
        )
        if mask.any():
            entry["cross_track_nm"] = float(cross_nm[mask].max() - cross_nm[mask].min())
            entry["along_track_nm"] = float(along_nm[mask].max() - along_nm[mask].min())
        else:
            entry["cross_track_nm"] = np.nan
            entry["along_track_nm"] = np.nan

        result[label] = entry

    return result


# ---------------------------------------------------------------------------
# HURDAT2 best-track radii reader (lightweight; does NOT call the parser)
# ---------------------------------------------------------------------------

def read_hurdat2_radii(landfall_fix, hurdat2_path=None):
    """
    Read the observed R34/R50/R64 quadrant radii for a single HURDAT2 fix.

    Performs a lightweight line-level scan of the raw HURDAT2 file keyed on
    storm_id + datetime. The production parser (calibration/parse_hurdat2.py) is
    NOT called — this function operates at the raw text level.

    HURDAT2 data-line field layout (0-indexed, comma-separated):
      0: YYYYMMDD   1: HHMM   2: record_id   3: status
      4: lat        5: lon    6: vmax_kt      7: pmin_mb
      8-11 : R34 NE/SE/SW/NW (nm)
      12-15: R50 NE/SE/SW/NW (nm)
      16-19: R64 NE/SE/SW/NW (nm)
    Missing sentinel: -999.  Pre-2004 storms have -999 for all radii.

    Parameters
    ----------
    landfall_fix  : pd.Series   -- HURDAT2 fix row; must have 'storm_id' and 'datetime'
    hurdat2_path  : str | None  -- path to raw HURDAT2 file; None → configured default

    Returns
    -------
    observed    : dict -- keys 'R64', 'R50', 'R34', each a dict with 'NE','SE','SW','NW'
                         (float nm or nan)
    has_observed: bool -- True if any non-nan radii found; False for pre-2004 storms
    """
    path      = hurdat2_path if hurdat2_path is not None else _HURDAT2_DEFAULT
    lf_sid    = str(landfall_fix["storm_id"]).strip()
    lf_dt     = pd.Timestamp(landfall_fix["datetime"])
    miss      = int(_HURDAT2_MISS)

    observed = {
        "R64": {q: np.nan for q in _QUADRANTS},
        "R50": {q: np.nan for q in _QUADRANTS},
        "R34": {q: np.nan for q in _QUADRANTS},
    }
    has_observed = False

    current_sid = None
    with open(path, encoding="ascii", errors="replace") as fh:
        for raw_line in fh:
            line   = raw_line.strip()
            if not line:
                continue
            fields = [f.strip() for f in line.split(",")]
            first  = fields[0]

            # Header line: starts with basin code letters (AL, EP, CP …)
            if first and first[0].isalpha():
                current_sid = first
                continue

            if current_sid != lf_sid:
                continue

            # Data line: YYYYMMDD, HHMM, …
            if len(fields) < 8:
                continue
            try:
                date_s = first          # YYYYMMDD
                time_s = fields[1].zfill(4)  # HHMM
                # parse_hurdat2.py uses utc=True → timestamps are tz-aware UTC
                row_dt = pd.Timestamp(
                    f"{date_s[:4]}-{date_s[4:6]}-{date_s[6:8]}"
                    f" {time_s[:2]}:{time_s[2:]}",
                    tz="UTC",
                )
            except Exception:
                continue

            if row_dt != lf_dt:
                continue

            # Matched: parse radii fields
            def _parse_quad(start):
                d = {}
                for i, q in enumerate(_QUADRANTS):
                    try:
                        v = int(fields[start + i])
                        d[q] = np.nan if v == miss else float(v)
                    except (IndexError, ValueError):
                        d[q] = np.nan
                return d

            if len(fields) >= 12:
                observed["R34"] = _parse_quad(8)
            if len(fields) >= 16:
                observed["R50"] = _parse_quad(12)
            if len(fields) >= 20:
                observed["R64"] = _parse_quad(16)

            has_observed = any(
                not np.isnan(v)
                for radii in observed.values()
                for v in radii.values()
            )
            break  # found the line — stop scanning

    return observed, has_observed


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_footprint_validation(name, year, hurdat2_path=None, *,
                             results_dir=None, grid_step_deg=None):
    """
    Evaluate the deterministic wind field on a regular grid and compute wind radii.

    Pipeline
    --------
    1. prepare_storm(name, year)            → track (N,4), storm_params, landfall_fix
    2. build_grid(track)                    → grid_lats (M,), grid_lons (M,)
    3. Instantaneous wind field:
         single-row track [lf_lat, lf_lon, kt_to_mph(lf_vmax_kt), 0]
         wind_at_locations(lf_track, storm_params, grid)
         → wind_instant_mph (M,)  [1-min sustained, NO gust factor]
    4. Envelope wind field:
         wind_at_locations(track, storm_params, grid)
         → wind_envelope_mph (M,) [max over full track]
    5. compute_radii(grid, wind_instant, lf, heading) → radii_instant
       compute_radii(grid, wind_envelope, lf, heading) → radii_envelope
    6. read_hurdat2_radii(lf, hurdat2_path) → observed, has_observed
    7. _write_radii_csv(…)                 → results/scenarios/{NAME}_{YEAR}_radii.csv
    8. _plot_footprint(…)                  → results/scenarios/{NAME}_{YEAR}_footprint.png

    The instantaneous vs envelope contrast:
      If R64 from instantaneous is compact but envelope Cat-3+ is broad, the breadth
      is along-track translation smearing (geometry). If instantaneous R64 is already
      broad, the V&W Rmax physics over-estimates the storm's actual size.

    Parameters
    ----------
    name         : str      -- storm name, case-insensitive (e.g. 'ANDREW')
    year         : int      -- calendar year (e.g. 1992)
    hurdat2_path : str|None -- HURDAT2 file; None → configured default
    results_dir  : str|None -- output directory; None → results/scenarios/
    grid_step_deg: float|None -- grid resolution override (degrees); useful for
                                  faster test runs (e.g. 0.5°)

    Returns
    -------
    dict with keys:
      'instantaneous_wind_mph' : (M,) ndarray -- wind from single-step landfall track
      'envelope_wind_mph'      : (M,) ndarray -- wind from full multi-step track
      'radii_instant'          : dict  -- R64/R50/R34/Cat3plus from instantaneous
      'radii_envelope'         : dict  -- R64/R50/R34/Cat3plus from envelope
      'observed'               : dict  -- best-track R34/R50/R64 (nm, nan if missing)
      'has_observed'           : bool
      'grid_lats'              : (M,) ndarray
      'grid_lons'              : (M,) ndarray
      'track'                  : (N, 4) ndarray
      'storm_params'           : StormParams
      'landfall_fix'           : pd.Series
    """
    track, storm_params, landfall_fix = prepare_storm(name, year, hurdat2_path)

    grid_lats, grid_lons = build_grid(track, step_deg=grid_step_deg)

    # Instantaneous: single-row track at the landfall fix
    lf_vmax_mph = float(kt_to_mph(float(landfall_fix["vmax_kt"])))
    lf_track = np.array([[
        float(landfall_fix["lat"]),
        float(landfall_fix["lon"]),
        lf_vmax_mph,
        0.0,
    ]])
    wind_instant  = wind_at_locations(lf_track, storm_params, grid_lats, grid_lons)

    # Envelope: full interpolated track (max over all steps)
    wind_envelope = wind_at_locations(track, storm_params, grid_lats, grid_lons)

    heading = storm_params.heading_deg
    radii_instant  = compute_radii(grid_lats, grid_lons, wind_instant,  landfall_fix, heading)
    radii_envelope = compute_radii(grid_lats, grid_lons, wind_envelope, landfall_fix, heading)

    observed, has_observed = read_hurdat2_radii(landfall_fix, hurdat2_path)

    rdir = results_dir or os.path.join(_ROOT, "results", "scenarios")
    os.makedirs(rdir, exist_ok=True)

    _write_radii_csv(name, year, radii_instant, radii_envelope,
                     observed, has_observed, rdir)
    _plot_footprint(name, year, grid_lats, grid_lons,
                    wind_instant, wind_envelope, landfall_fix, rdir)

    return {
        "instantaneous_wind_mph": wind_instant,
        "envelope_wind_mph":      wind_envelope,
        "radii_instant":          radii_instant,
        "radii_envelope":         radii_envelope,
        "observed":               observed,
        "has_observed":           has_observed,
        "grid_lats":              grid_lats,
        "grid_lons":              grid_lons,
        "track":                  track,
        "storm_params":           storm_params,
        "landfall_fix":           landfall_fix,
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _write_radii_csv(name, year, radii_instant, radii_envelope,
                     observed, has_observed, results_dir):
    """
    Write tidy radii comparison table to results_dir/{NAME}_{YEAR}_radii.csv.

    Columns: storm, year, metric, threshold_kt, quadrant, wind_source,
             modeled_nm, observed_nm, delta_nm

    Rows:
      One row per (threshold_kt, quadrant) for R34/R50/R64 from the instantaneous field.
      Two summary rows for Cat-3+ cross-track and along-track from the envelope field.
    """
    rows = []
    stem = f"{name.upper()}_{year}"

    for label, thresh_kt in zip(["R64", "R50", "R34"], [64, 50, 34]):
        mod_d  = radii_instant[label]
        obs_d  = observed.get(label, {})
        for q in _QUADRANTS:
            mod_nm = mod_d.get(q, np.nan)
            obs_nm = obs_d.get(q, np.nan) if has_observed else np.nan
            delta  = (mod_nm - obs_nm) if (not np.isnan(mod_nm) and not np.isnan(obs_nm)) else np.nan
            rows.append({
                "storm": name.upper(), "year": year,
                "metric": label, "threshold_kt": thresh_kt, "quadrant": q,
                "wind_source": "instantaneous",
                "modeled_nm": round(mod_nm, 1) if not np.isnan(mod_nm) else np.nan,
                "observed_nm": round(obs_nm, 1) if not np.isnan(obs_nm) else np.nan,
                "delta_nm": round(delta, 1) if not np.isnan(delta) else np.nan,
            })

    # Cat-3+ swath from envelope
    for metric_key, label_str in [("cross_track_nm", "Cat3plus_cross_track"),
                                   ("along_track_nm", "Cat3plus_along_track"),
                                   ("max_radius_nm",  "Cat3plus_max_radius")]:
        val = radii_envelope["Cat3plus"].get(metric_key, np.nan)
        rows.append({
            "storm": name.upper(), "year": year,
            "metric": label_str, "threshold_kt": _CAT3PLUS_KT, "quadrant": "all",
            "wind_source": "envelope",
            "modeled_nm": round(val, 1) if not np.isnan(val) else np.nan,
            "observed_nm": np.nan, "delta_nm": np.nan,
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(results_dir, f"{stem}_radii.csv"), index=False
    )


# ---------------------------------------------------------------------------
# PNG heatmap
# ---------------------------------------------------------------------------

def _plot_footprint(name, year, grid_lats, grid_lons,
                    wind_instant, wind_envelope, landfall_fix, results_dir):
    """
    Save a side-by-side wind footprint heatmap with 34/50/64/96-kt contours.

    Left panel: instantaneous landfall footprint (used for R34/R50/R64 comparison).
    Right panel: max-over-track envelope (used for Cat-3+ swath metrics).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm
    except ImportError:
        return  # matplotlib is optional; skip silently if absent

    lat_arr = np.unique(grid_lats)
    lon_arr = np.unique(grid_lons)
    M_lat, M_lon = len(lat_arr), len(lon_arr)

    # Reshape only if we have a perfect rectangular grid
    if M_lat * M_lon != len(grid_lats):
        return

    idx_lat = {v: i for i, v in enumerate(lat_arr)}
    idx_lon = {v: i for i, v in enumerate(lon_arr)}
    W_inst = np.full((M_lat, M_lon), np.nan)
    W_env  = np.full((M_lat, M_lon), np.nan)
    for k in range(len(grid_lats)):
        i = idx_lat.get(grid_lats[k])
        j = idx_lon.get(grid_lons[k])
        if i is not None and j is not None:
            W_inst[i, j] = wind_instant[k]
            W_env[i, j]  = wind_envelope[k]

    # Threshold levels in mph for contours
    thresh_mph = [float(kt_to_mph(t)) for t in [34, 50, 64, 96]]
    thresh_kt  = [34, 50, 64, 96]
    colors_ct  = ["green", "yellow", "orange", "red"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    lf_lat = float(landfall_fix["lat"])
    lf_lon = float(landfall_fix["lon"])

    for ax, W, title in [
        (axes[0], W_inst,  f"{name} {year} — Instantaneous (R34/R50/R64)"),
        (axes[1], W_env,   f"{name} {year} — Envelope (Cat-3+ swath)"),
    ]:
        levels = np.linspace(0, max(float(W[~np.isnan(W)].max()) if (~np.isnan(W)).any() else 200, 120), 20)
        pcm = ax.contourf(lon_arr, lat_arr, W, levels=levels, cmap="YlOrRd", extend="both")
        for t_mph, t_kt, col in zip(thresh_mph, thresh_kt, colors_ct):
            try:
                cs = ax.contour(lon_arr, lat_arr, W, levels=[t_mph], colors=[col], linewidths=1.2)
                ax.clabel(cs, fmt=f"{t_kt} kt", fontsize=7, inline=True)
            except Exception:
                pass
        ax.plot(lf_lon, lf_lat, "k*", markersize=10, label="Landfall")
        ax.set_xlabel("Longitude (°W)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        plt.colorbar(pcm, ax=ax, label="Wind (mph, 1-min sustained)")

    plt.tight_layout()
    stem = f"{name.upper()}_{year}"
    plt.savefig(os.path.join(results_dir, f"{stem}_footprint.png"), dpi=120)
    plt.close(fig)
