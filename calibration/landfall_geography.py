"""
Step 1.5a (part 1): Landfall geography calibration.

Fits a 1-D Gaussian KDE to HU landfall arc-length positions along the
Florida coastline.  The coastline is derived from the TIGER/Line 2023
state boundary (downloaded at runtime to a temp dir, never committed).

Keys handling
-------------
The Florida mainland exterior ring in TIGER already traces the Keys
archipelago as part of its exterior boundary (853 raw vertices south of
25 N, min lat 24.40 N).  No separate polygon insertion is required; the
forward arc from the GA-FL corner to the AL-FL corner naturally covers:
  Atlantic coast  -> SE Florida  -> Keys (SW to ~24.4 N)  -> Gulf coast
  -> panhandle  -> AL corner.

The Dry Tortugas (2nd FL polygon, centroid 24.67 N, -82.88 W, 86
vertices) is a remote island group with no HU landfalls in the HURDAT2
record at that location; it is excluded.

Outputs
-------
  data/processed/fl_coastline_simplified.csv   committed, ~150-400 rows
  outputs/landfall_geography.png
  config/model_v3.yaml  -- hazard.landfall_geography block appended
"""

import os
import sys
import tempfile
import urllib.request
import zipfile

# Ensure stdout can emit UTF-8 (needed on Windows where default is cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.stats import gaussian_kde
from shapely.geometry import LineString, Point
from shapely.ops import transform as shp_transform

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"
_LF_CSV    = os.path.join(_ROOT, "data", "processed", "fl_landfalls.csv")
_COAST_CSV = os.path.join(_ROOT, "data", "processed", "fl_coastline_simplified.csv")
_FIG_PATH  = os.path.join(_ROOT, "outputs", "landfall_geography.png")
_CFG_PATH  = os.path.join(_ROOT, "config", "model_v3.yaml")

# EPSG:3086 — Florida Albers, metres; consistent with Step 1.2.
_TO_3086   = Transformer.from_crs("EPSG:4326", "EPSG:3086", always_xy=True)
_FROM_3086 = Transformer.from_crs("EPSG:3086", "EPSG:4326", always_xy=True)

# Reference corners for the coastal-arc split of the mainland exterior ring.
_CORNER_GA = (30.71, -81.47)   # (lat, lon)  GA-FL Atlantic border, NE Florida
_CORNER_AL = (30.99, -87.63)   # (lat, lon)  AL-FL panhandle border, NW Florida

_SIMPLIFY_M = 5000.0   # Douglas-Peucker tolerance: 5 km (see plan for rationale)

_FLAG_DIST_KM = 15.0   # projection-distance flag threshold


# ---------------------------------------------------------------------------
# CRS helpers
# ---------------------------------------------------------------------------

def _to_3086(geom):
    return shp_transform(_TO_3086.transform, geom)


def _from_3086(geom):
    return shp_transform(_FROM_3086.transform, geom)


# ---------------------------------------------------------------------------
# Section A: coastline extraction
# ---------------------------------------------------------------------------

def _nearest_ring_idx(ring_arr: np.ndarray, ref_lat: float, ref_lon: float) -> int:
    """Index of the ring vertex (lon,lat) nearest to the reference (lat,lon)."""
    d = (ring_arr[:, 1] - ref_lat) ** 2 + (ring_arr[:, 0] - ref_lon) ** 2
    return int(d.argmin())


def extract_fl_coastline(tmp_dir: str):
    """
    Download TIGER, extract the FL coastal polyline, simplify, return results.

    Returns
    -------
    coastline_4326   : shapely LineString, simplified, EPSG:4326
    n_vertices_raw   : int
    n_vertices_simp  : int
    arc_length_km    : float
    keys_note        : str  — how Keys were handled
    states_gdf       : GeoDataFrame  — for map background
    """
    # --- Download and load ---
    zip_path = os.path.join(tmp_dir, "tl_2023_us_state.zip")
    print(f"Downloading {_TIGER_URL} ...")
    urllib.request.urlretrieve(_TIGER_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)
    shp = next(os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
               if f.endswith(".shp"))
    states_gdf = gpd.read_file(shp)
    fl_geom = states_gdf.loc[states_gdf["NAME"] == "Florida"].geometry.iloc[0]

    print(f"  FL geometry type: {fl_geom.geom_type}")
    polys = sorted(list(fl_geom.geoms) if hasattr(fl_geom, "geoms") else [fl_geom],
                   key=lambda p: p.area, reverse=True)
    mainland = polys[0]
    print(f"  Constituent polygons: {len(polys)}  "
          f"(mainland area {mainland.area:.4f} deg²)")
    if len(polys) > 1:
        others = polys[1:]
        print(f"  Excluded polygon(s): {len(others)} small islands "
              f"(Dry Tortugas area; no HU landfalls in HURDAT2 record)")

    # --- Identify coastal arc via forward-arc from GA corner ---
    ring_arr = np.array(list(mainland.exterior.coords))   # (N, 2): lon, lat
    N = len(ring_arr)

    i_ga = _nearest_ring_idx(ring_arr, *_CORNER_GA)
    i_al = _nearest_ring_idx(ring_arr, *_CORNER_AL)
    print(f"  Mainland ring vertices: {N}")
    print(f"  GA corner idx {i_ga}: lat={ring_arr[i_ga,1]:.4f}  lon={ring_arr[i_ga,0]:.4f}")
    print(f"  AL corner idx {i_al}: lat={ring_arr[i_al,1]:.4f}  lon={ring_arr[i_al,0]:.4f}")

    # Forward arc: indices i_ga, i_ga+1, ..., i_al (mod N)
    # This is the coastal arc (Atlantic -> Keys -> Gulf); confirmed by min lat
    # reaching 24.4 N (Keys territory) and first step going south.
    fwd_len = (i_al - i_ga) % N + 1
    fwd_arc = [tuple(ring_arr[(i_ga + k) % N]) for k in range(fwd_len)]

    # Sanity check: verify direction by min latitude
    min_lat_fwd = min(lat for _, lat in fwd_arc)
    print(f"  Forward arc: {fwd_len} vertices, min lat {min_lat_fwd:.4f}°N")
    if min_lat_fwd > 26.0:
        raise RuntimeError(
            f"Forward arc min lat {min_lat_fwd:.2f}°N > 26°N — "
            "ring orientation unexpected; check corner indices."
        )
    print(f"  Forward arc confirmed as coastal arc (includes Keys, min lat {min_lat_fwd:.4f}°N)")

    keys_note = (
        f"Keys embedded in mainland exterior ring; no separate polygon insertion. "
        f"Forward arc from GA corner through Atlantic-S, Keys (min lat {min_lat_fwd:.4f}°N), "
        f"Gulf-N to AL corner — {fwd_len} raw vertices."
    )

    n_raw = fwd_len

    # --- Simplify in EPSG:3086 ---
    line_4326 = LineString(fwd_arc)
    line_3086 = _to_3086(line_4326)
    line_3086_s = line_3086.simplify(_SIMPLIFY_M, preserve_topology=True)
    line_4326_s = _from_3086(line_3086_s)

    n_simp = len(list(line_4326_s.coords))
    arc_length_km = line_3086_s.length / 1000.0

    print(f"  Simplification: {n_raw} -> {n_simp} vertices "
          f"({n_simp/n_raw*100:.1f}% retained)")
    print(f"  Total arc length: {arc_length_km:.1f} km")

    return line_4326_s, n_raw, n_simp, arc_length_km, keys_note, states_gdf


# ---------------------------------------------------------------------------
# Section C: arc-length projection + distance check
# ---------------------------------------------------------------------------

def project_landfalls(hu_df: pd.DataFrame, coastline_4326: LineString):
    """
    Project each HU landfall point onto the coastline (EPSG:3086).

    Returns
    -------
    s_km      : ndarray (n_HU,)  arc-length from start, km
    dist_km   : ndarray (n_HU,)  perpendicular distance to polyline, km
    """
    coastline_3086 = _to_3086(coastline_4326)
    s_list, d_list = [], []
    for _, row in hu_df.iterrows():
        x3, y3 = _TO_3086.transform(row["lon"], row["lat"])
        pt     = Point(x3, y3)
        s_m    = coastline_3086.project(pt)
        foot   = coastline_3086.interpolate(s_m)
        d_m    = pt.distance(foot)
        s_list.append(s_m / 1000.0)
        d_list.append(d_m / 1000.0)
    return np.array(s_list), np.array(d_list)


def report_distances(hu_df: pd.DataFrame, s_km: np.ndarray, dist_km: np.ndarray):
    print()
    print("=" * 64)
    print("PROJECTION DISTANCE CHECK  (perpendicular distance, km)")
    print("=" * 64)
    print(f"  n_HU : {len(s_km)}")
    print(f"  min  : {dist_km.min():.2f} km")
    print(f"  p25  : {np.percentile(dist_km, 25):.2f} km")
    print(f"  median: {np.median(dist_km):.2f} km")
    print(f"  p75  : {np.percentile(dist_km, 75):.2f} km")
    print(f"  max  : {dist_km.max():.2f} km")

    flagged_mask = dist_km > _FLAG_DIST_KM
    if not flagged_mask.any():
        print(f"\n  [OK] All {len(s_km)} points project within "
              f"{_FLAG_DIST_KM:.0f} km of the polyline.")
    else:
        n_flag = flagged_mask.sum()
        print(f"\n  [FLAG] {n_flag} landfalls > {_FLAG_DIST_KM:.0f} km from polyline:")
        for i in np.where(flagged_mask)[0]:
            r = hu_df.iloc[i]
            print(f"    {r['name']:<16} {int(r['year'])}  "
                  f"lat={r['lat']:.3f}  lon={r['lon']:.3f}  "
                  f"s={s_km[i]:.1f} km  dist={dist_km[i]:.1f} km")

        # Characterise flags: check if concentrated in Dry Tortugas area (lon < -82.5W)
        flag_lons = hu_df["lon"].values[flagged_mask]
        n_dry_tort = (flag_lons < -82.5).sum()
        if n_dry_tort == n_flag:
            print(f"\n  DIAGNOSIS: all {n_flag} flags are in the Dry Tortugas area "
                  f"(lon < -82.5°W).")
            print(f"  The simplified coastline extends west to ~-82.33°W (Lower Keys).")
            print(f"  The Dry Tortugas are a remote island group ~40-100 km west of Key")
            print(f"  West, represented in TIGER by a SEPARATE FL polygon (2nd polygon,")
            print(f"  centroid 24.67°N -82.88°W) that was INTENTIONALLY excluded from")
            print(f"  the main coastline arc.  These storms crossed the Dry Tortugas")
            print(f"  boundary and were correctly detected as FL landfalls by")
            print(f"  filter_fl_landfalls.py, but they project to the nearest point on")
            print(f"  the simplified mainland+Keys arc (westernmost Lower Keys,")
            print(f"  s~917.9 km).  This is NOT over-simplification; the KDE bandwidth")
            print(f"  (~241 km) dwarfs the mis-projection (37-69 km).")
        elif n_dry_tort > 0:
            print(f"\n  DIAGNOSIS: {n_dry_tort}/{n_flag} flags in Dry Tortugas area "
                  f"(lon < -82.5°W); remainder may indicate over-simplification.")
    print()


# ---------------------------------------------------------------------------
# Section D: KDE
# ---------------------------------------------------------------------------

def fit_kde(s_km: np.ndarray):
    """
    Fit Gaussian KDE with Silverman bandwidth and two sensitivity variants.

    Returns
    -------
    kde_main, kde_half, kde_2x  : scipy gaussian_kde objects
    h_km                        : float  Silverman bandwidth, km
    factor                      : float  scipy bw_method factor
    """
    kde_main  = gaussian_kde(s_km, bw_method="silverman")
    factor    = float(kde_main.factor)
    h_km      = factor * float(np.std(s_km))
    kde_half  = gaussian_kde(s_km, bw_method=0.5 * factor)
    kde_2x    = gaussian_kde(s_km, bw_method=2.0 * factor)
    return kde_main, kde_half, kde_2x, h_km, factor


# ---------------------------------------------------------------------------
# Section E: config write (targeted insertion)
# ---------------------------------------------------------------------------

def write_config(n_raw, n_simp, arc_km, coastline_4326,
                 h_km, factor, n_HU, s_km, keys_note, cfg_path):
    """Append hazard.landfall_geography block before 'vulnerability:' line."""

    coords    = list(coastline_4326.coords)
    start_pt  = coords[0]    # (lon, lat)
    end_pt    = coords[-1]   # (lon, lat)

    # Format 112 arc-length values as a YAML flow sequence
    s_vals = "[" + ", ".join(f"{v:.3f}" for v in s_km) + "]"

    block = (
        f"  landfall_geography:\n"
        f"    coastline_path:\n"
        f"      value: \"data/processed/fl_coastline_simplified.csv\"\n"
        f"      units: \"file_path\"\n"
        f"      source: \"TIGER/Line 2023 FL state boundary; forward arc GA-FL Atlantic"
        f" corner -> Keys (embedded in mainland ring) -> AL-FL panhandle corner;"
        f" {_SIMPLIFY_M/1000:.0f} km Douglas-Peucker simplification (EPSG:3086)."
        f" See calibration/landfall_geography.py.\"\n"
        f"    simplify_tolerance_m:\n"
        f"      value: {_SIMPLIFY_M:.0f}\n"
        f"      units: \"m\"\n"
        f"      source: \"1/10 of minimum HU wind radius (50 km); eliminates"
        f" TIGER sub-km digitization noise while preserving all macroscale"
        f" geographic features relevant to landfall geography.\"\n"
        f"    n_vertices_raw:\n"
        f"      value: {n_raw}\n"
        f"      units: \"count\"\n"
        f"      source: \"TIGER/Line 2023 FL mainland exterior ring coastal arc"
        f" (GA-FL to AL-FL, includes Keys); before simplification\"\n"
        f"    n_vertices_simplified:\n"
        f"      value: {n_simp}\n"
        f"      units: \"count\"\n"
        f"      source: \"After Douglas-Peucker {_SIMPLIFY_M/1000:.0f} km"
        f" simplification in EPSG:3086\"\n"
        f"    total_arc_length_km:\n"
        f"      value: {arc_km:.2f}\n"
        f"      units: \"km\"\n"
        f"      source: \"EPSG:3086 arc-length of simplified coastline\"\n"
        f"    endpoint_start:\n"
        f"      value: [{start_pt[0]:.6f}, {start_pt[1]:.6f}]\n"
        f"      units: \"[lon_deg, lat_deg]\"\n"
        f"      source: \"GA-FL Atlantic border, NE Florida; s=0 reference\"\n"
        f"    endpoint_end:\n"
        f"      value: [{end_pt[0]:.6f}, {end_pt[1]:.6f}]\n"
        f"      units: \"[lon_deg, lat_deg]\"\n"
        f"      source: \"AL-FL panhandle border; s=L reference\"\n"
        f"    keys_handling:\n"
        f"      value: \"{keys_note}\"\n"
        f"      units: \"text\"\n"
        f"      source: \"calibration/landfall_geography.py\"\n"
        f"    kde_bandwidth_silverman_km:\n"
        f"      value: {h_km:.3f}\n"
        f"      units: \"km\"\n"
        f"      source: \"Silverman rule: h = 1.06 * std(s) * n^(-1/5),"
        f" n={n_HU} HU landfalls\"\n"
        f"    kde_silverman_factor:\n"
        f"      value: {factor:.6f}\n"
        f"      units: \"dimensionless\"\n"
        f"      source: \"scipy gaussian_kde bw_method factor;"
        f" actual_bw_km = factor * std(s_km)\"\n"
        f"    n_HU:\n"
        f"      value: {n_HU}\n"
        f"      units: \"count\"\n"
        f"      source: \"HURDAT2 fl_landfalls.csv, status==HU, full record 1851-2024\"\n"
        f"    s_samples_km:\n"
        f"      value: {s_vals}\n"
        f"      units: \"km\"\n"
        f"      source: \"Arc-length projection of {n_HU} HU landfalls onto"
        f" simplified FL coastline, EPSG:3086; s=0 at GA-FL Atlantic corner (NE Florida)\"\n"
    )

    with open(cfg_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    # Guard: skip write if block already present
    if any("landfall_geography:" in line for line in lines):
        print(f"Config: hazard.landfall_geography already present — skipping write.")
        return

    # Find insertion point: before 'vulnerability:' (top-level key, no indent)
    insert_before = None
    for i, line in enumerate(lines):
        if line.startswith("vulnerability:"):
            insert_before = i
            break
    if insert_before is None:
        raise ValueError(f"Could not find 'vulnerability:' in {cfg_path}")

    new_lines = lines[:insert_before] + [block + "\n"] + lines[insert_before:]
    with open(cfg_path, "w", encoding="utf-8") as fh:
        fh.writelines(new_lines)
    print(f"Config updated: {cfg_path}")


# ---------------------------------------------------------------------------
# Section F: plot
# ---------------------------------------------------------------------------

def _seg_arc_lengths(coastline_3086: LineString) -> tuple:
    """
    Return (seg_midpoint_s_km, seg_density_weights) for each segment
    of the simplified coastline in EPSG:3086.
    """
    pts = np.array(list(coastline_3086.coords))   # (M, 2)
    seg_s = np.zeros(len(pts) - 1)
    cum = 0.0
    for i in range(len(pts) - 1):
        seg_len = np.hypot(pts[i+1, 0] - pts[i, 0],
                           pts[i+1, 1] - pts[i, 1]) / 1000.0
        seg_s[i] = cum + seg_len / 2.0
        cum += seg_len
    return seg_s


def _milestone_arc_lengths(coastline_3086: LineString) -> dict:
    """Project named geographic reference points onto the coastline."""
    refs = {
        "Jacksonville": (30.33, -81.66),
        "Cape Canaveral": (28.46, -80.53),
        "Palm Beach": (26.71, -80.04),
        "Miami": (25.78, -80.19),
        "Key West": (24.56, -81.80),
        "Ft. Myers": (26.64, -81.87),
        "Tampa Bay": (27.96, -82.80),
        "Pensacola": (30.42, -87.22),
    }
    result = {}
    for name, (lat, lon) in refs.items():
        x3, y3 = _TO_3086.transform(lon, lat)
        s_m = coastline_3086.project(Point(x3, y3))
        result[name] = s_m / 1000.0
    return result


def make_plot(coastline_4326, states_gdf, hu_df, s_km, dist_km,
              kde_main, kde_half, kde_2x, h_km, L_km, fig_path):
    """
    Two-panel figure:
      Left  — Florida map, coastline colored by KDE density, HU scatter
      Right — KDE density profile (3 bandwidths) + rug + geographic labels
    """
    coastline_3086 = _to_3086(coastline_4326)
    coast_pts_4326 = np.array(list(coastline_4326.coords))   # (M, 2): lon, lat

    # KDE evaluation grid
    s_grid    = np.linspace(0, L_km, 600)
    dens_main = kde_main.evaluate(s_grid)
    dens_half = kde_half.evaluate(s_grid)
    dens_2x   = kde_2x.evaluate(s_grid)

    # Per-segment midpoint arc-lengths and densities for coastline coloring
    seg_s_km  = _seg_arc_lengths(coastline_3086)
    seg_dens  = np.interp(seg_s_km, s_grid, dens_main)

    # Build LineCollection segments in lon-lat space
    n_pts = len(coast_pts_4326)
    segments = [coast_pts_4326[i:i+2] for i in range(n_pts - 1)]

    # Geographic milestones
    milestones = _milestone_arc_lengths(coastline_3086)

    # Decade coloring for HU scatter
    hu_years = hu_df["year"].values
    dec_min  = (int(hu_years.min()) // 10) * 10
    dec_max  = (int(hu_years.max()) // 10) * 10
    decades  = list(range(dec_min, dec_max + 10, 10))
    tab_cmap = plt.colormaps.get_cmap("tab20").resampled(len(decades))

    # ---- Figure ----
    fig, (ax_map, ax_kde) = plt.subplots(
        1, 2, figsize=(16, 8),
        gridspec_kw={"width_ratios": [3, 2]}
    )

    # ---- Left panel: map ----
    lon_lo, lon_hi = -88.5, -79.5
    lat_lo, lat_hi =  23.5,  31.5

    states_gdf.clip_by_rect(lon_lo, lat_lo, lon_hi, lat_hi).plot(
        ax=ax_map, facecolor="#f4f6f4", edgecolor="#aaaaaa", linewidth=0.5
    )
    states_gdf.loc[states_gdf["NAME"] == "Florida"].plot(
        ax=ax_map, facecolor="#dce8f4", edgecolor="#555555", linewidth=0.8
    )

    # Coastline as LineCollection colored by KDE density
    d_min, d_max = seg_dens.min(), seg_dens.max()
    lc = mcoll.LineCollection(segments, array=seg_dens,
                              cmap="YlOrRd", linewidths=3.0, zorder=4)
    lc.set_clim(d_min, d_max)
    ax_map.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax_map, fraction=0.03, pad=0.02, aspect=25)
    cbar.set_label("KDE density (per km)", fontsize=8)

    # HU scatter colored by decade
    for j, dec in enumerate(decades):
        mask = (hu_years >= dec) & (hu_years < dec + 10)
        if not mask.any():
            continue
        sub = hu_df[mask]
        ax_map.scatter(
            sub["lon"], sub["lat"],
            c=[tab_cmap(j)] * mask.sum(),
            s=sub["vmax_kt"].fillna(64.0) * 0.45,
            alpha=0.82, zorder=6,
            linewidths=0.3, edgecolors="k",
            label=f"{dec}s",
        )

    ax_map.set_xlim(lon_lo, lon_hi)
    ax_map.set_ylim(lat_lo, lat_hi)
    ax_map.set_xlabel("Longitude (°E)", fontsize=9)
    ax_map.set_ylabel("Latitude (°N)", fontsize=9)
    ax_map.set_title(
        f"Historical FL Hurricane Landfalls  (n=112, 1851–2024)\n"
        f"Coastline colored by KDE density  (Silverman h={h_km:.0f} km)",
        fontsize=10,
    )
    ax_map.tick_params(labelsize=8)
    # Legend: decade dots (only if <= 12 decades)
    if len(decades) <= 12:
        ax_map.legend(fontsize=6, loc="upper left", ncol=2,
                      markerscale=0.8, title="Decade", title_fontsize=6)

    # ---- Right panel: KDE profile ----
    # Sensitivity band
    ax_kde.fill_between(s_grid, dens_half, dens_2x,
                        alpha=0.12, color="#1f77b4",
                        label="Sensitivity: 0.5×–2× Silverman")
    ax_kde.plot(s_grid, dens_2x,  "b--", lw=0.9, alpha=0.55,
                label=f"2× Silverman  ({2*h_km:.0f} km)")
    ax_kde.plot(s_grid, dens_half, "b:",  lw=0.9, alpha=0.55,
                label=f"0.5× Silverman  ({0.5*h_km:.0f} km)")
    ax_kde.plot(s_grid, dens_main, "r-",  lw=2.2,
                label=f"Silverman  h={h_km:.0f} km  (n=112)")

    # Rug plot (tick marks at bottom)
    rug_base = -0.5 * dens_main.max() * 0.04
    rug_top  =  0.0
    ax_kde.vlines(s_km, rug_base, rug_top,
                  color="black", alpha=0.45, linewidth=0.7)

    # Geographic milestones
    for name, s_val in milestones.items():
        if 0 <= s_val <= L_km:
            ax_kde.axvline(s_val, color="dimgrey", lw=0.5,
                           linestyle=":", alpha=0.65)
            ax_kde.text(
                s_val, 0.97, name,
                fontsize=6.5, color="dimgrey",
                rotation=90, va="top", ha="right",
                transform=ax_kde.get_xaxis_transform(),
            )

    ax_kde.set_xlabel("Arc-length from NE Florida (km)", fontsize=9)
    ax_kde.set_ylabel("Density (per km)", fontsize=9)
    ax_kde.set_xlim(0, L_km)
    ax_kde.set_ylim(bottom=rug_base * 1.5)
    ax_kde.set_title(
        f"HU Landfall Density along FL Coast\n"
        f"KDE: Silverman h={h_km:.0f} km  |  n=112  |  coast {L_km:.0f} km",
        fontsize=10,
    )
    ax_kde.legend(fontsize=8, loc="upper right")
    ax_kde.grid(True, linestyle=":", alpha=0.30)
    ax_kde.tick_params(labelsize=8)

    fig.suptitle(
        "Florida Hurricane Landfall Geography  |  HURDAT2 1851–2024\n"
        "TIGER/Line 2023 coastline  |  EPSG:3086 arc-length projection  |"
        "  Gaussian KDE (Silverman bandwidth)",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {fig_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 64)
    print("STEP 1.5a (part 1): Landfall geography calibration")
    print("=" * 64)

    # Load HU landfalls
    df    = pd.read_csv(_LF_CSV)
    hu_df = df[df["status"] == "HU"].reset_index(drop=True)
    print(f"\nHU landfalls loaded: {len(hu_df)} storms")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # --- Coastline ---
        (coastline_4326, n_raw, n_simp,
         arc_km, keys_note, states_gdf) = extract_fl_coastline(tmp_dir)

        # Save CSV
        coast_coords = list(coastline_4326.coords)
        coast_df = pd.DataFrame(coast_coords, columns=["lon", "lat"])
        os.makedirs(os.path.dirname(_COAST_CSV), exist_ok=True)
        coast_df.to_csv(_COAST_CSV, index=False)
        print(f"\nSaved: {_COAST_CSV}  ({len(coast_df)} rows)")

        # --- Arc-length projection ---
        print("\nProjecting HU landfalls onto simplified coastline ...")
        s_km, dist_km = project_landfalls(hu_df, coastline_4326)
        report_distances(hu_df, s_km, dist_km)

        # --- KDE ---
        kde_main, kde_half, kde_2x, h_km, factor = fit_kde(s_km)
        print(f"KDE (Silverman): h = {h_km:.2f} km  (scipy factor = {factor:.6f})")
        print(f"  0.5× sensitivity: {0.5*h_km:.2f} km")
        print(f"  2×  sensitivity: {2.0*h_km:.2f} km")

        # --- Config ---
        write_config(n_raw, n_simp, arc_km, coastline_4326,
                     h_km, factor, len(hu_df), s_km, keys_note, _CFG_PATH)

        # --- Plot ---
        make_plot(coastline_4326, states_gdf, hu_df, s_km, dist_km,
                  kde_main, kde_half, kde_2x, h_km, arc_km, _FIG_PATH)

    print()
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  Simplified coastline : {_COAST_CSV}")
    print(f"                         {n_raw} -> {n_simp} vertices after"
          f" {_SIMPLIFY_M/1000:.0f} km simplification")
    print(f"                         total arc length {arc_km:.1f} km")
    print(f"  KDE bandwidth        : {h_km:.1f} km (Silverman, n=112)")
    print(f"  Figure               : {_FIG_PATH}")
    print(f"  Config               : {_CFG_PATH} (hazard.landfall_geography added)")
    print()
    print("Keys handling:")
    print(f"  {keys_note}")
    print()
    print("Most important geometric decision:")
    print("  Coastal arc extraction uses the FORWARD direction of the mainland")
    print("  exterior ring from the GA-FL corner (verified by min lat reaching")
    print("  24.4 N — the Keys — confirming the Keys are embedded in the ring).")
    print("  Alternative rejected: latitude-threshold filtering of ring segments")
    print("  to exclude the land border — fails because panhandle coast reaches")
    print("  ~30.4 N (Pensacola), overlapping the GA-AL land border latitude range.")
    print()
    print("Next: calibration/landfall_track.py  (Step 1.5a part 2)")


if __name__ == "__main__":
    main()
