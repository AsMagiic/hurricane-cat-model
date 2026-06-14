"""
Identify historical CONUS Atlantic/Gulf hurricane landfalls from HURDAT2
using geometric coastline-crossing detection.

Generalises filter_fl_landfalls.py (FL-only) to the full CONUS east + Gulf
coast.  Three key differences:

1. Detection boundary: land_union.boundary (CONUS coastline + land borders)
   instead of fl_poly.boundary.  land_union.boundary on the CONUS MultiPolygon
   (mainland + Keys + barrier islands + Long Island) IS a MultiLineString, so
   _first_intersection handles that type explicitly.

2. Projection: EPSG:5070 (NAD83 / Conus Albers) for the interpolation-fraction
   computation.  EPSG:3086 (Florida GDL Albers) is only valid near FL; using
   it for TX or NC landfalls would bias the interpolated Vmax.

3. Bounding box: expanded to full east + Gulf coast (~-98/-66 lon, 24/47 lat).
   Forgetting this would silently drop all non-FL landfalls with no error;
   the per-state hard assert catches it.

Detection logic (unchanged from filter_fl_landfalls.py):
- Load all fixes; NHC 'L' record_id NOT used (sparsely populated pre-1990).
- For each consecutive fix pair (a, b): fix a must be over water; segment must
  cross the CONUS boundary.  Interpolate Vmax/pmin/datetime at the crossing.
- Keep the highest-Vmax crossing per storm (multiple landfalls are not
  independent WPR observations; one event = one peak-intensity sample).

US-Mexico and US-Canada land borders are technically part of land_union.boundary.
In practice they produce no false HU detections: a fix in Mexico/Canada would
cross the land border only at an inland point, and HU intensity at such a point
is physically impossible.  The hard assert in _assert_geography enforces this.

NOTE: this script is an intentional partial duplication of filter_fl_landfalls.py.
A shared calibration/landfall_detection.py refactor is a future cleanup item.

Output columns (same schema as fl_landfalls.csv):
  storm_id, name, year, datetime, lat, lon, vmax_kt, pmin_mb, status
"""

import math
import os
import tempfile
import urllib.request
import zipfile
from typing import Optional

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.prepared import prep

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from model_config import load_calibration_cfg

_ccfg = load_calibration_cfg()
_IN   = os.path.join(_ROOT, _ccfg.hurdat2.processed_path)
_OUT  = os.path.join(_ROOT, _ccfg.conus_landfalls.processed_path)
_MAP  = os.path.join(_ROOT, _ccfg.conus_landfalls.map_path)

_TIGER_URL      = "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"
_NON_CONUS_FIPS = {"02", "15", "60", "66", "69", "72", "78"}

# EPSG:5070: NAD83 / Conus Albers (metres).  Equal-area across full CONUS 24-49 N.
# Replaces EPSG:3086 (Florida GDL Albers) used by filter_fl_landfalls.py, which
# introduces distortion for landfalls outside ~24-32 N.
_PROJ = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

# Fast-reject bounding box: full east + Gulf coast.  If this is too narrow,
# non-FL landfalls are silently dropped — the per-state assert detects this.
_CONUS_LON_MIN, _CONUS_LON_MAX = -98.0, -66.0
_CONUS_LAT_MIN, _CONUS_LAT_MAX =  24.0,  47.0

_STATUS_COLOR = {"HU": "#d62728", "TS": "#ff7f0e", "TD": "#bcbd22"}
_STATUS_ORDER = ["HU", "TS", "TD"]

# All CONUS states bordering the Atlantic Ocean or Gulf of Mexico.
# A crossing outside this set means a land-border segment slipped through.
_COASTAL_STATES = {
    "Texas", "Louisiana", "Mississippi", "Alabama", "Florida",
    "Georgia", "South Carolina", "North Carolina", "Virginia",
    "Maryland", "Delaware", "New Jersey", "New York", "Connecticut",
    "Rhode Island", "Massachusetts", "New Hampshire", "Maine",
}

# Each of these must have >= 1 HU landfall in the full HURDAT2 record.
# Zero in any entry means the bbox or boundary dropped that coast segment.
_REQUIRED_HU_STATES = {
    "Texas", "Louisiana", "Mississippi", "Alabama",
    "Georgia", "South Carolina", "North Carolina",
}

# Known CONUS landfalls used to validate detection.
# (name, year, expected_states, note)
_VALIDATION_STORMS = [
    ("HUGO",    1989, {"South Carolina"},            "Cat 4 SC"),
    ("ANDREW",  1992, {"Florida"},                   "Cat 5 FL — dual-presence check"),
    ("KATRINA", 2005, {"Louisiana", "Mississippi"},  "Cat 3 LA/MS peak; FL was Cat 1"),
    ("WILMA",   2005, {"Florida"},                   "Cat 3 FL — dual-presence check"),
    ("HARVEY",  2017, {"Texas"},                     "Cat 4 TX"),
    ("IDA",     2021, {"Louisiana"},                 "Cat 4 LA"),
]


# ---------------------------------------------------------------------------
# Geometry loading
# ---------------------------------------------------------------------------

def _load_geometries(tmp_dir: str):
    """
    Download TIGER/Line 2023.  Return (conus_boundary, land_union, states_gdf).

    conus_boundary -- boundary of the CONUS land union (MultiLineString in
                      practice: mainland coast + Keys + islands + land borders)
    land_union     -- unary_union of all CONUS state polygons (water mask)
    states_gdf     -- full US states GeoDataFrame (map background)
    """
    zip_path = os.path.join(tmp_dir, "tl_2023_us_state.zip")
    print(f"Downloading {_TIGER_URL} ...")
    urllib.request.urlretrieve(_TIGER_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)
    shp = next(
        os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".shp")
    )
    states_gdf = gpd.read_file(shp)

    conus      = states_gdf.loc[~states_gdf["STATEFP"].isin(_NON_CONUS_FIPS)]
    land_union = unary_union(conus.geometry)
    conus_boundary = land_union.boundary
    print(f"  CONUS land polygon built ({len(conus)} states + DC).")
    print(f"  CONUS boundary type: {conus_boundary.geom_type}  "
          f"(MultiLineString expected for CONUS MultiPolygon)")
    return conus_boundary, land_union, states_gdf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_status(vmax_kt: float) -> str:
    """Saffir-Simpson classification from 1-min sustained wind (knots)."""
    if vmax_kt >= 64.0:
        return "HU"
    if vmax_kt >= 34.0:
        return "TS"
    return "TD"


def _interp_fraction(
    lon_a: float, lat_a: float,
    lon_c: float, lat_c: float,
    lon_b: float, lat_b: float,
) -> float:
    """
    Return f = dist(a->c) / dist(a->b) in EPSG:5070 metric distances.

    EPSG:5070 (NAD83 / Conus Albers) is equal-area across CONUS and removes
    the degree-lon != degree-lat bias from 24 N (TX) to 47 N (ME).
    Clamped to [0, 1] against floating-point overshoot at the boundary.
    """
    xa, ya = _PROJ.transform(lon_a, lat_a)
    xc, yc = _PROJ.transform(lon_c, lat_c)
    xb, yb = _PROJ.transform(lon_b, lat_b)
    dist_ab = math.hypot(xb - xa, yb - ya)
    if dist_ab < 1.0:
        return 0.0
    return max(0.0, min(1.0, math.hypot(xc - xa, yc - ya) / dist_ab))


def _first_intersection(line: LineString, boundary) -> Optional[Point]:
    """
    Return the entry crossing point: the intersection of line with boundary
    closest to line.coords[0].

    Handles all geometry types that can result from LineString.intersection():
      Point         -- single clean crossing
      LineString    -- collinear sub-segment; take the entry end
      MultiLineString -- CONUS land_union.boundary IS this type; each component
                        is a collinear sub-segment; take the start of each
      MultiPoint / GeometryCollection -- gather all sub-geometries recursively

    MultiLineString is handled EXPLICITLY (not via accidental fallthrough) because
    land_union.boundary on the CONUS MultiPolygon is always a MultiLineString.
    """
    isect = line.intersection(boundary)
    if isect.is_empty:
        return None

    def _collect(g) -> list:
        t = g.geom_type
        if t == "Point":
            return [g]
        if t == "LineString":
            return [Point(g.coords[0])]
        if t == "MultiLineString":
            # Each geom is a collinear sub-segment on the boundary.
            return [Point(seg.coords[0]) for seg in g.geoms]
        if t in ("MultiPoint", "GeometryCollection"):
            pts = []
            for sub in g.geoms:
                pts.extend(_collect(sub))
            return pts
        return []

    pts = _collect(isect)
    return min(pts, key=lambda p: line.project(p)) if pts else None


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def _detect_storm_crossings(
    storm_fixes: pd.DataFrame,
    prep_land,
    conus_boundary,
) -> list:
    """
    Detect all ocean->CONUS coastline crossings for a single storm.

    Returns a list of crossing dicts (one per detected crossing; may be empty).
    """
    rows = list(storm_fixes.itertuples(index=False))
    if len(rows) < 2:
        return []

    crossings = []
    for a, b in zip(rows[:-1], rows[1:]):
        # Fast reject: neither endpoint in the CONUS coast bbox.
        if not (
            (_CONUS_LON_MIN <= a.lon <= _CONUS_LON_MAX
             and _CONUS_LAT_MIN <= a.lat <= _CONUS_LAT_MAX)
            or
            (_CONUS_LON_MIN <= b.lon <= _CONUS_LON_MAX
             and _CONUS_LAT_MIN <= b.lat <= _CONUS_LAT_MAX)
        ):
            continue

        # Fix a must be over water — excludes overland entries.
        if prep_land.contains(Point(a.lon, a.lat)):
            continue

        seg = LineString([(a.lon, a.lat), (b.lon, b.lat)])
        if not seg.intersects(conus_boundary):
            continue

        crossing_pt = _first_intersection(seg, conus_boundary)
        if crossing_pt is None:
            continue

        f = _interp_fraction(
            a.lon, a.lat, crossing_pt.x, crossing_pt.y, b.lon, b.lat
        )

        vmax_lf = float(a.vmax_kt) + f * (float(b.vmax_kt) - float(a.vmax_kt))

        pmin_a, pmin_b = float(a.pmin_mb), float(b.pmin_mb)
        pmin_lf = (
            pmin_a + f * (pmin_b - pmin_a)
            if not (math.isnan(pmin_a) or math.isnan(pmin_b))
            else float("nan")
        )

        dt_lf = a.datetime + pd.Timedelta(
            seconds=f * (b.datetime - a.datetime).total_seconds()
        )

        crossings.append({
            "storm_id": a.storm_id,
            "name":     a.name,
            "datetime": dt_lf,
            "lat":      crossing_pt.y,
            "lon":      crossing_pt.x,
            "vmax_kt":  vmax_lf,
            "pmin_mb":  pmin_lf,
            "status":   _classify_status(vmax_lf),
        })

    return crossings


def detect_all_conus_landfalls(fixes: pd.DataFrame, land_union) -> pd.DataFrame:
    """
    Detect CONUS landfalls for all storms in the HURDAT2 fix table.

    Returns one row per storm (highest-Vmax CONUS crossing retained).
    Multiple landfalls of the same storm are not independent WPR observations;
    the highest-Vmax crossing is the physically appropriate anchor for fitting.
    """
    prep_land      = prep(land_union)
    conus_boundary = land_union.boundary

    all_crossings = []
    storm_groups  = list(fixes.groupby("storm_id", sort=False))
    n             = len(storm_groups)

    for i, (storm_id, group) in enumerate(storm_groups, 1):
        if i % 500 == 0:
            print(f"  {i}/{n} storms processed ...")
        group     = group.sort_values("datetime")
        crossings = _detect_storm_crossings(group, prep_land, conus_boundary)
        if crossings:
            all_crossings.extend(crossings)

    if not all_crossings:
        return pd.DataFrame(
            columns=["storm_id", "name", "year", "datetime",
                     "lat", "lon", "vmax_kt", "pmin_mb", "status"]
        )

    df = pd.DataFrame(all_crossings)

    # Keep the highest-Vmax crossing per storm.
    df = (
        df.sort_values("vmax_kt", ascending=False)
          .groupby("storm_id", sort=False)
          .first()
          .reset_index()
    )

    df["year"] = df["datetime"].dt.year
    cols = ["storm_id", "name", "year", "datetime",
            "lat", "lon", "vmax_kt", "pmin_mb", "status"]
    return df[cols].sort_values("datetime").reset_index(drop=True)


# ---------------------------------------------------------------------------
# State assignment
# ---------------------------------------------------------------------------

def _assign_states(df: pd.DataFrame, states_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Assign each crossing point to the nearest CONUS state (by sjoin_nearest).

    Crossing points lie exactly on state boundaries; sjoin_nearest handles
    this gracefully where 'within' predicate would fail.
    Returns a numpy array of state names aligned with df's index.
    """
    crossings_gdf = gpd.GeoDataFrame(
        {"_idx": range(len(df))},
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:5070")
    conus_states = (
        states_gdf[~states_gdf["STATEFP"].isin(_NON_CONUS_FIPS)][["NAME", "geometry"]]
        .reset_index(drop=True)
        .to_crs("EPSG:5070")
    )
    joined = gpd.sjoin_nearest(crossings_gdf, conus_states, how="left")
    # sjoin_nearest may produce duplicate rows if equidistant; keep first.
    joined = joined.drop_duplicates(subset="_idx").sort_values("_idx")
    return joined["NAME"].values


# ---------------------------------------------------------------------------
# Geography hard asserts (Adjustment 2)
# ---------------------------------------------------------------------------

def _assert_geography(df: pd.DataFrame) -> None:
    """
    Hard assert: all crossings must land in known CONUS coastal states,
    and every required state must have >= 1 HU landfall.

    Raises AssertionError with the offending rows if either check fails.
    Converts 'trust the downstream filter' into 'fail loudly on contamination'.
    """
    # 0. No NaN state assignments.
    null_mask = df["state"].isna()
    if null_mask.any():
        raise AssertionError(
            f"{null_mask.sum()} crossing(s) could not be assigned to any state:\n"
            f"{df[null_mask][['storm_id','name','year','lat','lon']].to_string()}"
        )

    # 1. No non-coastal / landlocked state.
    detected    = set(df["state"].unique())
    unexpected  = detected - _COASTAL_STATES
    if unexpected:
        bad = df[~df["state"].isin(_COASTAL_STATES)]
        raise AssertionError(
            f"Crossings detected in non-coastal state(s): {unexpected}\n"
            f"Offending rows:\n"
            f"{bad[['storm_id','name','year','lat','lon','vmax_kt','state']].to_string()}"
        )

    # 2. Required coastal states each have >= 1 HU.
    hu_df   = df[df["status"] == "HU"]
    missing = [s for s in sorted(_REQUIRED_HU_STATES)
               if int((hu_df["state"] == s).sum()) == 0]
    if missing:
        raise AssertionError(
            f"Expected >= 1 HU landfall in each required coastal state, "
            f"but found 0 in: {missing}.\n"
            f"Check CONUS bbox (_CONUS_LON_MIN/MAX, _CONUS_LAT_MIN/MAX) "
            f"or land_union.boundary construction."
        )

    print("[OK] Geography asserts passed — all crossings in coastal states; "
          "all required states have >= 1 HU.")


# ---------------------------------------------------------------------------
# Summary and validation
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    hu_count = int((df["status"] == "HU").sum())
    print(f"\nCONUS landfalls (unique storms) : {len(df):,}")
    for s in _STATUS_ORDER + [x for x in df["status"].unique() if x not in _STATUS_ORDER]:
        n = int((df["status"] == s).sum())
        if n:
            print(f"  {s:<6}: {n:,}")
    print(
        f"Date range                      : "
        f"{df['datetime'].min().strftime('%Y-%m-%d')} — "
        f"{df['datetime'].max().strftime('%Y-%m-%d')}"
    )

    print("\nTop 5 by vmax_kt:")
    top5 = df.nlargest(5, "vmax_kt")[["storm_id", "name", "year", "vmax_kt",
                                       "status", "state"]]
    for _, r in top5.iterrows():
        print(f"  {r['storm_id']}  {r['name']:<18} {int(r['year'])}"
              f"  {r['vmax_kt']:.1f} kt  {r['status']}  {r['state']}")

    # Per-state breakdown
    print("\nPer-state HU landfall counts:")
    print(f"  {'State':<20} {'HU':>4}  {'TS':>4}  {'TD':>4}  {'Total':>6}")
    print("  " + "-" * 44)
    state_summary = (
        df.groupby(["state", "status"])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=["HU", "TS", "TD"], fill_value=0)
    )
    state_summary["Total"] = state_summary.sum(axis=1)
    for state, row in state_summary.sort_values("HU", ascending=False).iterrows():
        print(f"  {state:<20} {row['HU']:>4}  {row['TS']:>4}  {row['TD']:>4}  "
              f"{row['Total']:>6}")

    # Validation spot-checks
    print(f"\nValidation — {len(_VALIDATION_STORMS)} known CONUS landfalls:")
    all_pass = True
    for name, year, expected_states, note in _VALIDATION_STORMS:
        row = df[(df["name"].str.upper() == name.upper()) & (df["year"] == year)]
        if not row.empty:
            r        = row.iloc[0]
            detected = r["state"] if "state" in r.index else "?"
            if r["status"] == "HU" and detected in expected_states:
                print(f"  PASS  {name:<10} {year}  "
                      f"({r['vmax_kt']:.1f} kt, {detected}, "
                      f"lat={r['lat']:.2f} lon={r['lon']:.2f})  [{note}]")
            else:
                all_pass = False
                print(f"  FAIL  {name:<10} {year}  "
                      f"status={r['status']} state={detected} "
                      f"(expected HU in {expected_states})  [{note}]")
        else:
            all_pass = False
            print(f"  FAIL  {name:<10} {year}  NOT DETECTED  [{note}]")

    print("  All validation storms PASS." if all_pass
          else "  WARNING: one or more validation storms FAILED.")


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def _save_map(df: pd.DataFrame, states_gdf: gpd.GeoDataFrame, path: str) -> None:
    """
    Scatter of CONUS landfall crossing points along the full east + Gulf coast.
    Color-coded by status, size proportional to Vmax.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    lon_min, lon_max = -98.0, -64.0
    lat_min, lat_max =  23.0,  48.0
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    states_gdf.clip_by_rect(lon_min, lat_min, lon_max, lat_max).plot(
        ax=ax, facecolor="#f0f0f0", edgecolor="#888888", linewidth=0.5
    )

    for status in _STATUS_ORDER:
        sub = df.loc[df["status"] == status]
        if sub.empty:
            continue
        ax.scatter(
            sub["lon"], sub["lat"],
            c=_STATUS_COLOR[status],
            s=sub["vmax_kt"].fillna(40.0) * 0.5,
            alpha=0.85, zorder=5, linewidths=0.3, edgecolors="k",
        )

    other = df.loc[~df["status"].isin(_STATUS_ORDER)]
    if not other.empty:
        ax.scatter(
            other["lon"], other["lat"], c="#7f7f7f",
            s=other["vmax_kt"].fillna(40.0) * 0.5,
            alpha=0.85, zorder=5, linewidths=0.3, edgecolors="k",
        )

    legend_handles = [
        mpatches.Patch(color=_STATUS_COLOR["HU"], label="Hurricane (HU)"),
        mpatches.Patch(color=_STATUS_COLOR["TS"], label="Tropical Storm (TS)"),
        mpatches.Patch(color=_STATUS_COLOR["TD"], label="Tropical Depression (TD)"),
    ]
    if not other.empty:
        legend_handles.append(mpatches.Patch(color="#7f7f7f", label="Other"))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.9)

    hu_n = int((df["status"] == "HU").sum())
    ax.set_title(
        f"Historical CONUS Atlantic/Gulf Landfalls  |  Geometric Crossing Detection\n"
        f"HURDAT2 1851–2024  |  {len(df):,} storms total  |  {hu_n:,} HU  "
        f"|  One crossing per storm (highest Vmax)",
        fontsize=11, pad=10,
    )
    ax.set_xlabel("Longitude (°E)", fontsize=9)
    ax.set_ylabel("Latitude (°N)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.annotate(
        "Point size ∝ Vmax (kt)",
        xy=(0.02, 0.97), xycoords="axes fraction",
        fontsize=7, va="top", color="#555555",
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nMap saved -> {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading {_IN} ...")
    fixes = pd.read_parquet(_IN)
    print(f"  {len(fixes):,} fixes  |  {fixes['storm_id'].nunique():,} unique storms")

    with tempfile.TemporaryDirectory() as tmp_dir:
        conus_boundary, land_union, states_gdf = _load_geometries(tmp_dir)

        print("Detecting geometric CONUS landfalls ...")
        df = detect_all_conus_landfalls(fixes, land_union)

        print(f"\n{len(df):,} unique-storm crossings detected.")
        print("Assigning states via sjoin_nearest ...")
        df["state"] = _assign_states(df, states_gdf)

        _assert_geography(df)

        # Write CSV without the ephemeral state column.
        out_cols = ["storm_id", "name", "year", "datetime",
                    "lat", "lon", "vmax_kt", "pmin_mb", "status"]
        os.makedirs(os.path.dirname(_OUT), exist_ok=True)
        df[out_cols].to_csv(_OUT, index=False)
        print(f"\nSaved -> {_OUT}  ({len(df):,} rows × {len(out_cols)} cols)")

        _print_summary(df)
        _save_map(df, states_gdf, _MAP)

    print()
    print("=" * 62)
    print("conus_landfalls.csv written.  fl_landfalls.csv is UNCHANGED.")
    print("This file feeds calibration/wind_pressure.py (WPR fit).")
    print("=" * 62)
