"""
Identify historical Florida landfalls from HURDAT2 using geometric
coastline-crossing detection.

Previous implementation relied on NHC record_id == 'L' landfall flags, which
are sparsely populated before ~1990 (e.g. Eloise 1975, David 1979 have zero
L-flags despite known Florida landfalls).  This version detects landfalls
geometrically:

1. Load all fixes (record_id is NOT used).
2. For each storm, iterate consecutive fix pairs (a -> b):
   - Fix a must be over water: a NOT within the CONUS land polygon.
   - The segment LineString(a, b) must intersect the Florida state boundary.
   These two conditions together identify an ocean->FL crossing and exclude
   overland entries from Georgia or Alabama.
3. The crossing point is the entry intersection (closest to a).
   The interpolation fraction f = dist(a->crossing) / dist(a->b) is computed
   in EPSG:3086 (Florida GDL Albers, metres) to avoid the degree-longitude !=
   degree-latitude bias at ~27 N latitude.
4. Vmax, pmin, and datetime are linearly interpolated at f.
5. Per storm, keep the crossing with the highest interpolated Vmax.

Output columns
--------------
storm_id  str        HURDAT2 id (e.g. 'AL041992')
name      str        Storm name
year      int        Year of FL landfall
datetime  Timestamp  UTC at geometric crossing
lat       float64    Crossing latitude (on FL boundary)
lon       float64    Crossing longitude (on FL boundary)
vmax_kt   float64    Interpolated Vmax at crossing, knots
pmin_mb   float64    Interpolated min pressure, mb (NaN if either endpoint missing)
status    str        HU (>=64 kt) / TS (34-63 kt) / TD (<34 kt)
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
_OUT  = os.path.join(_ROOT, _ccfg.fl_landfalls.processed_path)
_MAP  = os.path.join(_ROOT, _ccfg.fl_landfalls.map_path)

_TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"

# AK, HI, and US territories — excluded from the CONUS land mask.
_NON_CONUS_FIPS = {"02", "15", "60", "66", "69", "72", "78"}

# EPSG:3086: NAD83 / Florida GDL Albers (metres).
# Used only for the interpolation fraction; all detection stays in EPSG:4326.
_PROJ = Transformer.from_crs("EPSG:4326", "EPSG:3086", always_xy=True)

# Generous bounding box around Florida: fast rejection before the more
# expensive fl_boundary intersection test.
_FL_LON_MIN, _FL_LON_MAX = -88.0, -79.0
_FL_LAT_MIN, _FL_LAT_MAX =  23.0,  32.0

_STATUS_COLOR = {"HU": "#d62728", "TS": "#ff7f0e", "TD": "#bcbd22"}
_STATUS_ORDER = ["HU", "TS", "TD"]

# Known FL hurricanes used to validate the new detection logic.
_VALIDATION_STORMS = [
    ("ELOISE",  1975),
    ("DAVID",   1979),
    ("KATE",    1985),
    ("ANDREW",  1992),
    ("CHARLEY", 2004),
    ("WILMA",   2005),
]


# ---------------------------------------------------------------------------
# Geometry loading
# ---------------------------------------------------------------------------

def _load_geometries(tmp_dir: str):
    """
    Download TIGER/Line 2023, return (fl_poly, land_union, states_gdf).

    fl_poly    -- Florida state polygon (shapely geometry, WGS84)
    land_union -- unary_union of all CONUS state polygons (over-water mask)
    states_gdf -- full US states GeoDataFrame (map background)
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

    fl_row  = states_gdf.loc[states_gdf["NAME"] == "Florida"]
    fl_poly = fl_row.geometry.iloc[0]

    conus      = states_gdf.loc[~states_gdf["STATEFP"].isin(_NON_CONUS_FIPS)]
    land_union = unary_union(conus.geometry)
    print(f"  CONUS land polygon built ({len(conus)} states + DC).")
    return fl_poly, land_union, states_gdf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_status(vmax_kt: float) -> str:
    """Saffir-Simpson threshold classification from sustained wind (knots)."""
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
    Return f = dist(a->c) / dist(a->b) using EPSG:3086 metric distances.

    At FL's latitude (~27 N), 1 deg lon ~ 99 km != 1 deg lat ~ 111 km.
    Using raw degree distances would bias f and therefore interpolated Vmax.
    EPSG:3086 (Florida GDL Albers, metres) removes this distortion.
    Clamped to [0, 1] against floating-point overshoot at the boundary.
    """
    xa, ya = _PROJ.transform(lon_a, lat_a)
    xc, yc = _PROJ.transform(lon_c, lat_c)
    xb, yb = _PROJ.transform(lon_b, lat_b)
    dist_ab = math.hypot(xb - xa, yb - ya)
    if dist_ab < 1.0:       # degenerate: nearly identical endpoints
        return 0.0
    return max(0.0, min(1.0, math.hypot(xc - xa, yc - ya) / dist_ab))


def _first_intersection(line: LineString, boundary) -> Optional[Point]:
    """
    Return the entry crossing point: boundary intersection closest to
    line.coords[0] along the segment direction.

    Handles Point, MultiPoint, LineString (collinear), GeometryCollection.
    """
    isect = line.intersection(boundary)
    if isect.is_empty:
        return None
    if isect.geom_type == "Point":
        return isect
    if isect.geom_type == "LineString":
        # Segment collinear with boundary — take the start (entry end).
        return Point(isect.coords[0])
    # MultiPoint or GeometryCollection: gather all candidate points.
    pts = []
    for g in isect.geoms:
        if g.geom_type == "Point":
            pts.append(g)
        elif g.geom_type == "LineString":
            pts.append(Point(g.coords[0]))
        elif g.geom_type == "MultiPoint":
            pts.extend(g.geoms)
    return min(pts, key=lambda p: line.project(p)) if pts else None


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def _detect_storm_crossings(
    storm_fixes: pd.DataFrame,
    prep_land,
    fl_boundary,
) -> list:
    """
    Detect all ocean->Florida boundary crossings for a single storm.

    Returns a list of crossing dicts (one per detected crossing; may be empty).
    """
    rows = list(storm_fixes.itertuples(index=False))
    if len(rows) < 2:
        return []

    crossings = []
    for a, b in zip(rows[:-1], rows[1:]):
        # Bounding-box pre-filter: fast rejection for segments far from FL.
        if not (
            (_FL_LON_MIN <= a.lon <= _FL_LON_MAX and _FL_LAT_MIN <= a.lat <= _FL_LAT_MAX)
            or
            (_FL_LON_MIN <= b.lon <= _FL_LON_MAX and _FL_LAT_MIN <= b.lat <= _FL_LAT_MAX)
        ):
            continue

        # Fix a must be over water — excludes overland entries from GA / AL.
        if prep_land.contains(Point(a.lon, a.lat)):
            continue

        seg = LineString([(a.lon, a.lat), (b.lon, b.lat)])
        if not seg.intersects(fl_boundary):
            continue

        crossing_pt = _first_intersection(seg, fl_boundary)
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


def detect_all_fl_landfalls(fixes: pd.DataFrame, fl_poly, land_union) -> tuple:
    """
    Detect FL landfalls for all storms in the HURDAT2 fix table.

    Returns
    -------
    (df, genesis_skipped)
      df              -- one row per storm, highest-Vmax crossing retained
      genesis_skipped -- storms with FL fixes but no detectable ocean->FL crossing
    """
    prep_land   = prep(land_union)
    prep_fl     = prep(fl_poly)
    fl_boundary = fl_poly.boundary

    all_crossings   = []
    genesis_skipped = 0
    storm_groups    = list(fixes.groupby("storm_id", sort=False))
    n               = len(storm_groups)

    for i, (storm_id, group) in enumerate(storm_groups, 1):
        if i % 500 == 0:
            print(f"  {i}/{n} storms processed ...")
        group     = group.sort_values("datetime")
        crossings = _detect_storm_crossings(group, prep_land, fl_boundary)

        if crossings:
            all_crossings.extend(crossings)
        else:
            # Count storms that touched FL but had no water->FL crossing
            # (genesis over FL, or storm that stayed on land throughout).
            if any(prep_fl.contains(Point(r.lon, r.lat)) for _, r in group.iterrows()):
                genesis_skipped += 1

    if not all_crossings:
        empty = pd.DataFrame(
            columns=["storm_id", "name", "year", "datetime",
                     "lat", "lon", "vmax_kt", "pmin_mb", "status"]
        )
        return empty, genesis_skipped

    df = pd.DataFrame(all_crossings)

    # Keep the highest-Vmax crossing per storm (handles re-entry cases).
    df = (
        df.sort_values("vmax_kt", ascending=False)
          .groupby("storm_id", sort=False)
          .first()
          .reset_index()
    )

    df["year"] = df["datetime"].dt.year
    cols = ["storm_id", "name", "year", "datetime",
            "lat", "lon", "vmax_kt", "pmin_mb", "status"]
    return df[cols].sort_values("datetime").reset_index(drop=True), genesis_skipped


# ---------------------------------------------------------------------------
# Summary and validation
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame, genesis_skipped: int) -> None:
    print(f"\nFL landfalls (unique storms) : {len(df):,}")
    for s in _STATUS_ORDER + [x for x in df["status"].unique() if x not in _STATUS_ORDER]:
        n = int((df["status"] == s).sum())
        if n:
            print(f"  {s:<6}: {n:,}")
    print(
        f"Date range                  : "
        f"{df['datetime'].min().strftime('%Y-%m-%d')} - "
        f"{df['datetime'].max().strftime('%Y-%m-%d')}"
    )
    print(f"Genesis-in-FL skipped       : {genesis_skipped}")

    print("\nTop 5 by vmax_kt:")
    top5 = df.nlargest(5, "vmax_kt")[["storm_id", "name", "year", "vmax_kt", "status"]]
    for _, r in top5.iterrows():
        print(
            f"  {r['storm_id']}  {r['name']:<18} {int(r['year'])}"
            f"  {r['vmax_kt']:.1f} kt  {r['status']}"
        )

    print("\nValidation -- known FL hurricanes (must all appear as HU):")
    all_pass = True
    for name, year in _VALIDATION_STORMS:
        row = df[(df["name"].str.upper() == name) & (df["year"] == year)]
        if not row.empty and row.iloc[0]["status"] == "HU":
            r = row.iloc[0]
            print(
                f"  PASS  {name:<10} {year}  "
                f"({r['vmax_kt']:.1f} kt, "
                f"lat={r['lat']:.2f} lon={r['lon']:.2f})"
            )
        else:
            all_pass = False
            if row.empty:
                print(f"  FAIL  {name:<10} {year}  NOT DETECTED")
            else:
                r = row.iloc[0]
                print(
                    f"  FAIL  {name:<10} {year}  detected as "
                    f"{r['status']} ({r['vmax_kt']:.1f} kt)"
                )
    print(
        "  All 6 validation storms PASS."
        if all_pass else
        "  WARNING: one or more validation storms FAILED -- check detection logic."
    )


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def _save_map(df: pd.DataFrame, states_gdf: gpd.GeoDataFrame, path: str) -> None:
    """
    Scatter of FL landfall crossing points over SE US state boundaries.
    All points lie on the Florida coastline (geometric crossing -> exact boundary).
    Color-coded by status, size proportional to Vmax.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    lon_min, lon_max = -90.0, -74.5
    lat_min, lat_max =  23.0,  35.5
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    states_gdf.clip_by_rect(lon_min, lat_min, lon_max, lat_max).plot(
        ax=ax, facecolor="#f0f0f0", edgecolor="#888888", linewidth=0.6
    )
    states_gdf.loc[states_gdf["NAME"] == "Florida"].plot(
        ax=ax, facecolor="#cce5ff", edgecolor="#555555", linewidth=0.8
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
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9, framealpha=0.9)

    ax.set_title(
        "Historical Florida Landfalls  |  Geometric Crossing Detection\n"
        "HURDAT2 1851-2024  |  Ocean->FL boundary crossing (replaces NHC 'L' flag)",
        fontsize=11, pad=10,
    )
    ax.set_xlabel("Longitude (deg E)", fontsize=9)
    ax.set_ylabel("Latitude (deg N)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.annotate(
        "Point size proportional to Vmax (kt)",
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
        fl_poly, land_union, states_gdf = _load_geometries(tmp_dir)
        print("Detecting geometric Florida landfalls ...")
        df, genesis_skipped = detect_all_fl_landfalls(fixes, fl_poly, land_union)

        os.makedirs(os.path.dirname(_OUT), exist_ok=True)
        df.to_csv(_OUT, index=False)
        print(f"\nSaved  -> {_OUT}  ({len(df):,} rows x {len(df.columns)} columns)")

        _print_summary(df, genesis_skipped)
        _save_map(df, states_gdf, _MAP)

    print()
    print("=" * 62)
    print("ACTION REQUIRED -- landfall catalogue has changed")
    print("=" * 62)
    print("Geometric detection replaces the sparse NHC 'L' flag.")
    print("The HU count has increased; lambda will rise above 0.44.")
    print()
    print("Re-run in this order:")
    print("  python calibration/frequency.py")
    print("  python calibration/frequency_glm.py")
    print("=" * 62)
