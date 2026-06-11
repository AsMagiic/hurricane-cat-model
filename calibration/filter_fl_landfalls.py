"""
Identify historical Florida landfalls from HURDAT2 fix-level data.

Methodology
-----------
1. Load data/processed/hurdat2_fixes.parquet (produced by parse_hurdat2.py).
2. Keep only NHC-assigned landfall fixes (record_id == 'L').
3. Download the US Census Bureau TIGER/Line state shapefile at runtime,
   extract Florida's polygon, and run a point-in-polygon test.
4. Deduplicate to one row per storm (first FL landfall chronologically).
5. Save to data/processed/fl_landfalls.csv and outputs/fl_landfall_map.png.

Output columns
--------------
storm_id  str        HURDAT2 basin+number+year  (e.g. 'AL021992')
name      str        Storm name
year      int        Year of FL landfall
datetime  Timestamp  UTC
lat       float64    Decimal degrees N
lon       float64    Decimal degrees E (negative = W)
vmax_kt   float64    Maximum sustained wind, knots
pmin_mb   float64    Minimum central pressure, mb
status    str        HU / TS / TD / SS / EX / etc.
"""

import os
import sys
import tempfile
import urllib.request
import zipfile

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from model_config import load_calibration_cfg

_ccfg  = load_calibration_cfg()
_IN    = os.path.join(_ROOT, _ccfg.hurdat2.processed_path)
_OUT   = os.path.join(_ROOT, _ccfg.fl_landfalls.processed_path)
_MAP   = os.path.join(_ROOT, _ccfg.fl_landfalls.map_path)

_TIGER_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"
)

_STATUS_COLOR = {
    "HU": "#d62728",   # red
    "TS": "#ff7f0e",   # orange
    "TD": "#bcbd22",   # yellow-green
}
_STATUS_ORDER = ["HU", "TS", "TD"]


# ---------------------------------------------------------------------------
# Florida polygon
# ---------------------------------------------------------------------------

def _load_florida_geom(tmp_dir: str):
    """
    Download and return the Florida state geometry from TIGER/Line 2023.

    Parameters
    ----------
    tmp_dir : str  -- temporary directory for the downloaded zip and shapefile

    Returns
    -------
    shapely geometry  -- Florida state polygon (or multipolygon)
    geopandas.GeoDataFrame  -- full US states GDF (reused for map background)
    """
    zip_path = os.path.join(tmp_dir, "tl_2023_us_state.zip")
    print(f"Downloading {_TIGER_URL} ...")
    urllib.request.urlretrieve(_TIGER_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    shp = next(
        os.path.join(tmp_dir, f)
        for f in os.listdir(tmp_dir)
        if f.endswith(".shp")
    )
    states_gdf = gpd.read_file(shp)
    fl_row = states_gdf.loc[states_gdf["NAME"] == "Florida"]
    fl_geom = fl_row.geometry.iloc[0]
    return fl_geom, states_gdf


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def filter_fl_landfalls(fixes: pd.DataFrame, fl_geom) -> pd.DataFrame:
    """
    Return deduplicated Florida landfalls from a fix-level HURDAT2 DataFrame.

    Parameters
    ----------
    fixes   : pd.DataFrame  -- full fix table from parse_hurdat2.py
    fl_geom : shapely geometry  -- Florida state polygon

    Returns
    -------
    pd.DataFrame  -- one row per storm, sorted by datetime
    """
    # NHC-assigned landfall fixes only.
    lf = fixes.loc[fixes["record_id"] == "L"].copy()

    # Point-in-polygon: keep fixes whose (lon, lat) falls inside Florida.
    in_fl = lf.apply(lambda r: fl_geom.contains(Point(r["lon"], r["lat"])), axis=1)
    fl_lf = lf.loc[in_fl].copy()

    # First FL landfall per storm (chronological).
    fl_lf = (
        fl_lf
        .sort_values("datetime")
        .groupby("storm_id", sort=False)
        .first()
        .reset_index()
    )

    fl_lf["year"] = fl_lf["datetime"].dt.year

    cols = ["storm_id", "name", "year", "datetime", "lat", "lon",
            "vmax_kt", "pmin_mb", "status"]
    return fl_lf[cols].sort_values("datetime").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    print(f"\nFL landfalls (unique storms): {len(df):,}")
    for s in _STATUS_ORDER + [x for x in df["status"].unique() if x not in _STATUS_ORDER]:
        n = int((df["status"] == s).sum())
        if n:
            print(f"  {s:<6}: {n:,}")
    dt_min = df["datetime"].min()
    dt_max = df["datetime"].max()
    print(f"Date range : {dt_min.strftime('%Y-%m-%d')} – {dt_max.strftime('%Y-%m-%d')}")
    print("\nTop 5 by vmax_kt:")
    top5 = df.nlargest(5, "vmax_kt")[["storm_id", "name", "year", "vmax_kt", "status"]]
    for _, r in top5.iterrows():
        vmax = "N/A" if np.isnan(r["vmax_kt"]) else f"{int(r['vmax_kt'])} kt"
        print(f"  {r['storm_id']}  {r['name']:<18} {int(r['year'])}  {vmax}  {r['status']}")


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def _save_map(df: pd.DataFrame, states_gdf: gpd.GeoDataFrame, path: str) -> None:
    """
    Scatter of FL landfall points over SE US state boundaries.

    Parameters
    ----------
    df         : pd.DataFrame          -- fl_landfalls output
    states_gdf : gpd.GeoDataFrame      -- full US states (from TIGER)
    path       : str                   -- output PNG path
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # SE US extent — enough context, not the whole country.
    lon_min, lon_max = -90.0, -74.5
    lat_min, lat_max = 23.0,  35.5
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # State boundaries — clip to extent for speed.
    states_gdf.clip_by_rect(lon_min, lat_min, lon_max, lat_max).plot(
        ax=ax, facecolor="#f0f0f0", edgecolor="#888888", linewidth=0.6
    )
    # Florida highlighted.
    fl_row = states_gdf.loc[states_gdf["NAME"] == "Florida"]
    fl_row.plot(ax=ax, facecolor="#cce5ff", edgecolor="#555555", linewidth=0.8)

    # Landfall scatter — size scaled by intensity.
    for status in _STATUS_ORDER:
        sub = df.loc[df["status"] == status]
        if sub.empty:
            continue
        sizes = np.where(sub["vmax_kt"].isna(), 20.0, sub["vmax_kt"].fillna(0) * 0.5)
        ax.scatter(
            sub["lon"], sub["lat"],
            c=_STATUS_COLOR[status], s=sizes,
            alpha=0.85, zorder=5, linewidths=0.3, edgecolors="k",
        )
    # Any non-HU/TS/TD status.
    other = df.loc[~df["status"].isin(_STATUS_ORDER)]
    if not other.empty:
        sizes = np.where(other["vmax_kt"].isna(), 20.0, other["vmax_kt"].fillna(0) * 0.5)
        ax.scatter(other["lon"], other["lat"], c="#7f7f7f", s=sizes,
                   alpha=0.85, zorder=5, linewidths=0.3, edgecolors="k")

    # Legend.
    legend_handles = [
        mpatches.Patch(color=_STATUS_COLOR["HU"], label="Hurricane (HU)"),
        mpatches.Patch(color=_STATUS_COLOR["TS"], label="Tropical Storm (TS)"),
        mpatches.Patch(color=_STATUS_COLOR["TD"], label="Tropical Depression (TD)"),
    ]
    if not other.empty:
        legend_handles.append(mpatches.Patch(color="#7f7f7f", label="Other"))
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9, framealpha=0.9)

    ax.set_title(
        "Historical Florida Hurricane Landfalls\nHURDAT2 1851–2025  |  NHC record_id == 'L'",
        fontsize=12, pad=10,
    )
    ax.set_xlabel("Longitude (°E)", fontsize=9)
    ax.set_ylabel("Latitude (°N)", fontsize=9)
    ax.tick_params(labelsize=8)

    # Size legend note.
    ax.annotate(
        "Point size ∝ vmax (kt)", xy=(0.02, 0.97),
        xycoords="axes fraction", fontsize=7, va="top", color="#555555",
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        fl_geom, states_gdf = _load_florida_geom(tmp_dir)
        print("Filtering Florida landfall fixes ...")
        df = filter_fl_landfalls(fixes, fl_geom)

        os.makedirs(os.path.dirname(_OUT), exist_ok=True)
        df.to_csv(_OUT, index=False)
        print(f"Saved  -> {_OUT}  ({len(df):,} rows x {len(df.columns)} columns)")

        _print_summary(df)
        _save_map(df, states_gdf, _MAP)
