"""
Parse the HURDAT2 Atlantic hurricane database into a tidy fix-level DataFrame.

Input : config/calibration.yaml -> hurdat2.raw_path
Output: config/calibration.yaml -> hurdat2.processed_path  (Parquet)

Output columns
--------------
storm_id   str        Basin + number + year  (e.g. 'AL011851')
name       str        Storm name  (e.g. 'KATRINA')
datetime   Timestamp  UTC, combined from YYYYMMDD + HHMM columns
record_id  str        '' = routine fix, 'L' = landfall, others per HURDAT2 spec
status     str        HU / TS / TD / SS / EX / etc.
lat        float64    Decimal degrees  (positive = N, negative = S)
lon        float64    Decimal degrees  (positive = E, negative = W)
vmax_kt    float64    Maximum sustained wind, knots  (NaN where source is -999)
pmin_mb    float64    Minimum central pressure, mb   (NaN where source is -999)

Wind speeds are stored in knots — the native HURDAT2 unit.
Use model/units.py for any downstream conversion.
"""

import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from model_config import load_calibration_cfg

_ccfg = load_calibration_cfg()
_RAW  = os.path.join(_ROOT, _ccfg.hurdat2.raw_path)
_OUT  = os.path.join(_ROOT, _ccfg.hurdat2.processed_path)
_MISS = _ccfg.hurdat2.missing_sentinel


def _parse_coord(s):
    """
    Parse a HURDAT2 coordinate token to a signed decimal degree float.

    '28.0N' ->  28.0  (north = positive)
    '94.8W' -> -94.8  (west  = negative)
    '10.5S' -> -10.5  (south = negative)
    '80.0E' ->  80.0  (east  = positive)
    """
    s    = s.strip()
    hemi = s[-1]
    val  = float(s[:-1])
    return -val if hemi in ('S', 'W') else val


def parse(path=_RAW):
    """
    Parse a HURDAT2 text file into a tidy fix-level DataFrame.

    Parameters
    ----------
    path : str  -- path to the raw HURDAT2 file (default: from config)

    Returns
    -------
    pd.DataFrame  -- one row per fix; see module docstring for column schema
    """
    rows         = []
    current_id   = None
    current_name = None

    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip('\n')
            if not line.strip():
                continue

            # Header lines begin with a basin letter (AL, EP, CP…).
            # Data lines begin with a digit (date: YYYYMMDD).
            if line.lstrip()[0].isalpha():
                parts        = [p.strip() for p in line.split(',')]
                current_id   = parts[0]
                current_name = parts[1]
            else:
                fields = [f.strip() for f in line.split(',')]

                date_s    = fields[0]          # 'YYYYMMDD'
                time_s    = fields[1].zfill(4) # 'HHMM' — zfill guards stripped leading zeros
                record_id = fields[2]          # '' for routine fix, 'L' for landfall, etc.
                status    = fields[3]          # 'HU', 'TS', 'TD', …

                lat = _parse_coord(fields[4])
                lon = _parse_coord(fields[5])

                vmax_raw = int(fields[6])
                pmin_raw = int(fields[7])
                vmax_kt  = np.nan if vmax_raw == _MISS else float(vmax_raw)
                pmin_mb  = np.nan if pmin_raw == _MISS else float(pmin_raw)

                dt = pd.to_datetime(date_s + time_s, format="%Y%m%d%H%M", utc=True)

                rows.append({
                    'storm_id':  current_id,
                    'name':      current_name,
                    'datetime':  dt,
                    'record_id': record_id,
                    'status':    status,
                    'lat':       lat,
                    'lon':       lon,
                    'vmax_kt':   vmax_kt,
                    'pmin_mb':   pmin_mb,
                })

    return pd.DataFrame(rows)


def _print_summary(df):
    n_storms   = df['storm_id'].nunique()
    n_fixes    = len(df)
    dt_min     = df['datetime'].min()
    dt_max     = df['datetime'].max()
    n_landfall = int((df['record_id'] == 'L').sum())

    print(f"Total storms  : {n_storms:,}")
    print(f"Total fixes   : {n_fixes:,}")
    print(f"Date range    : {dt_min.strftime('%Y-%m-%d %H:%M')} UTC"
          f" – {dt_max.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"Landfall fixes: {n_landfall:,}")


if __name__ == "__main__":
    print(f"Parsing {_RAW} ...")
    df = parse()

    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    df.to_parquet(_OUT, index=False)
    print(f"Saved  -> {_OUT}  ({len(df):,} rows x {len(df.columns)} columns)\n")

    _print_summary(df)
