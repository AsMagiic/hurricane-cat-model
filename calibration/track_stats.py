"""
Step 1.5a Part 2 — FL landfall track statistics (heading + translation speed).

For each of the 112 HU FL landfalls (fl_landfalls.csv, status==HU), find the
HURDAT2 fix pair that brackets the landfall datetime and compute:
  - heading_deg : meteorological bearing of storm motion (deg clockwise from N)
  - speed_kmh   : translation speed from haversine distance / time delta
  - speed_kt    : same in knots (speed_kmh / 1.852)

Methodology
-----------
  fix_pre  = last HURDAT2 fix at or before the landfall datetime
  fix_post = first HURDAT2 fix strictly after the landfall datetime

This bracketing pair represents the storm's velocity at the moment it crossed
the FL boundary — physically the quantity hazard.py will need.  The entire storm
track (not a windowed subset) is searched so that storms with widely-spaced fixes
near landfall are not silently excluded.

Heading convention (meteorological)
-------------------------------------
  bearing = atan2(x, y)  [NOT atan2(y, x)]
  x = sin(dlon) * cos(lat2)
  y = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dlon)
  result in [0, 360)  —  N=0, E=90, S=180, W=270

Outputs
-------
  data/processed/fl_landfall_tracks.csv   per-storm table (committed)
  config/model_v3.yaml                    hazard.track_stats block
  outputs/track_stats.png                 wind rose + speed histogram
"""

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_LF_PATH  = os.path.join(_ROOT, "data", "processed", "fl_landfalls.csv")
_FX_PATH  = os.path.join(_ROOT, "data", "processed", "hurdat2_fixes.parquet")
_OUT_CSV  = os.path.join(_ROOT, "data", "processed", "fl_landfall_tracks.csv")
_OUT_PLOT = os.path.join(_ROOT, "outputs", "track_stats.png")
_CFG_PATH = os.path.join(_ROOT, "config", "model_v3.yaml")

_KMH_TO_KT = 1.0 / 1.852   # exact: 1 kt = 1.852 km/h
_DT_FLAG_H  = 6.0            # flag bracketing pairs wider than this (track gap)


# ---------------------------------------------------------------------------
# A: Geometry helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (scalar inputs)."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2.0 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def bearing_deg(lat1, lon1, lat2, lon2):
    """
    Forward azimuth from (lat1,lon1) to (lat2,lon2).
    Meteorological convention: clockwise from true North; N=0, E=90, S=180, W=270.

    Formula: atan2(x, y) — NOT atan2(y, x); the swap is the classic sign error.
      x = sin(dlon) * cos(lat2)
      y = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dlon)
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return math.degrees(math.atan2(x, y)) % 360.0


# ---------------------------------------------------------------------------
# B: Data loading
# ---------------------------------------------------------------------------

def load_data():
    lf = pd.read_csv(_LF_PATH)
    lf["datetime"] = pd.to_datetime(lf["datetime"], utc=True)
    hu = lf[lf["status"] == "HU"].reset_index(drop=True)
    print(f"HU landfalls loaded : {len(hu)}")

    fx = pd.read_parquet(_FX_PATH)
    fx["datetime"] = pd.to_datetime(fx["datetime"], utc=True)
    fx_by_storm = {
        sid: grp.sort_values("datetime").reset_index(drop=True)
        for sid, grp in fx.groupby("storm_id")
    }
    print(f"HURDAT2 fixes loaded: {len(fx):,} across {len(fx_by_storm):,} storms")
    return hu, fx_by_storm


# ---------------------------------------------------------------------------
# C: Per-storm bracketing pair
# ---------------------------------------------------------------------------

def compute_track_stats(hu, fx_by_storm):
    """
    For each HU landfall find the fix pair (fix_pre, fix_post) that brackets
    the landfall datetime.  Returns (records, excluded) where:
      records  -- list of dicts with storm_id, year, heading_deg, speed_kmh,
                  speed_kt, dt_h (dt_h kept for diagnostics, dropped in CSV)
      excluded -- list of (storm_id, reason) for storms with no valid pair
    """
    records  = []
    excluded = []

    for _, row in hu.iterrows():
        sid    = row["storm_id"]
        lf_dt  = row["datetime"]

        storm_fx = fx_by_storm.get(sid)
        if storm_fx is None or len(storm_fx) < 2:
            excluded.append((sid, "no fixes for storm"))
            continue

        pre_cands  = storm_fx[storm_fx["datetime"] <= lf_dt]
        post_cands = storm_fx[storm_fx["datetime"] >  lf_dt]

        if pre_cands.empty or post_cands.empty:
            excluded.append((sid, "no bracketing pair (landfall outside fix range)"))
            continue

        fp = pre_cands.iloc[-1]
        fn = post_cands.iloc[0]

        dt_h = (fn["datetime"] - fp["datetime"]).total_seconds() / 3600.0
        if dt_h < 1e-6:
            excluded.append((sid, "degenerate dt=0 between bracketing fixes"))
            continue

        dist_km   = haversine_km(fp["lat"], fp["lon"], fn["lat"], fn["lon"])
        speed_kmh = dist_km / dt_h
        hdg       = bearing_deg(fp["lat"], fp["lon"], fn["lat"], fn["lon"])

        records.append({
            "storm_id":    sid,
            "year":        int(row["year"]),
            "heading_deg": round(hdg, 2),
            "speed_kmh":   round(speed_kmh, 3),
            "speed_kt":    round(speed_kmh * _KMH_TO_KT, 3),
            "dt_h":        round(dt_h, 3),
        })

    return records, excluded


# ---------------------------------------------------------------------------
# D: Circular statistics
# ---------------------------------------------------------------------------

def circular_stats(headings_deg):
    """
    Compute mean and spread via the mean resultant vector.

    Mean direction  : mu  = atan2(mean_sin, mean_cos) mod 360
    Resultant length: R   = sqrt(mean_cos^2 + mean_sin^2)  in [0,1]
    Circular std    : sigma_c = sqrt(-2*ln(R)) radians, converted to degrees
                      (von Mises approximation; exact for the Rayleigh distribution)
    """
    theta = np.radians(np.asarray(headings_deg, dtype=float))
    C = float(np.mean(np.cos(theta)))
    S = float(np.mean(np.sin(theta)))
    R = math.sqrt(C ** 2 + S ** 2)
    mu_deg      = math.degrees(math.atan2(S, C)) % 360.0
    sigma_c_deg = math.degrees(math.sqrt(-2.0 * math.log(max(R, 1e-9))))
    arith_deg   = float(np.mean(np.asarray(headings_deg, dtype=float)))
    return {
        "mean_deg":             round(mu_deg, 2),
        "arithmetic_mean_deg":  round(arith_deg, 2),
        "circ_std_deg":         round(sigma_c_deg, 2),
        "resultant_R":          round(R, 4),
    }


# ---------------------------------------------------------------------------
# E: Translation speed distribution
# ---------------------------------------------------------------------------

def fit_speed_dist(speeds):
    """
    Fit lognormal and gamma by MLE (floc=0; speeds are strictly positive).
    Select winner by AIC = 2k - 2*loglik where k=2 (loc fixed).
    """
    p_ln  = scipy.stats.lognorm.fit(speeds, floc=0)
    ll_ln = float(np.sum(scipy.stats.lognorm.logpdf(speeds, *p_ln)))
    aic_ln = 4.0 - 2.0 * ll_ln

    p_gam  = scipy.stats.gamma.fit(speeds, floc=0)
    ll_gam = float(np.sum(scipy.stats.gamma.logpdf(speeds, *p_gam)))
    aic_gam = 4.0 - 2.0 * ll_gam

    print(f"  Lognormal : loglik={ll_ln:.2f}  AIC={aic_ln:.2f}")
    print(f"  Gamma     : loglik={ll_gam:.2f}  AIC={aic_gam:.2f}")

    if aic_ln <= aic_gam:
        winner, params = "lognormal", list(p_ln)
    else:
        winner, params = "gamma",     list(p_gam)

    all_fits = {
        "lognormal": (p_ln,  aic_ln),
        "gamma":     (p_gam, aic_gam),
    }
    print(f"  Winner    : {winner}  (dAIC={abs(aic_ln - aic_gam):.2f})")
    return winner, params, all_fits


# ---------------------------------------------------------------------------
# F: Config write (targeted insertion, no yaml round-trip)
# ---------------------------------------------------------------------------

def write_config(circ, winner, params, speeds, n_used, n_excluded):
    with open(_CFG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if any("track_stats:" in line for line in lines):
        print("Config: hazard.track_stats already present — skipping write.")
        return

    params_str = "[" + ", ".join(f"{p:.6f}" for p in params) + "]"
    mean_kmh   = round(float(np.mean(speeds)), 3)
    std_kmh    = round(float(np.std(speeds, ddof=1)), 3)

    block = (
        "  track_stats:\n"
        "    heading_mean_deg:\n"
        f"      value: {circ['mean_deg']}\n"
        "      units: \"degrees_clockwise_from_N\"\n"
        "      source: \"Circular mean of FL HU landfall headings via mean resultant vector:"
        " atan2(mean_sin, mean_cos) mod 360. See calibration/track_stats.py.\"\n"
        "    heading_arithmetic_mean_deg:\n"
        f"      value: {circ['arithmetic_mean_deg']}\n"
        "      units: \"degrees_clockwise_from_N\"\n"
        "      source: \"Naive arithmetic mean of headings; differs from circular mean when"
        " distribution spans the 0/360 wrap. Stored for transparency only.\"\n"
        "    heading_circ_std_deg:\n"
        f"      value: {circ['circ_std_deg']}\n"
        "      units: \"degrees\"\n"
        "      source: \"Circular std: sqrt(-2*ln(R)) in radians converted to degrees;"
        " R = heading_resultant_R.\"\n"
        "    heading_resultant_R:\n"
        f"      value: {circ['resultant_R']}\n"
        "      units: \"dimensionless\"\n"
        "      source: \"Mean resultant length sqrt(mean_cos^2 + mean_sin^2);"
        " 0 = uniform, 1 = all identical.\"\n"
        "    speed_dist:\n"
        f"      value: \"{winner}\"\n"
        "      units: \"scipy_dist_name\"\n"
        "      source: \"MLE fit to FL HU landfall translation speeds; selected by AIC"
        " over lognormal and gamma (both floc=0). See calibration/track_stats.py.\"\n"
        "    speed_params:\n"
        f"      value: {params_str}\n"
        f"      units: \"scipy_{winner}_params_(shape_s,_loc,_scale)\"\n"
        f"      source: \"MLE params; reconstruct with scipy.stats.{winner}(*speed_params)."
        " Units of scale are km/h.\"\n"
        "    speed_mean_kmh:\n"
        f"      value: {mean_kmh}\n"
        "      units: \"km/h\"\n"
        "      source: \"Sample mean of bracketing-pair translation speeds at FL HU landfall.\"\n"
        "    speed_std_kmh:\n"
        f"      value: {std_kmh}\n"
        "      units: \"km/h\"\n"
        "      source: \"Sample std (ddof=1) of bracketing-pair translation speeds.\"\n"
        "    n_HU:\n"
        "      value: 112\n"
        "      units: \"count\"\n"
        "      source: \"HURDAT2 fl_landfalls.csv, status==HU, full record 1851-2024\"\n"
        "    n_used:\n"
        f"      value: {n_used}\n"
        "      units: \"count\"\n"
        "      source: \"Storms with a valid bracketing fix pair; n_HU - n_excluded.\"\n"
        "    n_excluded:\n"
        f"      value: {n_excluded}\n"
        "      units: \"count\"\n"
        "      source: \"Storms excluded: no bracketing pair or degenerate dt=0."
        " See calibration/track_stats.py output.\"\n"
    )

    out_lines = []
    inserted  = False
    for line in lines:
        if not inserted and line.startswith("vulnerability:"):
            out_lines.append(block + "\n")
            inserted = True
        out_lines.append(line)

    if not inserted:
        raise RuntimeError(
            "Could not find 'vulnerability:' anchor in config — insertion failed."
        )

    with open(_CFG_PATH, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    print(f"Config updated -> {_CFG_PATH}")


# ---------------------------------------------------------------------------
# G: Plot
# ---------------------------------------------------------------------------

def make_plot(headings, speeds, circ_mean_deg, winner, all_fits):
    fig = plt.figure(figsize=(13, 6))

    # ---- Panel A: Wind rose ------------------------------------------------
    ax_rose = fig.add_subplot(1, 2, 1, projection="polar")
    ax_rose.set_theta_zero_location("N")
    ax_rose.set_theta_direction(-1)   # clockwise = meteorological

    # 8 bins x 45 deg, centered on N(0), NE(45), E(90), SE(135),
    #                              S(180), SW(225), W(270), NW(315)
    # Shift by +22.5 so the N-bin captures [337.5, 22.5) before histogramming
    h          = np.asarray(headings) % 360.0
    h_shifted  = (h + 22.5) % 360.0
    counts, _  = np.histogram(h_shifted, bins=np.arange(0, 361, 45))

    centers_deg = np.arange(0, 360, 45)
    centers_rad = np.radians(centers_deg)
    bar_width   = math.radians(43)    # 2 deg gap between adjacent bars

    ax_rose.bar(
        centers_rad, counts, width=bar_width, bottom=0,
        alpha=0.75, color="#4878cf", edgecolor="white", linewidth=0.7,
    )

    # Circular mean arrow
    mu_rad = math.radians(circ_mean_deg)
    r_max  = max(counts) * 1.05
    ax_rose.annotate(
        "", xy=(mu_rad, r_max), xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.0,
                        mutation_scale=14),
    )
    ax_rose.text(
        mu_rad, r_max * 1.22,
        f"circ. mean\n{circ_mean_deg:.0f}°",
        ha="center", va="center", fontsize=8.5,
        color="#d62728", fontweight="bold",
    )

    # Radial ticks — dynamic based on max count
    max_c    = int(max(counts))
    tick_step = 5 if max_c <= 40 else 10
    yticks   = list(range(tick_step, max_c + tick_step, tick_step))
    ax_rose.set_yticks(yticks)
    ax_rose.set_yticklabels([str(t) for t in yticks], fontsize=7, color="#555555")
    ax_rose.set_rlabel_position(112)   # move count labels to avoid NW bars

    compass = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ax_rose.set_xticks(centers_rad)
    ax_rose.set_xticklabels(compass, fontsize=9)
    ax_rose.set_title(
        "Landfall Heading\n"
        "(direction of storm motion, n=112 HU landfalls)",
        fontsize=10, pad=18,
    )

    # ---- Panel B: Speed histogram + fitted PDFs ----------------------------
    ax_spd = fig.add_subplot(1, 2, 2)

    ax_spd.hist(
        speeds, bins=15, density=True, color="#4878cf", alpha=0.55,
        edgecolor="white", linewidth=0.5, label="Observed",
    )

    x_lo  = max(0.0, min(speeds) * 0.7)
    x_hi  = max(speeds) * 1.10
    x_pdf = np.linspace(x_lo + 1e-3, x_hi, 500)

    dist_cfg = {
        "lognormal": dict(color="#d62728", lw=2.0, ls="-"),
        "gamma":     dict(color="#ff7f0e", lw=1.8, ls="--"),
    }
    for dname, (p, aic) in all_fits.items():
        dist_fn = scipy.stats.lognorm if dname == "lognormal" else scipy.stats.gamma
        pdf_vals = dist_fn.pdf(x_pdf, *p)
        suffix   = "  [winner]" if dname == winner else ""
        ax_spd.plot(
            x_pdf, pdf_vals,
            label=f"{dname}  AIC={aic:.1f}{suffix}",
            **dist_cfg[dname],
        )

    mean_spd = float(np.mean(speeds))
    ax_spd.axvline(
        mean_spd, color="#333333", ls=":", lw=1.5,
        label=f"mean = {mean_spd:.1f} km/h ({mean_spd * _KMH_TO_KT:.1f} kt)",
    )

    ax_spd.set_xlim(x_lo, x_hi)
    ax_spd.set_xlabel("Translation speed  (km/h)", fontsize=10)
    ax_spd.set_ylabel("Probability density", fontsize=10)
    ax_spd.set_title(
        "Landfall Translation Speed\n"
        "(bracketing fix pair, n=112 HU landfalls)",
        fontsize=10,
    )
    ax_spd.legend(fontsize=8.5, framealpha=0.85)
    ax_spd.tick_params(labelsize=9)

    # Secondary x-axis in kt (set limits before tight_layout)
    ax_kt = ax_spd.twiny()
    ax_kt.set_xlim(x_lo * _KMH_TO_KT, x_hi * _KMH_TO_KT)
    ax_kt.set_xlabel("Translation speed  (kt)", fontsize=9)
    ax_kt.tick_params(labelsize=8)

    fig.suptitle(
        "FL Landfall Track Statistics  |  HURDAT2 1851-2024  |  Step 1.5a Part 2",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(_OUT_PLOT), exist_ok=True)
    fig.savefig(_OUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {_OUT_PLOT}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Step 1.5a Part 2 — FL landfall track statistics")
    print("=" * 60)

    hu, fx_by_storm = load_data()

    print("\nComputing bracketing fix pairs ...")
    records, excluded = compute_track_stats(hu, fx_by_storm)

    n_used     = len(records)
    n_excluded = len(excluded)
    print(f"  n_used={n_used}  n_excluded={n_excluded}")
    if excluded:
        for sid, reason in excluded:
            print(f"    EXCLUDED  {sid} -- {reason}")

    # Bracketing-pair dt diagnostics
    dt_arr  = np.array([r["dt_h"] for r in records])
    dt_min  = float(dt_arr.min())
    dt_med  = float(np.median(dt_arr))
    dt_max  = float(dt_arr.max())
    print(f"\nBracketing-pair dt (h):  min={dt_min:.2f}  "
          f"median={dt_med:.2f}  max={dt_max:.2f}")

    long_dt = [(r["storm_id"], r["year"], r["dt_h"])
               for r in records if r["dt_h"] > _DT_FLAG_H]
    if long_dt:
        print(f"  Storms with dt > {_DT_FLAG_H:.0f}h ({len(long_dt)}) "
              f"-- speed estimate less reliable (track gap):")
        for sid, yr, dt in sorted(long_dt, key=lambda x: -x[2]):
            print(f"    {sid}  {yr}  dt={dt:.1f}h")
    else:
        print(f"  All dt <= {_DT_FLAG_H:.0f}h. No track gaps.")

    headings = [r["heading_deg"] for r in records]
    speeds   = np.array([r["speed_kmh"] for r in records], dtype=float)

    # Circular statistics
    print("\nCircular statistics (heading):")
    circ = circular_stats(headings)
    diff = abs(circ["mean_deg"] - circ["arithmetic_mean_deg"])
    # Handle wrap-around difference
    diff = min(diff, 360.0 - diff)
    print(f"  Circular mean      : {circ['mean_deg']:.1f} deg")
    print(f"  Arithmetic mean    : {circ['arithmetic_mean_deg']:.1f} deg")
    print(f"  Difference         : {diff:.1f} deg  "
          f"({'circular != arithmetic as expected' if diff > 1 else 'nearly equal'})")
    print(f"  Circular std       : {circ['circ_std_deg']:.1f} deg")
    print(f"  Resultant R        : {circ['resultant_R']:.4f}  "
          f"(0=uniform, 1=all identical)")

    # Speed distribution
    print("\nSpeed distribution MLE:")
    winner, params, all_fits = fit_speed_dist(speeds)
    print(f"  Mean  : {np.mean(speeds):.1f} km/h  "
          f"({np.mean(speeds) * _KMH_TO_KT:.1f} kt)")
    print(f"  Std   : {np.std(speeds, ddof=1):.1f} km/h")
    print(f"  Range : {speeds.min():.1f} -- {speeds.max():.1f} km/h")

    # Write CSV (drop dt_h diagnostic column)
    df_out = pd.DataFrame(records).drop(columns=["dt_h"])
    os.makedirs(os.path.dirname(_OUT_CSV), exist_ok=True)
    df_out.to_csv(_OUT_CSV, index=False)
    print(f"\nSaved -> {_OUT_CSV}  ({len(df_out)} rows x {len(df_out.columns)} cols)")

    # Update config
    print(f"\nUpdating {_CFG_PATH} ...")
    write_config(circ, winner, params, speeds, n_used, n_excluded)

    # Plot
    print("\nGenerating plot ...")
    make_plot(headings, speeds, circ["mean_deg"], winner, all_fits)

    print()
    print("=" * 60)
    print("track_stats.py complete")
    print("=" * 60)
    print("Next: Step 1.5b -- update model/hazard.py to use calibrated")
    print("  geography (s_samples_km KDE) + track stats (heading, speed).")
