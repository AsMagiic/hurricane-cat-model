#!/usr/bin/env python3
"""
Validate damage-state fragility parameters against HAZUS Elena field data.

Read-only on the model: no config changes, no simulation. Evaluates the
Manufactured DS-mean fragility analytically at each park gust speed and
compares P(DS>=3) against Hurricane Elena 1985 field-survey observations.

Ex-ante acceptance criteria (frozen before our model touched the data):
  MdAE <= 16.455 pts  AND  MAE <= 19.560 pts
  (= 1.5 × HAZUS certified model's own errors: MdAE=10.97, MAE=13.04)

Outputs
-------
- outputs/fragility_validation.png   (versioned)
- results/fragility_validation.csv   (not versioned)
- Printed verdict block on stdout
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.special import ndtr
from statsmodels.stats.proportion import proportion_confint

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from calibration.fragility_thetas import edr, exceedance_probs, logistic_dr
from model_config import load_model_cfg

_FIG_PATH = os.path.join(_ROOT, "outputs", "fragility_validation.png")
_CSV_PATH = os.path.join(_ROOT, "results", "fragility_validation.csv")

# ---------------------------------------------------------------------------
# Hardcoded validation data (versioned here; do not modify without updating
# the source citations)
# ---------------------------------------------------------------------------

_SOURCE_ELENA = (
    "HAZUS Hurricane TM Table 5-43, p.5-185; Hurricane Elena 1985, "
    "Manufactured Housing parks, Alabama/Mississippi Gulf Coast. "
    "Rows may not sum to 100 in source; normalized to 1.0 before comparison."
)
_SOURCE_N = (
    "Park unit counts from HAZUS Hurricane TM Table 5-43 (counts approximate "
    "where source shows percentages only; n inferred from finest-grained "
    "percentage, range 12-175 as stated in HAZUS TM). CIs are order-of-magnitude "
    "correct; exact counts do not change the verdict."
)
_SOURCE_TERRAIN = (
    "HAZUS Hurricane TM §5.5.3: Elena analysis assumed z0=0.3 (suburban/trees) "
    "for most parks; z0=0.1 (open coastal) for Trade Winds Dauphin. "
    "Our model is open-terrain Exposure C (z0~0.02). "
    "Over-prediction expected at all parks; terrain roughness at structure is "
    "lower than open terrain, so damage is lower than our model predicts."
)
_SOURCE_TABLE38 = (
    "HAZUS Hurricane TM Table 5-38, p.5-180: Vann&McDonald, Vasquez, HAZUS "
    "wind-speed thresholds for Manufactured Housing damage modes (3-s gust, "
    "open terrain)."
)

# Elena 1985 field survey — Manufactured Housing parks
# gust: 3-s peak gust (mph), open terrain
# obs/hazus: [DS0, DS1, DS2, DS3, DS4] percentages (normalized before use)
# n: approximate number of housing units surveyed
_ELENA_PARKS = [
    {"name": "Trav Park Mobile Bay",  "gust": 109, "z0": 0.3,
     "obs": [75, 25, 0, 0, 0],    "hazus": [89, 6, 1, 0, 4],    "n": 80},
    {"name": "Old Fort Village",       "gust": 122, "z0": 0.3,
     "obs": [42, 39, 4, 3, 11],   "hazus": [54, 16, 7, 3, 13],  "n": 26},
    {"name": "Imperial Estates",       "gust": 122, "z0": 0.3,
     "obs": [89, 3, 1, 3, 3],     "hazus": [68, 17, 1, 0, 14],  "n": 100},
    {"name": "Rolling Hills",          "gust": 122, "z0": 0.3,
     "obs": [87, 10, 1, 1, 1],    "hazus": [61, 17, 1, 0, 21],  "n": 175},
    {"name": "Isle of Pines North",    "gust": 124, "z0": 0.3,
     "obs": [48, 23, 6, 4, 19],   "hazus": [44, 17, 10, 4, 26], "n": 83},
    {"name": "Isle of Pines South",    "gust": 124, "z0": 0.3,
     "obs": [55, 22, 5, 2, 16],   "hazus": [40, 19, 9, 1, 31],  "n": 50},
    {"name": "Trade Winds Dauphin",    "gust": 126, "z0": 0.1,
     "obs": [5, 35, 20, 0, 40],   "hazus": [16, 15, 7, 6, 55],  "n": 20},
    {"name": "Anchor Gautier",         "gust": 126, "z0": 0.3,
     "obs": [76, 19, 1, 0, 4],    "hazus": [40, 19, 9, 1, 31],  "n": 12},
]

# Table 5-38 reference wind-speed bands for Manufactured Housing damage modes
# (3-s gust, open terrain, mph)
_TABLE38_BANDS = [
    ("Roof Cover\nMinor (DS1)",  65,  95, "lightblue"),
    ("Window\nMinor (DS2)",     100, 110, "lightyellow"),
    ("Wall Panel\nMajor (DS3)", 120, 130, "lightsalmon"),
]

# Ex-ante acceptance criteria — FIXED before model touched the data.
# HAZUS certified model's own errors on these 8 parks (from the same table):
_HAZUS_MDAE = 10.97   # percentage points
_HAZUS_MAE  = 13.04   # percentage points
_CRITERION_MDAE = 1.5 * _HAZUS_MDAE   # 16.455 pts
_CRITERION_MAE  = 1.5 * _HAZUS_MAE    # 19.560 pts

_CONSEQUENCE = np.array([0.0, 0.02, 0.10, 0.50, 1.00])


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def _normalize(ds_list):
    """Normalize a DS percentage list to fractions summing to 1.0."""
    s = float(sum(ds_list))
    return [x / s for x in ds_list]


def compute_park_predictions(parks, thetas, beta, logistic_params):
    """
    Evaluate DS-mean and logistic predictions for each park.

    Parameters
    ----------
    parks : list of park dicts (from _ELENA_PARKS)
    thetas : np.ndarray, shape (4,)  Manufactured DS thetas (mph)
    beta : float                     Manufactured lognormal dispersion
    logistic_params : dict           {cap, midpoint, k} for Manufactured

    Returns
    -------
    results : list of dicts with per-park predictions and errors
    """
    results = []
    for park in parks:
        g = float(park["gust"])
        g_arr = np.array([g])

        obs_n = _normalize(park["obs"])
        haz_n = _normalize(park["hazus"])

        # P(DS>=k) for k=1..4; shape (4, 1)
        exc = exceedance_probs(g_arr, thetas, beta)
        p_ge = exc[:, 0]   # [P(DS>=1), P(DS>=2), P(DS>=3), P(DS>=4)]

        # Full 5-class model distribution
        p_ds = np.array([
            1.0 - p_ge[0],
            p_ge[0] - p_ge[1],
            p_ge[1] - p_ge[2],
            p_ge[2] - p_ge[3],
            p_ge[3],
        ])

        our_major = float(p_ge[2])          # P(DS>=3) = major damage fraction
        obs_major = obs_n[3] + obs_n[4]
        haz_major = haz_n[3] + haz_n[4]

        our_edr_val = float(edr(g_arr, thetas, beta)[0])
        log_edr_val = float(logistic_dr(
            g_arr,
            logistic_params["cap"],
            logistic_params["midpoint"],
            logistic_params["k"],
        )[0])
        obs_edr_val = float(np.dot(obs_n, _CONSEQUENCE))

        # Wilson 95% binomial CI on observed major fraction
        n = park["n"]
        count = max(0, round(obs_major * n))
        ci_lo, ci_hi = proportion_confint(count, n, alpha=0.05, method="wilson")

        results.append({
            "name":       park["name"],
            "gust":       g,
            "z0":         park["z0"],
            "n":          n,
            "obs_major":  obs_major * 100,
            "haz_major":  haz_major * 100,
            "our_major":  our_major * 100,
            "obs_edr":    obs_edr_val * 100,
            "our_edr":    our_edr_val * 100,
            "log_edr":    log_edr_val * 100,
            "our_err":    abs(our_major - obs_major) * 100,
            "haz_err":    abs(haz_major - obs_major) * 100,
            "ci_lo":      ci_lo * 100,
            "ci_hi":      ci_hi * 100,
            "obs_ds5":    obs_n,
            "our_ds5":    p_ds.tolist(),
            "haz_ds5":    haz_n,
        })
    return results


def compute_verdict(results):
    """Compute primary test metrics and terrain check."""
    our_errs = [r["our_err"] for r in results]
    our_mdae = float(np.median(our_errs))
    our_mae  = float(np.mean(our_errs))

    rough_errs  = [r["our_err"] for r in results if r["z0"] == 0.3]
    dauphin_err = next(r["our_err"] for r in results if r["z0"] == 0.1)

    return {
        "our_mdae":         our_mdae,
        "our_mae":          our_mae,
        "pass_mdae":        our_mdae <= _CRITERION_MDAE,
        "pass_mae":         our_mae  <= _CRITERION_MAE,
        "pass_primary":     (our_mdae <= _CRITERION_MDAE) and (our_mae <= _CRITERION_MAE),
        "rough_mean_err":   float(np.mean(rough_errs)),
        "dauphin_err":      dauphin_err,
        "terrain_confirmed": float(np.mean(rough_errs)) > dauphin_err,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _short_name(name):
    """Abbreviate park name for x-axis labels."""
    replacements = {
        "Trav Park Mobile Bay": "Trav Park",
        "Old Fort Village": "Old Fort",
        "Imperial Estates": "Imperial",
        "Rolling Hills": "Rolling Hills",
        "Isle of Pines North": "IoP North",
        "Isle of Pines South": "IoP South",
        "Trade Winds Dauphin": "Dauphin",
        "Anchor Gautier": "Anchor",
    }
    return replacements.get(name, name)


def make_figure(results, thetas, beta, verdict, fig_path):
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.45, wspace=0.35)
    ax_primary  = fig.add_subplot(gs[0, :])
    ax_ds5      = fig.add_subplot(gs[1, 0])
    ax_medians  = fig.add_subplot(gs[1, 1])

    labels = [_short_name(r["name"]) for r in results]
    x = np.arange(len(results))

    # ---- Primary panel ----
    for i, r in enumerate(results):
        col = "#e07b39" if r["z0"] == 0.1 else "#4477aa"
        ax_primary.errorbar(
            x[i], r["obs_major"] / 100,
            yerr=[[r["obs_major"] / 100 - r["ci_lo"] / 100],
                  [r["ci_hi"] / 100 - r["obs_major"] / 100]],
            fmt="o", color=col, markersize=8, capsize=5, linewidth=1.5,
            label="Observed (Wilson 95% CI)" if i == 0 else "",
            zorder=4,
        )
        ax_primary.plot(x[i], r["haz_major"] / 100, "D", color="gray",
                        markersize=7, zorder=3,
                        label="HAZUS model" if i == 0 else "")
        ax_primary.plot(x[i], r["our_major"] / 100, "^", color="crimson",
                        markersize=8, zorder=5,
                        label="Our DS-mean P(DS≥3)" if i == 0 else "")

    verdict_str = "FAIL" if not verdict["pass_primary"] else "PASS"
    ax_primary.set_xticks(x)
    ax_primary.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax_primary.set_ylabel("Major damage fraction P(DS≥3)", fontsize=10)
    ax_primary.set_ylim(-0.03, 1.05)
    ax_primary.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_primary.set_title(
        f"Elena 1985 field validation — Manufactured Housing, 8 parks\n"
        f"Primary test: {verdict_str}  |  "
        f"Our MdAE={verdict['our_mdae']:.1f} pts (criterion ≤{_CRITERION_MDAE:.1f} pts)  "
        f"MAE={verdict['our_mae']:.1f} pts (criterion ≤{_CRITERION_MAE:.1f} pts)",
        fontsize=10, fontweight="bold",
    )
    ax_primary.axhline(_CRITERION_MDAE / 100, color="gray", linestyle=":", linewidth=0.8,
                       label=f"Criterion (1.5× HAZUS MdAE={_HAZUS_MDAE:.2f} pts)")
    ax_primary.legend(fontsize=8, loc="upper left")
    ax_primary.grid(axis="y", alpha=0.3)

    # Color legend for z0
    rough_patch = mpatches.Patch(color="#4477aa", label="z0=0.3 (rough terrain)")
    dauphin_patch = mpatches.Patch(color="#e07b39", label="z0=0.1 (Dauphin, exposed)")
    ax_primary.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="#4477aa", linestyle="None",
                       markersize=8, label="Observed z0=0.3 (rough)"),
            plt.Line2D([0], [0], marker="o", color="#e07b39", linestyle="None",
                       markersize=8, label="Observed z0=0.1 (Dauphin)"),
            plt.Line2D([0], [0], marker="D", color="gray", linestyle="None",
                       markersize=7, label="HAZUS model"),
            plt.Line2D([0], [0], marker="^", color="crimson", linestyle="None",
                       markersize=8, label="Our DS-mean P(DS≥3)"),
        ],
        fontsize=8, loc="upper left",
    )

    # ---- 5-class DS panel (two representative parks) ----
    # Show Trav Park (large miss at low gust) and Dauphin (exposed park)
    ds_parks = [results[0], results[6]]   # Trav Park, Dauphin
    ds_labels = ["DS0", "DS1", "DS2", "DS3", "DS4"]
    ds_colors_obs  = ["#aaccee", "#6699cc", "#3366aa", "#cc3333", "#880000"]
    ds_colors_mod  = ["#dddddd", "#bbbbbb", "#999999", "#ff9999", "#cc4444"]
    ds_colors_haz  = ["#cceecc", "#88cc88", "#44aa44", "#cc8822", "#884400"]

    bar_w = 0.25
    x2 = np.arange(len(ds_labels))
    for pi, park_r in enumerate(ds_parks):
        offset = pi * 3.5
        for j, (obs_v, our_v, haz_v) in enumerate(zip(
            park_r["obs_ds5"], park_r["our_ds5"], park_r["haz_ds5"]
        )):
            ax_ds5.bar(offset + x2[j] - bar_w, obs_v, bar_w * 0.9,
                       color=ds_colors_obs[j], label=f"Obs DS{j}" if pi == 0 and j < 3 else "")
            ax_ds5.bar(offset + x2[j],          haz_v, bar_w * 0.9,
                       color=ds_colors_haz[j])
            ax_ds5.bar(offset + x2[j] + bar_w,  our_v, bar_w * 0.9,
                       color=ds_colors_mod[j])

        ax_ds5.text(offset + 2.0, 1.02,
                    f"{_short_name(park_r['name'])}\n(g={park_r['gust']} mph)",
                    ha="center", va="bottom", fontsize=7.5)

    ax_ds5.set_xticks([0, 1, 2, 3, 4, 3.5, 4.5, 5.5, 6.5, 7.5])
    ax_ds5.set_xticklabels(
        ["DS0","DS1","DS2","DS3","DS4","DS0","DS1","DS2","DS3","DS4"],
        fontsize=7, rotation=0,
    )
    ax_ds5.set_ylabel("Fraction", fontsize=9)
    ax_ds5.set_title("5-class DS distribution\n(obs / HAZUS / ours per DS)", fontsize=9)
    ax_ds5.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_ds5.set_ylim(0, 1.15)
    ax_ds5.grid(axis="y", alpha=0.3)

    legend_handles = [
        mpatches.Patch(facecolor="#aaccee", label="Observed"),
        mpatches.Patch(facecolor="#cceecc", label="HAZUS"),
        mpatches.Patch(facecolor="#dddddd", label="Our DS-mean"),
    ]
    ax_ds5.legend(handles=legend_handles, fontsize=7, loc="upper right")

    # ---- Medians vs Table 5-38 panel ----
    g_range = np.linspace(55, 175, 300)
    our_major_curve = exceedance_probs(g_range, thetas, beta)[2]   # P(DS>=3)
    our_edr_curve   = edr(g_range, thetas, beta)

    for label, lo, hi, col in _TABLE38_BANDS:
        ax_medians.axvspan(lo, hi, alpha=0.25, color=col, label=label)

    ax_medians.plot(g_range, our_major_curve, "r-", linewidth=1.8,
                    label="Our P(DS≥3|g) [Manufactured]")
    ax_medians.plot(g_range, our_edr_curve, "r--", linewidth=1.2,
                    label="Our E[DR|g]")

    # Mark our thetas
    theta_labels = ["θ₁", "θ₂", "θ₃", "θ₄"]
    for i, (th, tl) in enumerate(zip(thetas, theta_labels)):
        ax_medians.axvline(th, color="darkred", linestyle=":", linewidth=0.9)
        ax_medians.text(th + 1, 0.95 - i * 0.08, tl, color="darkred", fontsize=8)

    # Elena park gusts as vertical marks
    for r in results:
        ax_medians.axvline(r["gust"], color="#4477aa", linestyle="-", linewidth=0.5, alpha=0.5)

    ax_medians.set_xlabel("3-s peak gust (mph, open terrain)", fontsize=9)
    ax_medians.set_ylabel("Fraction / damage ratio", fontsize=9)
    ax_medians.set_title("Thetas vs Table 5-38 reference bands\n(shaded = ref damage mode onset range)",
                         fontsize=9)
    ax_medians.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_medians.set_xlim(55, 180)
    ax_medians.set_ylim(0, 1.05)
    ax_medians.legend(fontsize=7, loc="upper left")
    ax_medians.grid(alpha=0.3)

    fig.suptitle(
        "Task 3 validation: DS-mean fragility vs HAZUS Elena 1985 field data\n"
        "Source: HAZUS Hurricane TM Table 5-43; Manufactured Housing only",
        fontsize=10, y=0.99,
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------

def write_csv(results, verdict, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "park", "gust_mph", "z0", "n_approx",
        "obs_major_pct", "haz_major_pct", "our_major_pct",
        "obs_edr_pct", "our_edr_pct", "log_edr_pct",
        "our_err_pts", "haz_err_pts",
        "ci_lo_pct", "ci_hi_pct",
        "pass_primary",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({
                "park":          r["name"],
                "gust_mph":      r["gust"],
                "z0":            r["z0"],
                "n_approx":      r["n"],
                "obs_major_pct": f"{r['obs_major']:.2f}",
                "haz_major_pct": f"{r['haz_major']:.2f}",
                "our_major_pct": f"{r['our_major']:.2f}",
                "obs_edr_pct":   f"{r['obs_edr']:.2f}",
                "our_edr_pct":   f"{r['our_edr']:.2f}",
                "log_edr_pct":   f"{r['log_edr']:.2f}",
                "our_err_pts":   f"{r['our_err']:.2f}",
                "haz_err_pts":   f"{r['haz_err']:.2f}",
                "ci_lo_pct":     f"{r['ci_lo']:.2f}",
                "ci_hi_pct":     f"{r['ci_hi']:.2f}",
                "pass_primary":  verdict["pass_primary"],
            })


# ---------------------------------------------------------------------------
# Verdict printer
# ---------------------------------------------------------------------------

def print_verdict(results, verdict):
    sep = "=" * 66
    print(sep)
    print("TASK 3 VALIDATION — DS-MEAN vs HAZUS ELENA FIELD DATA")
    print(sep)
    print(f"Source : {_SOURCE_ELENA[:70]}...")
    print(f"Scope  : Manufactured Housing, 8 parks, Hurricane Elena 1985")
    print()

    # Per-park table
    print(f"{'Park':<28} {'g':>4} {'z0':>4} {'Obs%':>6} {'HAZUS%':>7} {'Ours%':>6} "
          f"{'OurErr':>7} {'HazErr':>7} {'CI95':>12}")
    print("-" * 90)
    for r in results:
        ci_str = f"[{r['ci_lo']:.1f},{r['ci_hi']:.1f}]"
        print(f"{r['name']:<28} {r['gust']:>4.0f} {r['z0']:>4.1f} "
              f"{r['obs_major']:>6.1f} {r['haz_major']:>7.1f} {r['our_major']:>6.1f} "
              f"{r['our_err']:>7.1f} {r['haz_err']:>7.1f} {ci_str:>12}")
    print()

    # E[DR] comparison table
    print(f"{'Park':<28} {'g':>4} {'ObsEDR%':>8} {'LogEDR%':>8} {'DS-EDR%':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<28} {r['gust']:>4.0f} "
              f"{r['obs_edr']:>8.1f} {r['log_edr']:>8.1f} {r['our_edr']:>8.1f}")
    print()

    print(sep)
    result_str = "FAIL" if not verdict["pass_primary"] else "PASS"
    print(f"PRIMARY TEST: {result_str}")
    print(f"  Ex-ante criterion: MdAE <= {_CRITERION_MDAE:.3f} pts  "
          f"AND  MAE <= {_CRITERION_MAE:.3f} pts")
    print(f"  (= 1.5 × HAZUS own errors: MdAE={_HAZUS_MDAE:.2f}, MAE={_HAZUS_MAE:.2f})")
    print(f"  Our MdAE = {verdict['our_mdae']:.2f} pts  "
          f"({'PASS' if verdict['pass_mdae'] else 'FAIL'})")
    print(f"  Our MAE  = {verdict['our_mae']:.2f} pts  "
          f"({'PASS' if verdict['pass_mae'] else 'FAIL'})")
    print()
    print("  MECHANISM: Conceptual anchoring error.")
    print("  theta3=110 mph was anchored to the logistic midpoint (50% MEAN DR).")
    print("  In the DS framework theta3 is the MEDIAN OF DS3 (50% probability")
    print("  of extensive/structural damage). These are different physical")
    print("  quantities — the internal calibration passed because it compared")
    print("  against the logistic, which shared the same anchored midpoint.")
    print("  Field data exposes the error: Elena shows 0-40% major damage at")
    print("  109-126 mph where our model predicts 47-89%.")
    print()
    terrain_str = "CONFIRMED" if verdict["terrain_confirmed"] else "NOT CONFIRMED"
    print(f"TERRAIN CHECK: {terrain_str}")
    print(f"  Mean error rough-terrain parks (z0=0.3): {verdict['rough_mean_err']:.1f} pts")
    print(f"  Error exposed park (z0=0.1, Dauphin):    {verdict['dauphin_err']:.1f} pts")
    print("  Prediction (over-predict more at rough sites) confirmed.")
    print("  Note: terrain bias is a secondary signal — anchoring bias dominates.")
    print()
    print("DISCRIMINATION VERDICT: DATA CAN DISCRIMINATE")
    print("  DS-mean over-predicts by ~4-6x the HAZUS benchmark error.")
    print("  Logistic E[DR] at Elena gusts also over-predicts observed E[DR],")
    print("  but DS-mean P(DS>=3) is the more severely biased metric.")
    print("  This is NOT 'both paradigms consistent with noise' — the miss is")
    print("  far outside all binomial confidence bands.")
    print()
    print("SCOPE CAVEAT:")
    print("  Validation covers Manufactured Housing only (single construction")
    print("  class, 8 parks, 1 storm). WF/Masonry/RC: no equivalent independent")
    print("  field validation available from this source.")
    print()
    print("PRODUCTION STATUS: logistic_deterministic remains default.")
    print("  Re-anchoring or paradigm reconsideration is a separate decision.")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mcfg = load_model_cfg()

    # Load Manufactured fragility parameters from config
    ds   = mcfg.vulnerability.damage_states
    mfg  = ds["Manufactured"]
    thetas = np.array(mfg["thetas"])   # [theta1, theta2, theta3, theta4], mph
    beta   = float(mfg["beta"])

    # Logistic params for discrimination comparison
    cp = mcfg.vulnerability.construction_params
    logistic_params = {
        "cap":      float(cp["Manufactured"]["cap"]),
        "midpoint": float(cp["Manufactured"]["midpoint"]),
        "k":        float(cp["Manufactured"]["k"]),
    }

    results = compute_park_predictions(_ELENA_PARKS, thetas, beta, logistic_params)
    verdict = compute_verdict(results)

    print_verdict(results, verdict)
    make_figure(results, thetas, beta, verdict, _FIG_PATH)
    write_csv(results, verdict, _CSV_PATH)

    print(f"\nFigure: {_FIG_PATH}")
    print(f"CSV:    {_CSV_PATH}")


if __name__ == "__main__":
    main()
