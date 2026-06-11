"""
Calibrate FL HU landfall intensity distribution (vmax_kt).

Methodology
-----------
1. Load fl_landfalls.csv, filter status == 'HU'. Report n (full/satellite era).
2. Fit TWO candidate families by MLE, support [64, inf):
   a. Shifted Weibull: weibull_min(c, loc=64, scale) via scipy floc=64.
      64 kt is the HU definitional threshold, not a sampling cutoff; fitting
      excess-over-threshold directly is physically motivated — no renormalization.
   b. Truncated Lognormal: parent Lognormal(mu, sigma) conditioned on X >= 64.
      LL(mu,sigma) = sum_i [norm.logpdf(log xi; mu,sigma) - log xi]
                    - n * norm.logsf((log 64 - mu)/sigma)
      Optimised via L-BFGS-B; initialised at untruncated lognormal MLE.
3. Sensitivity check: fit both on satellite era (1966-2024) vs full record
   (1851-2024); compare parameters and implied P(Cat 4+). State if material.
4. Selection: lower upper-tail MAD (top-quartile QQ deviation) is primary;
   AIC is tiebreaker.
5. Category frequency table: empirical vs selected distribution (full record).
6. Two-panel figure: histogram + PDFs (left); QQ-plot with tail annotation (right).
7. Write selected distribution parameters to config/model_v3.yaml (intensity:).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from scipy.stats import norm as _norm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from model_config import load_calibration_cfg

_ccfg      = load_calibration_cfg()
_FL_CSV    = os.path.join(_ROOT, _ccfg.fl_landfalls.processed_path)
_FIG       = os.path.join(_ROOT, _ccfg.intensity_calibration.figure_path)
_MODEL_CFG = os.path.join(_ROOT, "config", "model_v3.yaml")

HU_LB      = 64.0         # kt — HU definitional threshold (Saffir-Simpson)
_LB_LOG    = np.log(HU_LB)

FULL_START = 1851
SAT_START  = 1966
END        = 2024

# SS category lower bounds in kt (NHC 1-min sustained wind, open on right)
# Cat i occupies [_SS_KT[i], _SS_KT[i+1]);  last interval is [137, inf)
_SS_KT = [64, 83, 96, 113, 137]
_SS_LABELS = [
    "Cat 1 (64-82 kt)",
    "Cat 2 (83-95 kt)",
    "Cat 3 (96-112 kt)",
    "Cat 4 (113-136 kt)",
    "Cat 5 (>=137 kt)",
]

_SOURCE_TMPL = (
    "MLE {dist_label} fit to FL HU landfall Vmax, HURDAT2 full record "
    "{start}-{end} (n={n}); selected over {alt_label} on upper-tail QQ MAD "
    "({sel_mad:.2f} kt vs {alt_mad:.2f} kt), dAIC={daic:+.2f}. "
    "Full record used for intensity (tail events scarce; pre-satellite Vmax "
    "measurement bias non-directional) vs satellite era for frequency "
    "(count detection bias). See calibration/intensity.py."
)


# ---------------------------------------------------------------------------
# Weibull fit
# ---------------------------------------------------------------------------

def _fit_weibull(vmax: np.ndarray):
    """Shifted Weibull MLE (loc=64 fixed). Returns (c, scale, ll, aic)."""
    c, _loc, scale = scipy.stats.weibull_min.fit(vmax, floc=HU_LB)
    ll  = float(np.sum(scipy.stats.weibull_min.logpdf(vmax, c, loc=HU_LB, scale=scale)))
    aic = 4.0 - 2.0 * ll   # k=2: c, scale
    return float(c), float(scale), ll, aic


# ---------------------------------------------------------------------------
# Truncated Lognormal fit
# ---------------------------------------------------------------------------

def _trunclognorm_nll(params, y):
    """
    Negative log-likelihood for truncated lognormal (lower bound = HU_LB).

    params = [mu, log_sigma]; y = log(vmax_kt).
    Includes the Jacobian term -log(x) so the AIC is on the same scale as
    the Weibull AIC (both are evaluated in the original x domain).
    """
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    # logsf = log(1 - CDF) = log(P(X >= lb)) — norm.logsf is numerically stable
    log_sf_lb = float(_norm.logsf((_LB_LOG - mu) / sigma))
    ll = (
        float(np.sum(_norm.logpdf(y, loc=mu, scale=sigma) - y))
        - len(y) * log_sf_lb
    )
    return -ll


def _fit_trunclognorm(vmax: np.ndarray):
    """Truncated Lognormal MLE (lower bound HU_LB). Returns (mu, sigma, ll, aic)."""
    y   = np.log(vmax)
    mu0 = float(y.mean())
    ls0 = float(np.log(y.std(ddof=1)))
    res = scipy.optimize.minimize(
        _trunclognorm_nll,
        x0=[mu0, ls0],
        args=(y,),
        method="L-BFGS-B",
        bounds=[(None, None), (-4.0, 4.0)],
        options={"ftol": 1e-12, "gtol": 1e-8},
    )
    if not res.success:
        raise RuntimeError(f"TruncLognorm MLE did not converge: {res.message}")
    mu    = float(res.x[0])
    sigma = float(np.exp(res.x[1]))
    ll    = -float(res.fun)
    aic   = 4.0 - 2.0 * ll   # k=2: mu, sigma
    return mu, sigma, ll, aic


# ---------------------------------------------------------------------------
# Distribution utilities
# ---------------------------------------------------------------------------

def _weibull_ppf(p: np.ndarray, c: float, scale: float) -> np.ndarray:
    return scipy.stats.weibull_min.ppf(p, c, loc=HU_LB, scale=scale)


def _weibull_sf(x: float, c: float, scale: float) -> float:
    return float(scipy.stats.weibull_min.sf(x, c, loc=HU_LB, scale=scale))


def _trunclognorm_ppf(p: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    p_lb       = float(_norm.cdf((_LB_LOG - mu) / sigma))
    norm_const = 1.0 - p_lb
    return np.exp(mu + sigma * _norm.ppf(p * norm_const + p_lb))


def _trunclognorm_sf(x: float, mu: float, sigma: float) -> float:
    p_lb = float(_norm.cdf((_LB_LOG - mu) / sigma))
    return float(_norm.sf((np.log(x) - mu) / sigma)) / (1.0 - p_lb)


def _trunclognorm_pdf_grid(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    p_lb       = float(_norm.cdf((_LB_LOG - mu) / sigma))
    norm_const = 1.0 - p_lb
    log_pdf    = _norm.logpdf(np.log(x), loc=mu, scale=sigma) - np.log(x) - np.log(norm_const)
    return np.exp(log_pdf)


# ---------------------------------------------------------------------------
# Selection metric: upper-tail MAD in QQ-space
# ---------------------------------------------------------------------------

def _plotting_positions(n: int) -> np.ndarray:
    """Hazen plotting positions: (i - 0.5)/n for i=1,...,n."""
    return (np.arange(1, n + 1) - 0.5) / n


def _tail_mad(vmax_sorted: np.ndarray, ppf_fn) -> float:
    """Median absolute deviation of QQ upper quartile (p >= 0.75).

    Measures how well the distribution fits the heavy-wind tail — the primary
    selection criterion for a CAT model where Cat 4/5 losses drive the PML.
    """
    n    = len(vmax_sorted)
    p    = _plotting_positions(n)
    mask = p >= 0.75
    emp  = vmax_sorted[mask]
    theo = ppf_fn(p[mask])
    return float(np.median(np.abs(emp - theo)))


# ---------------------------------------------------------------------------
# Category frequency table
# ---------------------------------------------------------------------------

def _category_probs_weibull(c: float, scale: float) -> np.ndarray:
    bounds = _SS_KT + [np.inf]
    probs  = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if np.isinf(hi):
            probs.append(_weibull_sf(lo, c, scale))
        else:
            probs.append(
                scipy.stats.weibull_min.cdf(hi, c, HU_LB, scale) -
                scipy.stats.weibull_min.cdf(lo, c, HU_LB, scale)
            )
    return np.array(probs)


def _category_probs_trunclognorm(mu: float, sigma: float) -> np.ndarray:
    p_lb       = float(_norm.cdf((_LB_LOG - mu) / sigma))
    norm_const = 1.0 - p_lb
    bounds     = _SS_KT + [np.inf]
    probs      = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if np.isinf(hi):
            probs.append(_trunclognorm_sf(lo, mu, sigma))
        else:
            p_lo = float(_norm.cdf((np.log(lo) - mu) / sigma))
            p_hi = float(_norm.cdf((np.log(hi) - mu) / sigma))
            probs.append((p_hi - p_lo) / norm_const)
    return np.array(probs)


def _empirical_cat_probs(vmax: np.ndarray) -> np.ndarray:
    bounds = _SS_KT + [np.inf]
    n      = len(vmax)
    probs  = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if np.isinf(hi):
            probs.append(float((vmax >= lo).sum()) / n)
        else:
            probs.append(float(((vmax >= lo) & (vmax < hi)).sum()) / n)
    return np.array(probs)


# ---------------------------------------------------------------------------
# Config write — targeted text insertion (no yaml.dump)
# ---------------------------------------------------------------------------

def _write_intensity_config(
    path: str,
    dist_name: str,
    params: dict,
    source: str,
) -> None:
    """
    Insert or replace the intensity: block in config/model_v3.yaml.

    dist_name = "weibull_min"  -> params must contain {loc, shape_c, scale}
    dist_name = "trunclognorm" -> params must contain {loc, mu_log, sigma_log}

    Preserves all existing comments and formatting (no yaml.dump round-trip).
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    loc_src  = "HU definitional threshold (Saffir-Simpson 64 kt)"
    param_src = source  # same full source on each fitted parameter leaf

    if dist_name == "weibull_min":
        block = (
            "intensity:\n"
            "  distribution:\n"
            f'    value: "weibull_min"\n'
            f'    units: "scipy_dist_name"\n'
            f'    source: "{source}"\n'
            "  loc:\n"
            f"    value: {params['loc']:.1f}\n"
            f'    units: "kt"\n'
            f'    source: "{loc_src}"\n'
            "  shape_c:\n"
            f"    value: {round(params['shape_c'], 4)}\n"
            f'    units: "dimensionless"\n'
            f'    source: "{param_src}"\n'
            "  scale:\n"
            f"    value: {round(params['scale'], 4)}\n"
            f'    units: "kt"\n'
            f'    source: "{param_src}"\n'
        )
    else:  # trunclognorm
        block = (
            "intensity:\n"
            "  distribution:\n"
            f'    value: "trunclognorm"\n'
            f'    units: "scipy_dist_name"\n'
            f'    source: "{source}"\n'
            "  loc:\n"
            f"    value: {params['loc']:.1f}\n"
            f'    units: "kt"\n'
            f'    source: "{loc_src}"\n'
            "  mu_log:\n"
            f"    value: {round(params['mu_log'], 4)}\n"
            f'    units: "log_kt"\n'
            f'    source: "{param_src}"\n'
            "  sigma_log:\n"
            f"    value: {round(params['sigma_log'], 4)}\n"
            f'    units: "dimensionless"\n'
            f'    source: "{param_src}"\n'
        )

    # Case 1: intensity: already present — replace entire block.
    start_i = None
    for i, line in enumerate(lines):
        if line.rstrip() == "intensity:":
            start_i = i
            break

    if start_i is not None:
        j = start_i + 1
        while j < len(lines):
            stripped = lines[j].rstrip()
            if stripped and not stripped.startswith(" ") and not stripped.startswith("#"):
                break
            j += 1
        lines[start_i:j] = [block + "\n"]
    else:
        # First run — insert before hazard:.
        for i, line in enumerate(lines):
            if line.rstrip() == "hazard:":
                lines.insert(i, block + "\n")
                break
        else:
            raise RuntimeError(
                "_write_intensity_config: cannot find 'hazard:' anchor in " + path
            )

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _make_figure(
    vmax: np.ndarray,
    wb_c: float, wb_scale: float,
    tl_mu: float, tl_sigma: float,
    selected: str,
    path: str,
) -> None:
    vmax_sorted = np.sort(vmax)
    n           = len(vmax_sorted)
    p           = _plotting_positions(n)

    x_grid  = np.linspace(HU_LB, vmax.max() + 25.0, 600)
    pdf_wb  = scipy.stats.weibull_min.pdf(x_grid, wb_c, loc=HU_LB, scale=wb_scale)
    pdf_tl  = _trunclognorm_pdf_grid(x_grid, tl_mu, tl_sigma)

    wb_theo  = _weibull_ppf(p, wb_c, wb_scale)
    tl_theo  = _trunclognorm_ppf(p, tl_mu, tl_sigma)

    cat45_mask = vmax_sorted >= 113.0
    n_cat45    = int(cat45_mask.sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left: histogram + fitted PDFs ----
    ax1.hist(vmax, bins=20, density=True, color="#b0c4de", alpha=0.75,
             edgecolor="#888888", linewidth=0.4, label="Empirical (HU landfalls)")
    ax1.plot(x_grid, pdf_wb, color="#1f77b4", linewidth=2.0,
             label=f"Weibull (c={wb_c:.3f}, scale={wb_scale:.1f} kt)")
    ax1.plot(x_grid, pdf_tl, color="#d62728", linewidth=2.0, linestyle="--",
             label=f"TruncLognorm (mu={tl_mu:.3f}, sigma={tl_sigma:.3f})")
    for kt_bound, cat_label in zip(_SS_KT[1:], ["Cat 2", "Cat 3", "Cat 4", "Cat 5"]):
        ax1.axvline(kt_bound, color="#aaaaaa", linewidth=0.7, linestyle=":")
        ax1.text(kt_bound + 0.5, ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0 else 0.04,
                 cat_label, fontsize=6, color="#777777", va="top", rotation=90)
    ax1.set_xlabel("Vmax at FL landfall (kt)", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.set_title("FL HU Landfall Intensity\nFitted PDFs vs Empirical", fontsize=11)
    ax1.legend(fontsize=8)

    # ---- Right: QQ-plot (empirical on x, theoretical on y) ----
    ax2.scatter(vmax_sorted[~cat45_mask], wb_theo[~cat45_mask],
                marker="o", color="#1f77b4", s=22, alpha=0.65, label="Weibull",
                zorder=3)
    ax2.scatter(vmax_sorted[~cat45_mask], tl_theo[~cat45_mask],
                marker="s", color="#d62728", s=22, alpha=0.65, label="TruncLognorm",
                facecolors="none", linewidths=0.9, zorder=3)
    # Cat 4/5 tail — larger, filled, annotated
    if n_cat45 > 0:
        ax2.scatter(vmax_sorted[cat45_mask], wb_theo[cat45_mask],
                    marker="o", color="#1f77b4", s=60, zorder=5,
                    label=f"Weibull tail (Cat 4/5, n={n_cat45})")
        ax2.scatter(vmax_sorted[cat45_mask], tl_theo[cat45_mask],
                    marker="s", color="#d62728", s=60, zorder=5,
                    label=f"TruncLognorm tail (Cat 4/5)")
        # Annotate with empirical Vmax value
        n_ann = min(n_cat45, 8)
        for emp_v, wb_t, tl_t in zip(
            vmax_sorted[cat45_mask][-n_ann:],
            wb_theo[cat45_mask][-n_ann:],
            tl_theo[cat45_mask][-n_ann:],
        ):
            y_ann = max(wb_t, tl_t)
            ax2.annotate(f"{emp_v:.0f}", xy=(emp_v, y_ann),
                         xytext=(3, 4), textcoords="offset points",
                         fontsize=7, color="#444444")

    # 45-degree reference line
    all_vals = np.concatenate([vmax_sorted, wb_theo, tl_theo])
    lo = float(all_vals.min()) - 2.0
    hi = float(all_vals.max()) + 5.0
    ax2.plot([lo, hi], [lo, hi], "k-", linewidth=0.8, alpha=0.4, label="45 deg (perfect fit)")

    # Cat 4 threshold lines
    ax2.axvline(113, color="#ff7f0e", linewidth=0.7, linestyle="--", alpha=0.6)
    ax2.axhline(113, color="#ff7f0e", linewidth=0.7, linestyle="--", alpha=0.6)
    ax2.text(65, 115, "Cat 4+", fontsize=7, color="#ff7f0e")

    ax2.set_xlabel("Empirical Vmax (kt)", fontsize=10)
    ax2.set_ylabel("Theoretical Vmax (kt)", fontsize=10)
    ax2.set_title("QQ-Plot: Empirical vs Theoretical\n(Cat 4/5 tail annotated in kt)", fontsize=11)
    ax2.legend(fontsize=7, ncol=2, loc="upper left")
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)

    sel_label = "Shifted Weibull" if selected == "weibull_min" else "Truncated Lognormal"
    fig.suptitle(
        f"FL HU Landfall Intensity Calibration  |  HURDAT2 1851-{END}"
        f"  |  Selected: {sel_label}",
        fontsize=12, y=1.01,
    )
    fig.tight_layout(pad=2.0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sensitivity print helper
# ---------------------------------------------------------------------------

def _print_sensitivity(
    label: str,
    wb_c: float, wb_scale: float, wb_aic: float,
    tl_mu: float, tl_sigma: float, tl_aic: float,
    n: int,
) -> None:
    p_cat4_wb = _weibull_sf(113.0, wb_c, wb_scale)
    p_cat4_tl = _trunclognorm_sf(113.0, tl_mu, tl_sigma)
    p_cat5_wb = _weibull_sf(137.0, wb_c, wb_scale)
    p_cat5_tl = _trunclognorm_sf(137.0, tl_mu, tl_sigma)
    print(f"  {label} (n={n}):")
    print(f"    Weibull       c={wb_c:.4f}  scale={wb_scale:.4f} kt  "
          f"AIC={wb_aic:.2f}  P(Cat4+)={p_cat4_wb:.3f}  P(Cat5)={p_cat5_wb:.3f}")
    print(f"    TruncLognorm  mu={tl_mu:.4f}  sigma={tl_sigma:.4f}      "
          f"AIC={tl_aic:.2f}  P(Cat4+)={p_cat4_tl:.3f}  P(Cat5)={p_cat5_tl:.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Step 1: load and filter ----------------------------------------
    df_all = pd.read_csv(_FL_CSV)
    hu_all = df_all.loc[df_all["status"] == "HU"].copy()

    full_vmax = hu_all["vmax_kt"].values.astype(float)
    sat_vmax  = hu_all.loc[hu_all["year"] >= SAT_START, "vmax_kt"].values.astype(float)

    n_full = len(full_vmax)
    n_sat  = len(sat_vmax)

    print("=== FL HU Landfall Intensity Calibration ===\n")
    print(f"Full record  ({FULL_START}-{END}): n = {n_full} HU landfalls")
    print(f"Satellite era ({SAT_START}-{END}): n = {n_sat} HU landfalls")

    # Guard: shift any observation exactly at the lower bound (rare but possible
    # for storms interpolated to exactly 64.0 kt at the boundary crossing).
    n_at_lb = int((full_vmax <= HU_LB).sum())
    if n_at_lb > 0:
        print(f"\n  Note: {n_at_lb} observation(s) at vmax_kt <= {HU_LB} kt "
              f"(HU lower bound) — shifted to {HU_LB + 0.01:.2f} kt for MLE.")
        full_vmax = np.where(full_vmax <= HU_LB, HU_LB + 0.01, full_vmax)
        sat_vmax  = np.where(sat_vmax  <= HU_LB, HU_LB + 0.01, sat_vmax)

    # ---- Step 2: fit both distributions on FULL record ------------------
    print("\n--- Full record MLE fits ---")
    wb_c_f,  wb_s_f,  wb_ll_f,  wb_aic_f  = _fit_weibull(full_vmax)
    tl_mu_f, tl_sg_f, tl_ll_f, tl_aic_f  = _fit_trunclognorm(full_vmax)
    print(f"  Weibull       c={wb_c_f:.4f}  scale={wb_s_f:.4f} kt  "
          f"LL={wb_ll_f:.2f}  AIC={wb_aic_f:.2f}")
    print(f"  TruncLognorm  mu={tl_mu_f:.4f}  sigma={tl_sg_f:.4f}      "
          f"LL={tl_ll_f:.2f}  AIC={tl_aic_f:.2f}")

    # ---- Step 3: fit both on satellite era + sensitivity check ----------
    print("\n--- Satellite era MLE fits ---")
    wb_c_s,  wb_s_s,  _,       wb_aic_s  = _fit_weibull(sat_vmax)
    tl_mu_s, tl_sg_s, _,       tl_aic_s  = _fit_trunclognorm(sat_vmax)
    print(f"  Weibull       c={wb_c_s:.4f}  scale={wb_s_s:.4f} kt  "
          f"AIC={wb_aic_s:.2f}")
    print(f"  TruncLognorm  mu={tl_mu_s:.4f}  sigma={tl_sg_s:.4f}      "
          f"AIC={tl_aic_s:.2f}")

    print("\n--- Sensitivity check: full record vs satellite era ---")
    _print_sensitivity("Full record  ", wb_c_f, wb_s_f, wb_aic_f, tl_mu_f, tl_sg_f, tl_aic_f, n_full)
    _print_sensitivity("Satellite era", wb_c_s, wb_s_s, wb_aic_s, tl_mu_s, tl_sg_s, tl_aic_s, n_sat)

    # Implied P(Cat4+) comparison across record choices
    p_cat4_wb_f = _weibull_sf(113.0, wb_c_f, wb_s_f)
    p_cat4_wb_s = _weibull_sf(113.0, wb_c_s, wb_s_s)
    p_cat4_tl_f = _trunclognorm_sf(113.0, tl_mu_f, tl_sg_f)
    p_cat4_tl_s = _trunclognorm_sf(113.0, tl_mu_s, tl_sg_s)

    def _rel_diff(a, b):
        mid = 0.5 * (a + b)
        return abs(a - b) / mid if mid > 1e-9 else 0.0

    wb_rel  = _rel_diff(p_cat4_wb_f, p_cat4_wb_s)
    tl_rel  = _rel_diff(p_cat4_tl_f, p_cat4_tl_s)
    material_threshold = 0.15  # 15% relative difference
    wb_mat  = wb_rel  > material_threshold
    tl_mat  = tl_rel  > material_threshold

    print(f"\n  P(Cat4+) relative diff - Weibull      : {wb_rel * 100:.1f}%"
          f"  {'[MATERIAL]' if wb_mat else '[within threshold]'}")
    print(f"  P(Cat4+) relative diff - TruncLognorm : {tl_rel * 100:.1f}%"
          f"  {'[MATERIAL]' if tl_mat else '[within threshold]'}")

    if wb_mat or tl_mat:
        print(
            "\n  WARNING: >= 1 distribution shows material sensitivity to record choice "
            "(>15% relative diff in P(Cat4+)). Full-record fit still written to config "
            "per plan; review satellite-era parameters above before finalizing."
        )
    else:
        print(
            "\n  Full-record and satellite-era fits agree within threshold "
            "(both P(Cat4+) relative diff < 15%). Full-record fit used for config."
        )

    # ---- Step 4: Model selection (full-record fits) ----------------------
    print("\n--- Model selection (full record) ---")

    wb_mad = _tail_mad(np.sort(full_vmax), lambda p: _weibull_ppf(p, wb_c_f, wb_s_f))
    tl_mad = _tail_mad(np.sort(full_vmax), lambda p: _trunclognorm_ppf(p, tl_mu_f, tl_sg_f))

    print(f"  Upper-tail QQ MAD (p >= 0.75):")
    print(f"    Weibull      : {wb_mad:.3f} kt")
    print(f"    TruncLognorm : {tl_mad:.3f} kt")
    print(f"  AIC:")
    print(f"    Weibull      : {wb_aic_f:.2f}")
    print(f"    TruncLognorm : {tl_aic_f:.2f}")

    # Primary: lower tail MAD; tiebreaker: lower AIC
    if abs(wb_mad - tl_mad) / max(wb_mad, tl_mad) < 0.05:
        # within 5% relative — use AIC as tiebreaker
        if wb_aic_f <= tl_aic_f:
            selected  = "weibull_min"
            reason    = f"tail MADs within 5% relative ({wb_mad:.2f} vs {tl_mad:.2f} kt); AIC tiebreaker favours Weibull"
        else:
            selected  = "trunclognorm"
            reason    = f"tail MADs within 5% relative ({wb_mad:.2f} vs {tl_mad:.2f} kt); AIC tiebreaker favours TruncLognorm"
    elif wb_mad < tl_mad:
        selected  = "weibull_min"
        reason    = f"lower upper-tail QQ MAD ({wb_mad:.2f} kt vs {tl_mad:.2f} kt)"
    else:
        selected  = "trunclognorm"
        reason    = f"lower upper-tail QQ MAD ({tl_mad:.2f} kt vs {wb_mad:.2f} kt)"

    sel_label = "Shifted Weibull"     if selected == "weibull_min" else "Truncated Lognormal"
    alt_label = "Truncated Lognormal" if selected == "weibull_min" else "Shifted Weibull"
    daic      = (wb_aic_f - tl_aic_f) if selected == "weibull_min" else (tl_aic_f - wb_aic_f)
    sel_mad   = wb_mad  if selected == "weibull_min" else tl_mad
    alt_mad   = tl_mad  if selected == "weibull_min" else wb_mad

    print(f"\n  Selected : {sel_label}")
    print(f"  Reason   : {reason}")
    print(f"  dAIC     : {daic:+.2f} (selected vs alternative)")

    # ---- Step 5: Category frequency table --------------------------------
    emp_probs = _empirical_cat_probs(full_vmax)
    if selected == "weibull_min":
        fit_probs = _category_probs_weibull(wb_c_f, wb_s_f)
    else:
        fit_probs = _category_probs_trunclognorm(tl_mu_f, tl_sg_f)

    print(f"\n--- Category frequency comparison (full record, n={n_full}) ---")
    print(f"  {'Category':<22}  {'Empirical':>10}  {'Fitted':>10}  {'Diff':>8}")
    print("  " + "-" * 56)
    for lbl, emp, fit in zip(_SS_LABELS, emp_probs, fit_probs):
        diff = fit - emp
        print(f"  {lbl:<22}  {emp:>10.3f}  {fit:>10.3f}  {diff:>+8.3f}")
    print(f"  {'Total':22}  {emp_probs.sum():>10.3f}  {fit_probs.sum():>10.3f}")

    # ---- Step 6: Figure --------------------------------------------------
    _make_figure(full_vmax, wb_c_f, wb_s_f, tl_mu_f, tl_sg_f, selected, _FIG)
    print(f"\nFigure saved -> {_FIG}")
    print("  Review QQ upper tail before finalizing config.")

    # ---- Step 7: Write to config -----------------------------------------
    source = _SOURCE_TMPL.format(
        dist_label=sel_label,
        alt_label=alt_label,
        start=FULL_START,
        end=END,
        n=n_full,
        sel_mad=sel_mad,
        alt_mad=alt_mad,
        daic=daic,
    )
    print(f"\n--- Config source string ---")
    print(f"  {source}")

    if selected == "weibull_min":
        params = {"loc": HU_LB, "shape_c": wb_c_f, "scale": wb_s_f}
    else:
        params = {"loc": HU_LB, "mu_log": tl_mu_f, "sigma_log": tl_sg_f}

    _write_intensity_config(_MODEL_CFG, selected, params, source)
    print(f"\nintensity: block written to config/model_v3.yaml")
    if selected == "weibull_min":
        print(f"  distribution = weibull_min  |  loc=64.0  c={wb_c_f:.4f}  scale={wb_s_f:.4f} kt")
    else:
        print(f"  distribution = trunclognorm  |  loc=64.0  mu={tl_mu_f:.4f}  sigma={tl_sg_f:.4f}")
