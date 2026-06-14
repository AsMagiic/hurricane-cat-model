"""
Derive damage-state fragility parameters for the v3 vulnerability model.

Framework: 5 damage states DS0-DS4 (HAZUS Hurricane TM Table 5-44).
  Fragility:   P(DS >= k | gust) = Phi( ln(gust / theta_k) / beta )
  Consequence: loss_ratio = [0, 0.02, 0.10, 0.50, 1.00]  (HAZUS TM §8.1.4.3)
  Expected DR: E[DR | gust] = sum_{k=1}^{4} delta_lr_k * Phi( ln(g/theta_k) / beta )
               where delta_lr = lr_k - lr_{k-1} = [0.02, 0.08, 0.40, 0.50]

This summation-by-parts form of E[DR] is algebraically equivalent to
  sum_k P(DS=k|g) * lr_k  but evaluated with only 4 Phi() calls instead of 5.
It also guarantees monotone E[DR] in g by construction (each Phi term is
monotone increasing), so the monotonicity assertion is cheap insurance against
future edits that touch the formula rather than a test of correctness.

Parameter derivation — all design decisions are own-derived from public sources
(HAZUS Hurricane TM, FPHLM methodology, primary fragility literature).
HAZUS and FPHLM publish methodology but NOT calibrated parameters; Pinelli et al.
(2004) inputs are explicitly "hypothetical". Parameters below are triangulated:

  theta_DS3  := v2 logistic midpoint = wind at ~50% damage ratio ≈ median of
                the Extensive damage state.  Values: Mfg 110, WF 145, Mas 165,
                RC 185 mph (3-s gust).  Source: model_v3.yaml construction_params.

  theta_DS1  := 88.0 * (midpoint / 145.0).  Reference: 88 mph = median onset
                of roof-cover damage (>2% loss, DS1-dominant per Table 5-44) for
                typical Wood Frame; scaled across classes by midpoint ratio.
                Rationale: sits above model 65-mph threshold; below 110 mph where
                large-opening failures begin (DS2, Rosowsky/Ellingwood-style lit).

  beta       := grid search over [0.10, 0.25] step 0.01 (literature range;
                ~0.16 typical, Rosowsky & Ellingwood 2001).  For each beta,
                theta_DS2 and theta_DS4 are optimized (Nelder-Mead).  Best beta
                minimises fit MSE against v2 logistic E[DR] target.

  SEP(beta)  := exp(beta) — minimum multiplicative gap between consecutive
                thetas, equal to one log-sd of the fragility.  Guarantees each
                DS has distinct probability mass across the physically meaningful
                wind range.

  Feasibility: theta_DS3/theta_DS1 = 145/88 = 1.6477 for all classes.
               Need exp(2*beta) <= 1.6477, i.e. beta <= ln(1.6477)/2 = 0.2493.
               beta=0.25 → exp(0.5)=1.6487 > 1.6477 → INFEASIBLE (skipped).
               Grid effective range: [0.10, 0.24].  Edge warning fires at 0.10
               or 0.24 (the effective bounds), not nominal 0.10/0.25.

  theta_DS2, theta_DS4: Nelder-Mead minimising MSE(E[DR], logistic) over
    g = linspace(70, 220, 80).  Multistart with FIXED x0 set (determinism):
      {[0.2,0.2], [0.3,0.15], [0.1,0.3]}   (these are the initial [x0,x1] vectors)
    Parametrisation: theta2 = theta1*exp(|x0|), theta4 = theta3*exp(|x1|).
    Hard-reject (cost=1e3) when SEP constraints are violated.

Units: all wind speeds 3-s peak gust (mph) throughout this script.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr  # Phi — faster than norm.cdf for large arrays

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model_config import load_calibration_cfg, load_model_cfg

_ccfg    = load_calibration_cfg()
_FIG     = os.path.join(_ROOT, _ccfg.fragility_calibration.figure_path)
_CSV     = os.path.join(_ROOT, _ccfg.fragility_calibration.csv_path)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONSEQUENCE = np.array([0.0, 0.02, 0.10, 0.50, 1.00])   # DS0-DS4 mean loss ratio
_DELTA_LR    = np.diff(_CONSEQUENCE)                        # [0.02, 0.08, 0.40, 0.50]

_GUST_MIN    = 65.0   # model damage threshold (mph)
_FIT_GRID    = np.linspace(70.0, 220.0, 80)   # calibration domain (mph)

_BETA_MIN    = 0.10
_BETA_MAX    = 0.25
_BETA_STEP   = 0.01
_BETA_GRID   = np.round(np.arange(_BETA_MIN, _BETA_MAX + _BETA_STEP / 2, _BETA_STEP), 10)

# Fixed multistart initial points — same for every (class, beta) pair.
# Using these exact vectors is the determinism guarantee: no RNG involved.
_X0_SET = [
    np.array([0.2, 0.2]),
    np.array([0.3, 0.15]),
    np.array([0.1, 0.3]),
]

# theta_DS1 reference: 88 mph for Wood Frame (see module docstring)
_THETA1_REF_MPH      = 88.0
_THETA1_REF_MIDPOINT = 145.0   # WF midpoint used as scaling denominator

_CLASS_ORDER = ["Manufactured", "Wood Frame", "Masonry", "Reinforced Concrete"]


# ---------------------------------------------------------------------------
# Pure functions — no I/O, directly testable
# ---------------------------------------------------------------------------

def edr(gust_mph: "float | np.ndarray", thetas: np.ndarray, beta: float) -> np.ndarray:
    """
    Expected damage ratio at gust speed(s).

    E[DR|g] = sum_{k=1}^{4} delta_lr_k * Phi( ln(g / theta_k) / beta )

    Summation-by-parts form: algebraically identical to sum_k P(DS=k)*lr_k but
    with 4 Phi evaluations instead of 5, and monotone by construction since each
    Phi(ln(g/theta_k)/beta) is strictly increasing in g for positive theta_k.

    Parameters
    ----------
    gust_mph : array_like, 3-s peak gust (mph)
    thetas   : array of 4 floats, [theta1, theta2, theta3, theta4] (mph)
    beta     : float, lognormal dispersion (dimensionless)

    Returns
    -------
    edr_arr : np.ndarray, damage ratio in [0, 1]
    """
    g = np.atleast_1d(np.asarray(gust_mph, dtype=float))
    out = np.zeros(g.shape)
    for dlr, th in zip(_DELTA_LR, thetas):
        out += dlr * ndtr(np.log(g / th) / beta)
    return out


def logistic_dr(gust_mph: "float | np.ndarray", cap: float, midpoint: float, k: float) -> np.ndarray:
    """
    v2 damage ratio: dr(g) = cap / (1 + exp(-k*(g - midpoint))) for g >= 65, else 0.

    Parameters
    ----------
    gust_mph : array_like, 3-s peak gust (mph)
    cap      : float, maximum damage ratio (fraction)
    midpoint : float, 50%-damage wind speed (mph)
    k        : float, logistic slope (per mph)

    Returns
    -------
    dr : np.ndarray
    """
    g  = np.atleast_1d(np.asarray(gust_mph, dtype=float))
    dr = np.where(g >= _GUST_MIN, cap / (1.0 + np.exp(-k * (g - midpoint))), 0.0)
    return dr


def exceedance_probs(gust_mph: "float | np.ndarray", thetas: np.ndarray, beta: float) -> np.ndarray:
    """
    P(DS >= k | gust) for k = 1, 2, 3, 4.

    Returns array of shape (4, len(gust_mph)).

    Parameters
    ----------
    gust_mph : array_like, 3-s peak gust (mph)
    thetas   : array of 4 floats (mph)
    beta     : float

    Returns
    -------
    probs : np.ndarray, shape (4, n)
    """
    g = np.atleast_1d(np.asarray(gust_mph, dtype=float))
    return np.stack([ndtr(np.log(g / th) / beta) for th in thetas])


# ---------------------------------------------------------------------------
# Per-class calibration
# ---------------------------------------------------------------------------

def _feasible_betas(theta1: float, theta3: float) -> list[float]:
    """Return beta values from _BETA_GRID that satisfy theta3/theta1 >= SEP^2."""
    ratio = theta3 / theta1
    return [b for b in _BETA_GRID if np.exp(2.0 * b) <= ratio]


def _fit_thetas(
    cap: float,
    midpoint: float,
    k: float,
    beta: float,
    theta1: float,
    theta3: float,
) -> "tuple[float, float, float]":
    """
    Optimise theta2, theta4 for a fixed (beta, theta1, theta3) to minimise
    MSE(E[DR], logistic) over _FIT_GRID.

    Returns (theta2, theta4, rmse).
    """
    target = logistic_dr(_FIT_GRID, cap, midpoint, k)
    sep    = np.exp(beta)

    th2_lo = theta1 * sep
    th2_hi = theta3 / sep
    th4_lo = theta3 * sep

    def _cost(x: np.ndarray) -> float:
        th2 = theta1 * np.exp(abs(x[0]))
        th4 = theta3 * np.exp(abs(x[1]))
        # Hard-reject outside SEP constraints
        if not (th2_lo <= th2 <= th2_hi) or th4 < th4_lo:
            return 1e3
        pred = edr(_FIT_GRID, np.array([theta1, th2, theta3, th4]), beta)
        return float(np.mean((pred - target) ** 2))

    best_cost = np.inf
    best_x    = None
    for x0 in _X0_SET:
        res = minimize(_cost, x0, method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
        if res.fun < best_cost:
            best_cost = res.fun
            best_x    = res.x

    th2 = float(theta1 * np.exp(abs(best_x[0])))
    th4 = float(theta3 * np.exp(abs(best_x[1])))
    rmse = float(np.sqrt(best_cost))
    return th2, th4, rmse


def calibrate_class(
    class_name: str,
    params: dict,
) -> dict:
    """
    Full calibration for one construction class.

    Runs the beta grid search, fits theta2/theta4 at each feasible beta,
    returns the result with the lowest RMSE.

    Parameters
    ----------
    class_name : str
    params     : dict with keys 'cap', 'midpoint', 'k' (from model_v3.yaml)

    Returns
    -------
    dict with keys: class, beta, theta1, theta2, theta3, theta4, rmse
    """
    cap      = float(params["cap"])
    midpoint = float(params["midpoint"])
    k        = float(params["k"])

    theta1 = _THETA1_REF_MPH * (midpoint / _THETA1_REF_MIDPOINT)
    theta3 = midpoint

    feasible = _feasible_betas(theta1, theta3)
    if not feasible:
        raise RuntimeError(
            f"{class_name}: no feasible beta in grid. "
            f"theta3/theta1 = {theta3/theta1:.4f}, need > exp(2*{_BETA_MIN})={np.exp(2*_BETA_MIN):.4f}"
        )

    best = None
    for beta in feasible:
        th2, th4, rmse = _fit_thetas(cap, midpoint, k, beta, theta1, theta3)
        if best is None or rmse < best["rmse"]:
            best = dict(
                cls=class_name, beta=beta,
                theta1=theta1, theta2=th2, theta3=theta3, theta4=th4,
                rmse=rmse,
            )

    return best


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _check_result(result: dict, effective_beta_min: float, effective_beta_max: float) -> None:
    """Assert all DoD sanity constraints for one class."""
    cls    = result["cls"]
    thetas = np.array([result["theta1"], result["theta2"], result["theta3"], result["theta4"]])
    beta   = result["beta"]
    sep    = np.exp(beta)

    # Thetas strictly increasing
    assert np.all(np.diff(thetas) > 0), f"{cls}: thetas not strictly increasing: {thetas}"

    # SEP respected between consecutive pairs
    for i in range(3):
        ratio = thetas[i + 1] / thetas[i]
        assert ratio >= sep - 1e-8, (
            f"{cls}: SEP violated between theta{i+1} and theta{i+2}: "
            f"ratio={ratio:.4f} < SEP={sep:.4f}"
        )

    # Beta strictly inside effective grid bounds (warn if pinned at edge)
    if beta <= effective_beta_min + 1e-9:
        print(f"  [WARN] {cls}: beta={beta:.2f} pinned at effective lower edge "
              f"({effective_beta_min:.2f}). Review fragility width.")
    if beta >= effective_beta_max - 1e-9:
        print(f"  [WARN] {cls}: beta={beta:.2f} pinned at effective upper edge "
              f"({effective_beta_max:.2f}). Review feasibility constraint.")

    # E[DR] monotone non-decreasing over fit grid (cheap insurance, not tautology)
    dr_vals = edr(_FIT_GRID, thetas, beta)
    assert np.all(np.diff(dr_vals) >= -1e-12), (
        f"{cls}: E[DR] not monotone non-decreasing. Min diff: {np.diff(dr_vals).min():.2e}"
    )


def check_cross_class_hierarchy(results: list[dict]) -> None:
    """
    Assert theta_k(Mfg) < theta_k(WF) < theta_k(Mas) < theta_k(RC) for k=1..4.
    """
    for k, key in enumerate(["theta1", "theta2", "theta3", "theta4"], 1):
        vals = [r[key] for r in results]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1], (
                f"Cross-class hierarchy violated for theta{k}: "
                f"{results[i]['cls']} ({vals[i]:.2f}) >= "
                f"{results[i+1]['cls']} ({vals[i+1]:.2f})"
            )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

_DS_COLORS  = ["#e41a1c", "#ff7f00", "#984ea3", "#377eb8"]  # DS>=1..4
_DS_LABELS  = ["P(DS≥1)", "P(DS≥2)", "P(DS≥3)", "P(DS≥4)"]

def _make_figure(results: list[dict], cparams: dict, path: str) -> None:
    """
    2x2 grid of panels, one per construction class.

    Each panel:
    - x: 3-s gust (mph), range 60-230
    - y: damage ratio / exceedance probability [0, 1]
    - Black dashed (thick): v2 logistic E[DR]
    - Blue solid (thick):   damage-state E[DR] (calibrated)
    - 4 thin dashed lines:  P(DS>=k|g) for k=1..4
    - Vertical gray line:   65 mph gust threshold
    - Text box: beta, RMSE
    """
    g_plot = np.linspace(60.0, 230.0, 400)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes_flat = axes.flatten()

    for ax, result in zip(axes_flat, results):
        cls    = result["cls"]
        p      = cparams[cls]
        thetas = np.array([result["theta1"], result["theta2"],
                           result["theta3"], result["theta4"]])
        beta   = result["beta"]
        rmse   = result["rmse"]

        # v2 logistic
        dr_log = logistic_dr(g_plot, p["cap"], p["midpoint"], p["k"])
        ax.plot(g_plot, dr_log, "k--", linewidth=2.0, label="v2 logistic", zorder=4)

        # DS-scheme E[DR]
        dr_ds  = edr(g_plot, thetas, beta)
        ax.plot(g_plot, dr_ds, color="#1f78b4", linewidth=2.0,
                label="E[DR] DS scheme", zorder=4)

        # Fragility curves P(DS>=k)
        probs = exceedance_probs(g_plot, thetas, beta)
        for i, (prob_row, color, lbl) in enumerate(zip(probs, _DS_COLORS, _DS_LABELS)):
            ax.plot(g_plot, prob_row, color=color, linewidth=1.0,
                    linestyle="--", alpha=0.75, label=lbl, zorder=3)

        # Gust threshold
        ax.axvline(x=_GUST_MIN, color="gray", linewidth=0.8, linestyle=":",
                   alpha=0.6, zorder=2)

        # Text box
        info = (f"β = {beta:.2f}\nRMSE = {rmse:.4f}\n"
                f"θ = [{thetas[0]:.0f}, {thetas[1]:.0f}, {thetas[2]:.0f}, {thetas[3]:.0f}]")
        ax.text(0.97, 0.05, info, transform=ax.transAxes,
                fontsize=8, verticalalignment="bottom", horizontalalignment="right",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85})

        ax.set_title(cls, fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(60, 230)
        ax.grid(True, linestyle=":", alpha=0.4)

    # Shared labels
    for ax in axes[1]:
        ax.set_xlabel("3-s peak gust (mph)", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Damage ratio / P(DS≥k)", fontsize=10)

    # Single legend from first panel
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "v3 Damage-State Fragility Calibration — vs. v2 Logistic Baseline",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved -> {path}")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def _write_csv(results: list[dict], path: str) -> None:
    rows = []
    for r in results:
        rows.append({
            "class":   r["cls"],
            "theta1":  round(r["theta1"],  4),
            "theta2":  round(r["theta2"],  4),
            "theta3":  round(r["theta3"],  4),
            "theta4":  round(r["theta4"],  4),
            "beta":    round(r["beta"],    4),
            "rmse":    round(r["rmse"],    6),
        })
    df = pd.DataFrame(rows, columns=["class", "theta1", "theta2", "theta3", "theta4", "beta", "rmse"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  CSV saved -> {path}")


# ---------------------------------------------------------------------------
# YAML block printer
# ---------------------------------------------------------------------------

def _print_yaml(results: list[dict]) -> None:
    """Print YAML-ready block for human review — do NOT paste into config yet."""
    print()
    print("# ---- YAML-ready block (for human review; Task 2b will add to config) ----")
    print("damage_states:")
    print(f"  consequence: {[float(x) for x in _CONSEQUENCE]}")
    for r in results:
        cls    = r["cls"]
        thetas = [round(r["theta1"], 2), round(r["theta2"], 2),
                  round(r["theta3"], 2), round(r["theta4"], 2)]
        beta   = round(r["beta"], 4)
        src    = (
            f"Own-derived: theta3=v2-logistic midpoint; theta1=88*(midpoint/145);"
            f" theta2/theta4=Nelder-Mead fit to v2 logistic; beta=grid-search MSE;"
            f" consequence from HAZUS TM §8.1.4.3. See calibration/fragility_thetas.py."
        )
        print(f"  {cls}:")
        print(f"    thetas: {thetas}    # mph gust, DS1-DS4")
        print(f"    beta:   {beta}       # lognormal dispersion")
        print(f"    source: \"{src}\"")
    print("# -------------------------------------------------------------------------")
    print()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_calibration() -> list[dict]:
    """
    Load logistic params -> calibrate each class -> sanity checks -> outputs.

    Returns list of result dicts (one per class, in _CLASS_ORDER).
    """
    mcfg    = load_model_cfg()
    cparams = mcfg.vulnerability.construction_params  # dict-of-dicts

    print("=== Fragility Calibration (damage-state framework) ===\n")
    print(f"  Classes:      {_CLASS_ORDER}")
    print(f"  Consequence:  {list(_CONSEQUENCE)}")
    print(f"  Beta grid:    [{_BETA_MIN}, {_BETA_MAX}] step {_BETA_STEP}")
    print(f"  Fit domain:   linspace(70, 220, 80) mph")
    print(f"  Multistarts:  {len(_X0_SET)} fixed x0 vectors")
    print()

    results = []
    for cls in _CLASS_ORDER:
        p = cparams[cls]
        print(f"  [{cls}]  midpoint={p['midpoint']:.0f}  cap={p['cap']:.2f}  k={p['k']:.3f}")
        r = calibrate_class(cls, p)
        thetas = [r["theta1"], r["theta2"], r["theta3"], r["theta4"]]
        print(f"    beta={r['beta']:.2f}  thetas={[round(t,1) for t in thetas]}  RMSE={r['rmse']:.4f}")
        results.append(r)

    # --- Effective beta bounds (after feasibility filtering) -----------------
    # All classes share the same theta3/theta1 = 145/88 ratio (by construction),
    # so feasible betas are identical across classes.
    ref_th1 = _THETA1_REF_MPH
    ref_th3 = _THETA1_REF_MIDPOINT  # WF class: theta3 == midpoint == 145
    feas = _feasible_betas(ref_th1, ref_th3)
    eff_min = min(feas)
    eff_max = max(feas)
    print(f"\n  Effective beta range after feasibility: [{eff_min:.2f}, {eff_max:.2f}]")
    print(f"  (beta={_BETA_MAX:.2f} excluded: exp(2*{_BETA_MAX})={np.exp(2*_BETA_MAX):.4f}"
          f" > theta3/theta1={ref_th3/ref_th1:.4f})")

    # --- Sanity checks -------------------------------------------------------
    print("\n  Running sanity checks...")
    for r in results:
        _check_result(r, eff_min, eff_max)
    check_cross_class_hierarchy(results)
    print("  All sanity checks passed.")

    # --- E[DR] at gust threshold ---------------------------------------------
    print("\n  E[DR] at gust_threshold = 65 mph (no hard cutoff in DS scheme):")
    for r in results:
        thetas = np.array([r["theta1"], r["theta2"], r["theta3"], r["theta4"]])
        dr_at_65 = float(edr(np.array([65.0]), thetas, r["beta"])[0])
        print(f"    {r['cls']:<22}  E[DR]@65 = {dr_at_65:.6f}")

    # --- Outputs -------------------------------------------------------------
    print()
    _write_csv(results, _CSV)
    _make_figure(results, cparams, _FIG)
    _print_yaml(results)

    # --- Summary table -------------------------------------------------------
    print("Summary table:")
    hdr = f"  {'Class':<22} {'beta':>5} {'theta1':>8} {'theta2':>8} {'theta3':>8} {'theta4':>8} {'RMSE':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        print(f"  {r['cls']:<22} {r['beta']:>5.2f} "
              f"{r['theta1']:>8.1f} {r['theta2']:>8.1f} "
              f"{r['theta3']:>8.1f} {r['theta4']:>8.1f} "
              f"{r['rmse']:>8.4f}")

    return results


if __name__ == "__main__":
    run_calibration()
