"""
Step 3.1 diagnostic: rho sweep for secondary uncertainty.

Runs 5 configurations (100k years each) in-process by monkeypatching module-level
constants in model.loss. Writes outputs to results/waterfall/ (never production
results/summary_metrics.csv). Saves star figure to outputs/secondary_uncertainty_ep.png.

Configurations:
  0. Baseline (deterministic, uncertainty=off)
  1. rho=0.0  (independent noise, uncertainty=on)
  2. rho=0.3  (low correlation)
  3. rho=0.7  (high correlation)
  4. rho=1.0  (perfect common shock)

Predicted signs (stated before running):
  - AAL   : approximately flat across all rho (Beta is mean-preserving; LLN further flattens)
  - tail  : OEP/AEP tail (esp. 500/1000yr) rises monotonically with rho
  - rho=0 : tail close to deterministic baseline (independent noise washes out at n=1000 locs)
  - rho=1 : fattest tail
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import model.loss as _loss
from model.ep_utils import oep_pml, ep_curve

_RESULTS_DIR = os.path.join(_ROOT, "results", "waterfall")
_OUTPUTS_DIR = os.path.join(_ROOT, "outputs")
os.makedirs(_RESULTS_DIR, exist_ok=True)
os.makedirs(_OUTPUTS_DIR, exist_ok=True)

N_YEARS = 100_000
SEED    = 42
RETURN_PERIODS = [100, 250, 500, 1_000]


def _run_config(uncertainty, rho):
    """Monkeypatch module constants and run the full simulation."""
    _loss._DAMAGE_UNCERTAINTY = uncertainty
    _loss._DAMAGE_RHO         = float(rho)
    res = _loss.run_simulation(n_years=N_YEARS, seed=SEED)
    return res[0], res[1]   # events_df, annual_df


def _pml_row(annual_df, label, uncertainty, rho):
    oep_arr = annual_df["max_event_gross"].to_numpy()
    aep_arr = annual_df["aggregate_gross"].to_numpy()
    aal     = float(aep_arr.mean())
    row = {"Config": label, "uncertainty": uncertainty, "rho": rho,
           "AAL (M)": aal / 1e6}
    for rp in RETURN_PERIODS:
        row[f"OEP-{rp} (M)"] = oep_pml(oep_arr, rp, N_YEARS) / 1e6
    for rp in [100, 250]:
        row[f"AEP-{rp} (M)"] = oep_pml(aep_arr, rp, N_YEARS) / 1e6
    return row, oep_arr, aep_arr


def main():
    print()
    print("=" * 70)
    print("Step 3.1 — secondary uncertainty rho sweep")
    print(f"N_YEARS = {N_YEARS:,} | SEED = {SEED} | CV = {_loss._DAMAGE_CV:.2f}")
    print()
    print("PREDICTED SIGNS (stated before running):")
    print("  AAL    : approximately flat across all rho (mean-preserving)")
    print("  OEP tail: rises monotonically with rho")
    print("  rho=0  : tail close to deterministic baseline")
    print("  rho=1  : fattest tail")
    print("=" * 70)

    configs = [
        ("Baseline (det.)", "off", None),
        ("rho=0.0",         "on",  0.0),
        ("rho=0.3",         "on",  0.3),
        ("rho=0.7",         "on",  0.7),
        ("rho=1.0",         "on",  1.0),
    ]

    rows   = []
    oep_series = {}  # label -> (sorted_losses, N)

    for label, uncertainty, rho in configs:
        rho_display = "—" if rho is None else f"{rho:.1f}"
        print(f"\nRunning: {label}  (uncertainty={uncertainty}, rho={rho_display})")
        _, annual_df = _run_config(uncertainty, rho if rho is not None else _loss._DAMAGE_RHO)
        row, oep_arr, _ = _pml_row(annual_df, label, uncertainty, rho)
        rows.append(row)
        sorted_oep, _ = ep_curve(oep_arr, N_YEARS)
        oep_series[label] = sorted_oep

    # Restore production defaults
    _loss._DAMAGE_UNCERTAINTY = "off"
    _loss._DAMAGE_RHO         = 0.5

    # ---- Print table -------------------------------------------------------
    df = pd.DataFrame(rows)
    print()
    print("=" * 70)
    print("RHO SWEEP TABLE")
    print("=" * 70)
    float_cols = [c for c in df.columns if "(M)" in c]
    df[float_cols] = df[float_cols].round(2)
    with pd.option_context("display.float_format", "{:.2f}".format,
                           "display.max_columns", 20,
                           "display.width", 120):
        print(df.to_string(index=False))

    # Write to results/waterfall/
    out_csv = os.path.join(_RESULTS_DIR, "sensitivity_secondary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nTable saved: {out_csv}")

    # ---- Check predicted signs --------------------------------------------
    print()
    print("PREDICTED-SIGN CHECK:")
    aal_vals = [row["AAL (M)"] for row in rows]
    aal_max_delta = max(aal_vals) - min(aal_vals)
    print(f"  AAL range: {min(aal_vals):.2f}–{max(aal_vals):.2f} M  "
          f"(delta = {aal_max_delta:.2f} M, expect < ~0.3 M)")

    rp = 1000
    oep_vals = [row[f"OEP-{rp} (M)"] for row in rows]
    monotone = all(oep_vals[i] <= oep_vals[i+1] for i in range(len(oep_vals)-1))
    print(f"  OEP-{rp} across configs: {' < '.join(f'{v:.1f}' for v in oep_vals)}")
    print(f"  Monotone rising: {'YES' if monotone else 'NO — CHECK MECHANISM'}")

    if aal_max_delta > 1.0:
        print("  WARNING: AAL varies > 1M across rho — unexpected (Beta is mean-preserving).")
    if not monotone:
        print("  STOP: tail not monotone in rho. Investigate before committing.")
    else:
        print("  Predicted signs CONFIRMED.")

    # ---- Star figure -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#222222", "#4393C3", "#74ADD1", "#F4A582", "#D6604D"]
    lws    = [2.5, 1.4, 1.4, 1.4, 1.4]
    for i, (label, uncertainty, rho) in enumerate(configs):
        sorted_oep = oep_series[label]
        n_ev       = len(sorted_oep[sorted_oep > 0])
        rps        = np.arange(1, n_ev + 1)[::-1]
        rp_vals    = N_YEARS / rps
        mask       = (rp_vals >= 10) & (sorted_oep[sorted_oep > 0][::-1][::-1] is not None)
        # Filter to OEP > 0 events
        oep_pos  = sorted_oep[sorted_oep > 0]
        if len(oep_pos) == 0:
            continue
        rp_pos   = N_YEARS / np.arange(1, len(oep_pos) + 1)
        mask2    = rp_pos >= 10
        ax.plot(rp_pos[mask2], oep_pos[mask2] / 1e6,
                color=colors[i], lw=lws[i],
                ls="-" if uncertainty == "off" else "--",
                label=label, zorder=3 - i * 0.1)

    # Shaded band rho=0 to rho=1
    rho0_oep = oep_series["rho=0.0"]
    rho1_oep = oep_series["rho=1.0"]
    min_len  = min(len(rho0_oep[rho0_oep > 0]), len(rho1_oep[rho1_oep > 0]))
    rp_band  = N_YEARS / np.arange(1, min_len + 1)
    mask_b   = rp_band >= 10
    ax.fill_between(rp_band[mask_b],
                    rho0_oep[rho0_oep > 0][:min_len][mask_b] / 1e6,
                    rho1_oep[rho1_oep > 0][:min_len][mask_b] / 1e6,
                    color="#74ADD1", alpha=0.18, label="ρ=0→1 band")

    ax.set_xscale("log")
    ax.set_xlim(10, 10_000)
    ax.set_xlabel("Return period (years)", fontsize=11)
    ax.set_ylabel("Gross OEP loss (USD M)", fontsize=11)
    ax.set_title("Step 3.1 — Secondary uncertainty: ρ sensitivity\n"
                 f"CV=0.40 (placeholder) | N={N_YEARS:,} years | seed={SEED}",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.text(0.98, 0.03,
            "uncertainty=off default\nAbsolute tail depends on heuristic vulnerability\nρ is a sensitivity knob (not calibrated)",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
            color="#555555")
    ax.grid(True, which="major", ls=":", alpha=0.5)
    fig.tight_layout()

    out_png = os.path.join(_OUTPUTS_DIR, "secondary_uncertainty_ep.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Star figure saved: {out_png}")


if __name__ == "__main__":
    main()
