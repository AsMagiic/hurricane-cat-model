"""
Portfolio metrics summary for the Florida hurricane cat model (Step 6).

Reads Step 4 and Step 5 outputs -- no new simulation.  All PML calculations
go through ep_utils (single source of truth).

Four EP views
-------------
  AEP gross   Annual aggregate loss (net of per-policy deductible only)
  AEP net     Annual aggregate loss (net of deductible + XoL reinsurance)
  OEP gross   Annual max per-occurrence loss (gross)
  OEP net     Annual max per-occurrence loss (net of XoL)

Net series reconstruction (from events_net.csv)
------------------------------------------------
  aggregate_net[yr]  = aggregate_gross[yr]
                       - sum(recovery_total for all events in yr)
  max_event_net[yr]  = max(portfolio_net for all events in yr)
                       [0 for years with no events]

Outputs
-------
  results/summary_metrics.csv   -- tidy CSV with all metrics
  outputs/ep_master.png         -- 2-panel AEP+OEP gross vs net plot
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from model.ep_utils import oep_pml, ep_curve

RESULTS_DIR = os.path.join(_ROOT, "results")
OUT_DIR     = os.path.join(_ROOT, "outputs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

RETURN_PERIODS = [100, 250]

# Plot colours consistent with the rest of the model outputs
_C_GROSS = "#1f77b4"   # blue
_C_NET   = "#d62728"   # red
_C_FILL  = "#2ca02c"   # green (recovery band)


# ---------------------------------------------------------------------------
# Data loading and net series reconstruction
# ---------------------------------------------------------------------------
def _load(results_dir):
    ann_path = os.path.join(results_dir, "annual_losses.csv")
    evn_path = os.path.join(results_dir, "events_net.csv")

    for p in (ann_path, evn_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found -- run model/loss.py then model/reinsurance.py first."
            )

    annual_df  = pd.read_csv(ann_path)
    events_net = pd.read_csv(evn_path)
    N_YEARS    = len(annual_df)

    if len(events_net) > 0:
        yr_rec = events_net.groupby("year")["recovery_total"].sum()
        yr_net = events_net.groupby("year")["portfolio_net"].max()
    else:
        yr_rec = pd.Series(dtype=float)
        yr_net = pd.Series(dtype=float)

    annual_df["annual_recovery"] = annual_df["year"].map(yr_rec).fillna(0.0)
    annual_df["aggregate_net"]   = (
        annual_df["aggregate_gross"] - annual_df["annual_recovery"]
    )
    annual_df["max_event_net"]   = annual_df["year"].map(yr_net).fillna(0.0)

    return annual_df, N_YEARS


# ---------------------------------------------------------------------------
# Compute all metrics
# ---------------------------------------------------------------------------
def _compute_metrics(annual_df, N_YEARS):
    series = {
        "aep_g": annual_df["aggregate_gross"].to_numpy(),
        "aep_n": annual_df["aggregate_net"].to_numpy(),
        "oep_g": annual_df["max_event_gross"].to_numpy(),
        "oep_n": annual_df["max_event_net"].to_numpy(),
    }

    aal = {k: float(v.mean()) for k, v in series.items()}

    pml = {}
    for rp in RETURN_PERIODS:
        for k, arr in series.items():
            pml[(rp, k)] = oep_pml(arr, rp, N_YEARS)

    return series, aal, pml


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate(aal, pml):
    print()
    print("=" * 64)
    print("VALIDATION")
    print("=" * 64)

    # 1. AEP >= OEP (aggregate >= max-occurrence, always true by definition)
    for rp in RETURN_PERIODS:
        assert pml[(rp, "aep_g")] >= pml[(rp, "oep_g")] - 1e-4, \
            f"AEP gross < OEP gross at 1-in-{rp}"
        assert pml[(rp, "aep_n")] >= pml[(rp, "oep_n")] - 1e-4, \
            f"AEP net < OEP net at 1-in-{rp}"
    print("[OK] AEP >= OEP at 1-in-100 and 1-in-250 (gross and net)")

    # 2. Net <= Gross in every metric
    for rp in RETURN_PERIODS:
        assert pml[(rp, "aep_n")] <= pml[(rp, "aep_g")] + 1e-4, \
            f"AEP net > AEP gross at 1-in-{rp}"
        assert pml[(rp, "oep_n")] <= pml[(rp, "oep_g")] + 1e-4, \
            f"OEP net > OEP gross at 1-in-{rp}"
    assert aal["aep_n"] <= aal["aep_g"] + 1e-4, "AAL AEP net > AEP gross"
    assert aal["oep_n"] <= aal["oep_g"] + 1e-4, "AAL OEP net > OEP gross"
    print("[OK] net <= gross for all metrics (AAL, PML 1-in-100, PML 1-in-250)")

    # 3. OEP net reduction: sanity-check that reinsurance doesn't produce impossible results.
    #    Lower bound is 0%: after Step 1.5 calibration, the 1-in-100 gross PML (~58M) sits
    #    below the 60M tower attachment, so 0% reduction is physically correct (no losses
    #    reach Layer 1).  Tower attachments (60/100/150M) are illustrative v2 values and will
    #    be re-anchored to OEP return periods in Phase 4 (Paso 4.1).
    #    Upper bound flags implausible over-recovery (net < 30% of gross is a model error).
    r100 = (pml[(100, "oep_g")] - pml[(100, "oep_n")]) / pml[(100, "oep_g")] * 100
    r250 = (pml[(250, "oep_g")] - pml[(250, "oep_n")]) / pml[(250, "oep_g")] * 100
    assert 0.0 <= r100 <= 70.0, \
        f"OEP 1-in-100 reduction {r100:.1f}% outside plausible range 0-70%"
    assert 0.0 <= r250 <= 75.0, \
        f"OEP 1-in-250 reduction {r250:.1f}% outside plausible range 0-75%"
    print(f"[OK] OEP net reduction {r100:.1f}% (1-in-100) and {r250:.1f}% (1-in-250) "
          f"-- consistent with step 5")


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------
def _print_table(aal, pml, N_YEARS):
    COLS = ["aep_g", "aep_n", "oep_g", "oep_n"]
    HDRS = ["AEP Gross", "AEP Net", "OEP Gross", "OEP Net"]
    W    = 11

    print()
    print("=" * 64)
    print("PORTFOLIO METRICS  --  Florida Coastal Homeowners")
    print(f"  {N_YEARS:,} simulated years  |  ep_utils convention (p_k = k/N)")
    print("=" * 64)

    hdr = f"{'Metric':<22}" + "".join(f"{h:>{W}}" for h in HDRS)
    print()
    print(hdr)
    print("-" * len(hdr))

    # AAL row
    print(f"{'AAL (USD M)':<22}" + "".join(
        f"{aal[c] / 1e6:>{W}.2f}" for c in COLS
    ))

    # PML rows
    for rp in RETURN_PERIODS:
        print(f"{'PML 1-in-' + str(rp) + ' (USD M)':<22}" + "".join(
            f"{pml[(rp, c)] / 1e6:>{W}.1f}" for c in COLS
        ))

    print()
    print("Net reduction (gross -> net):")
    for rp in RETURN_PERIODS:
        red_aep = (pml[(rp, "aep_g")] - pml[(rp, "aep_n")]) / pml[(rp, "aep_g")] * 100
        red_oep = (pml[(rp, "oep_g")] - pml[(rp, "oep_n")]) / pml[(rp, "oep_g")] * 100
        print(f"  1-in-{rp:<5}"
              f"AEP: {pml[(rp, 'aep_g')]/1e6:.1f}M -> {pml[(rp, 'aep_n')]/1e6:.1f}M"
              f" (-{red_aep:.1f}%)"
              f"   OEP: {pml[(rp, 'oep_g')]/1e6:.1f}M -> {pml[(rp, 'oep_n')]/1e6:.1f}M"
              f" (-{red_oep:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# Save summary_metrics.csv
# ---------------------------------------------------------------------------
def _save_csv(aal, pml, results_dir):
    COLS = ["aep_g", "aep_n", "oep_g", "oep_n"]
    COL_NAMES = ["aep_gross_M", "aep_net_M", "oep_gross_M", "oep_net_M"]

    def _red(rp, gross_k, net_k):
        g = pml[(rp, gross_k)]
        n = pml[(rp, net_k)]
        return round((g - n) / g * 100, 1) if g > 0 else None

    rows = [
        {"metric": "AAL_M",
         **{cn: round(aal[ck] / 1e6, 3) for ck, cn in zip(COLS, COL_NAMES)}},
    ]
    for rp in RETURN_PERIODS:
        rows.append({
            "metric": f"PML_1in{rp}_M",
            **{cn: round(pml[(rp, ck)] / 1e6, 2) for ck, cn in zip(COLS, COL_NAMES)},
        })
    for rp in RETURN_PERIODS:
        rows.append({
            "metric": f"Reduction_1in{rp}_pct",
            "aep_gross_M": None,
            "aep_net_M":   _red(rp, "aep_g", "aep_n"),
            "oep_gross_M": None,
            "oep_net_M":   _red(rp, "oep_g", "oep_n"),
        })

    df = pd.DataFrame(rows, columns=["metric"] + COL_NAMES)
    path = os.path.join(results_dir, "summary_metrics.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Master EP plot (2-panel AEP / OEP)
# ---------------------------------------------------------------------------
def _plot_ep_master(series, aal, pml, N_YEARS, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    panels = [
        (ax1, "AEP", series["aep_g"], series["aep_n"]),
        (ax2, "OEP", series["oep_g"], series["oep_n"]),
    ]

    for ax, label, gross_arr, net_arr in panels:
        gross_desc, ep = ep_curve(gross_arr, N_YEARS)
        net_desc,   _  = ep_curve(net_arr,   N_YEARS)

        ax.plot(gross_desc / 1e6, ep, color=_C_GROSS, lw=2.2,
                label=f"{label} Gross")
        ax.plot(net_desc   / 1e6, ep, color=_C_NET,   lw=2.2,
                label=f"{label} Net (post-XoL)")
        ax.fill_betweenx(ep, net_desc / 1e6, gross_desc / 1e6,
                         alpha=0.10, color=_C_FILL, label="XoL recovery band")

        # Return-period guidelines + PML annotations
        key_g = "aep_g" if label == "AEP" else "oep_g"
        key_n = "aep_n" if label == "AEP" else "oep_n"

        for rp in RETURN_PERIODS:
            p = 1.0 / rp
            ax.axhline(p, color="grey", linestyle=":", lw=0.9, alpha=0.6)

            pg = pml[(rp, key_g)] / 1e6
            pn = pml[(rp, key_n)] / 1e6
            red = (pg - pn) / pg * 100

            # Dots at intersections
            ax.scatter([pg], [p], color=_C_GROSS, s=45, zorder=6)
            ax.scatter([pn], [p], color=_C_NET,   s=45, zorder=6)

            # Text label: return period on Y axis (left margin)
            ax.text(0.5, p * 1.35, f"1-in-{rp}",
                    fontsize=8, color="dimgrey", va="bottom")

            # PML callout box
            box_x = max(pg, pn) + 2.0
            ax.annotate(
                f"G={pg:.1f}M  N={pn:.1f}M  (-{red:.0f}%)",
                xy=(pg, p), xytext=(box_x, p),
                fontsize=7.5, color="black", va="center",
                arrowprops=dict(arrowstyle="-", color="grey",
                                lw=0.6, relpos=(0, 0.5)),
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="grey", alpha=0.85),
            )

        ax.set_yscale("log")
        ax.set_ylim(5e-5, 1.2)
        ax.set_xlim(left=0)
        ax.set_ylabel("Annual exceedance probability", fontsize=10)
        ax.set_title(
            f"{label} -- Annual {'Aggregate' if label == 'AEP' else 'Max-Occurrence'} "
            f"Loss: Gross vs Net of XoL",
            fontsize=11,
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, which="both", linestyle=":", alpha=0.35)

        # Secondary right Y: return period
        ax_r = ax.twinx()
        ax_r.set_yscale("log")
        ax_r.set_ylim(ax.get_ylim())
        rp_ticks  = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 10000, 20000]
        ep_ticks  = [1.0 / rp for rp in rp_ticks if 5e-5 <= 1.0 / rp <= 1.2]
        rp_labels = [str(rp) for rp in rp_ticks if 5e-5 <= 1.0 / rp <= 1.2]
        ax_r.set_yticks(ep_ticks)
        ax_r.set_yticklabels(rp_labels, fontsize=8)
        ax_r.set_ylabel("Return period (years)", fontsize=9)

    ax2.set_xlabel("Annual portfolio loss (USD million)", fontsize=11)

    fig.suptitle(
        "EP Curves -- Florida Coastal Homeowners\n"
        f"TIV USD 500M  |  {N_YEARS:,} simulated years  |  "
        "XoL tower 60M-200M (140M capacity)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    path = os.path.join(out_dir, "ep_master.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print()
    print("=" * 64)
    print("SUMMARY  (Step 6)")
    print("=" * 64)

    annual_df, N_YEARS = _load(RESULTS_DIR)
    print(f"Loaded: {N_YEARS:,} simulated years  |  "
          f"{int((annual_df['n_events'] > 0).sum()):,} event years")

    series, aal, pml = _compute_metrics(annual_df, N_YEARS)
    _validate(aal, pml)
    _print_table(aal, pml, N_YEARS)
    _save_csv(aal, pml, RESULTS_DIR)
    _plot_ep_master(series, aal, pml, N_YEARS, OUT_DIR)

    print()
    print("=" * 64)
    print("Summary complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
