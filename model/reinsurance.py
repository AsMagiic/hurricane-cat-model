"""
Reinsurance engine for the Florida hurricane cat model (Step 5).

Applies a per-occurrence multi-layer XoL programme to the gross portfolio losses
from Step 4 (loss.py) and derives the net (post-reinsurance) loss series.

Three-layer tower (occurrence XoL, contiguous):
  Layer 1  : 40M xs  60M   covers  60M - 100M per occurrence
  Layer 2  : 50M xs 100M   covers 100M - 150M per occurrence
  Layer 3  : 50M xs 150M   covers 150M - 200M per occurrence

Recovery per event (vectorized over portfolio_gross):
  rec_layer = clip(gross - attachment, 0, layer_limit)
  net       = gross - sum(rec_layer for all layers)

For a contiguous tower from attachment_0 to exhaustion, the net is always equal
to min(gross, attachment_0) for losses within the tower:
  gross in [60M, 200M]  ->  net = 60M  (retention is capped at first attachment)
  gross > 200M          ->  net = gross - 140M  (above exhaustion)
  gross < 60M           ->  net = gross         (below retention)

Note: reinstatement premiums are NOT modelled here (future extension).

Inputs  (from results/  -- produced by loss.py):
  events.csv        one row per event
  annual_losses.csv N_YEARS rows, provides the full year list + max_event_gross

Outputs:
  results/events_net.csv      events enriched with per-layer recovery and net
  outputs/ep_gross_vs_net.png OEP gross vs net comparison plot
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from model_config import load_model_cfg
_mcfg = load_model_cfg()

from model.ep_utils import oep_pml, ep_curve, pml_rank_diagnostic

RESULTS_DIR = os.path.join(_ROOT, "results")
OUT_DIR     = os.path.join(_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# XoL programme definition -- loaded from config/model_v3.yaml
# ---------------------------------------------------------------------------
LAYERS = _mcfg.reinsurance.layers

TOTAL_CAPACITY = sum(lyr["limit"]  for lyr in LAYERS)            # 140M
TOP_OF_TOWER   = LAYERS[-1]["attachment"] + LAYERS[-1]["limit"]  # 200M
FIRST_ATTACH   = LAYERS[0]["attachment"]                          #  60M

# ---------------------------------------------------------------------------
# Layer contiguity validation
# ---------------------------------------------------------------------------
def _check_contiguity(layers):
    for i in range(len(layers) - 1):
        expected = layers[i]["attachment"] + layers[i]["limit"]
        actual   = layers[i + 1]["attachment"]
        if actual != expected:
            delta = actual - expected
            kind  = "gap" if delta > 0 else "overlap"
            print(f"  [WARN] {kind} of {abs(delta)/1e6:.0f}M between "
                  f"'{layers[i]['name']}' (top {expected/1e6:.0f}M) and "
                  f"'{layers[i+1]['name']}' (attach {actual/1e6:.0f}M)")
            return False
    print(f"[OK] Tower contiguous: {FIRST_ATTACH/1e6:.0f}M to "
          f"{TOP_OF_TOWER/1e6:.0f}M  ({TOTAL_CAPACITY/1e6:.0f}M total capacity)")
    return True

# ---------------------------------------------------------------------------
# Vectorized recovery kernel
# ---------------------------------------------------------------------------
def apply_xol(gross_arr):
    """
    Apply the XoL tower to a (N,) array of gross per-occurrence losses.

    Returns
    -------
    rec_matrix : ndarray (N, n_layers)  per-layer recovery (USD)
    rec_total  : ndarray (N,)           sum across all layers (USD)
    net        : ndarray (N,)           gross - rec_total  (USD)
    """
    gross      = np.asarray(gross_arr, dtype=float)
    rec_matrix = np.zeros((len(gross), len(LAYERS)))
    for j, lyr in enumerate(LAYERS):
        rec_matrix[:, j] = np.clip(gross - lyr["attachment"], 0.0, lyr["limit"])
    rec_total = rec_matrix.sum(axis=1)
    return rec_matrix, rec_total, gross - rec_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ---- Load Step 4 outputs -----------------------------------------------
    ev_path  = os.path.join(RESULTS_DIR, "events.csv")
    ann_path = os.path.join(RESULTS_DIR, "annual_losses.csv")

    for p in (ev_path, ann_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found.  Run model/loss.py first.")

    events_df = pd.read_csv(ev_path)
    annual_df = pd.read_csv(ann_path)

    N_YEARS  = len(annual_df)
    N_EVENTS = len(events_df)
    print(f"Loaded: {N_EVENTS:,} events | {N_YEARS:,} years")
    print()

    # ---- Check programme contiguity ----------------------------------------
    ok = _check_contiguity(LAYERS)
    assert ok, "Layer contiguity check failed -- fix LAYERS before continuing"

    # ---- Apply XoL tower (vectorized over all events) ---------------------
    gross_arr                      = events_df["portfolio_gross"].to_numpy()
    rec_matrix, rec_total, net_arr = apply_xol(gross_arr)

    # Enrich events DataFrame
    for j, lyr in enumerate(LAYERS):
        col = "rec_" + lyr["name"].replace(" ", "_")
        events_df[col] = rec_matrix[:, j]

    events_df["recovery_total"] = rec_total
    events_df["portfolio_net"]  = net_arr

    # ---- Build annual OEP net series (N_YEARS rows, 0 for no-event years) --
    if N_EVENTS > 0:
        yr_max_net = (events_df
                      .groupby("year")["portfolio_net"]
                      .max())
    else:
        yr_max_net = pd.Series(dtype=float)

    annual_df["max_event_net"] = (
        annual_df["year"].map(yr_max_net).fillna(0.0)
    )

    # ---- Validation asserts -----------------------------------------------
    print()
    print("=" * 64)
    print("VALIDATION")
    print("=" * 64)

    # 1. net <= gross for all events
    assert (net_arr <= gross_arr + 1e-6).all(), \
        "FAIL: portfolio_net > portfolio_gross for some event"
    print("[OK] net <= gross for all events")

    # 2. recovery_total in [0, TOTAL_CAPACITY]
    assert (rec_total >= -1e-6).all() and \
           (rec_total <= TOTAL_CAPACITY + 1e-6).all(), \
        f"FAIL: recovery_total outside [0, {TOTAL_CAPACITY/1e6:.0f}M]"
    print(f"[OK] recovery_total in [0, {TOTAL_CAPACITY/1e6:.0f}M] for all events")

    # 3. recovery_total == 0 when gross <= first attachment
    below = gross_arr <= FIRST_ATTACH
    assert (rec_total[below] < 1e-6).all(), \
        f"FAIL: recovery > 0 for event with gross <= {FIRST_ATTACH/1e6:.0f}M"
    print(f"[OK] recovery = 0 for all {int(below.sum()):,} events "
          f"with gross <= {FIRST_ATTACH/1e6:.0f}M")

    # 4. recovery_total == TOTAL_CAPACITY when gross >= top of tower
    above = gross_arr >= TOP_OF_TOWER
    if above.any():
        assert (np.abs(rec_total[above] - TOTAL_CAPACITY) < 1e-4).all(), \
            "FAIL: tower not exhausted when gross >= top of programme"
        print(f"[OK] Full exhaustion ({TOTAL_CAPACITY/1e6:.0f}M) for "
              f"{int(above.sum()):,} events with gross >= {TOP_OF_TOWER/1e6:.0f}M")
    else:
        max_gross = gross_arr.max() if N_EVENTS > 0 else 0.0
        print(f"[OK] No events exceeded programme top ({TOP_OF_TOWER/1e6:.0f}M) "
              f"-- max gross was {max_gross/1e6:.1f}M; exhaustion assert not applicable")

    # 5. net PML <= gross PML at 1-in-100 and 1-in-250
    oep_gross_arr = annual_df["max_event_gross"].to_numpy()
    oep_net_arr   = annual_df["max_event_net"].to_numpy()
    oep_g, ep = ep_curve(oep_gross_arr, N_YEARS)   # sorted desc + ep; reused for plot
    oep_n, _  = ep_curve(oep_net_arr,   N_YEARS)
    pml_g100  = oep_pml(oep_gross_arr, 100, N_YEARS)
    pml_g250  = oep_pml(oep_gross_arr, 250, N_YEARS)
    pml_n100  = oep_pml(oep_net_arr,   100, N_YEARS)
    pml_n250  = oep_pml(oep_net_arr,   250, N_YEARS)
    print()
    pml_rank_diagnostic(oep_g, N_YEARS)   # raw rank sanity check

    assert pml_n100 <= pml_g100 + 1e-6, \
        f"FAIL: net OEP 1-in-100 ({pml_n100/1e6:.1f}M) > gross ({pml_g100/1e6:.1f}M)"
    assert pml_n250 <= pml_g250 + 1e-6, \
        f"FAIL: net OEP 1-in-250 ({pml_n250/1e6:.1f}M) > gross ({pml_g250/1e6:.1f}M)"
    print("[OK] net OEP PML <= gross OEP PML at 1-in-100 and 1-in-250")

    # ---- Programme metrics ------------------------------------------------
    print()
    print("=" * 64)
    print("PROGRAMME METRICS")
    print("=" * 64)

    # Annual recovery totals (0 for no-event years) using map
    if N_EVENTS > 0:
        yr_rec_total = events_df.groupby("year")["recovery_total"].sum()
    else:
        yr_rec_total = pd.Series(dtype=float)

    ann_rec_total = annual_df["year"].map(yr_rec_total).fillna(0.0)
    aal_rec = float(ann_rec_total.mean())

    print(f"\nAAL of total recovery : USD {aal_rec:>14,.0f}  "
          f"(technical premium floor, excl. loadings + risk margin)")

    print(f"\nPer-layer breakdown:")
    print(f"  {'Layer':<12} {'Attach':>8} {'Limit':>8} "
          f"{'AAL rec ($M)':>14} {'Freq (% years)':>15}")
    print("  " + "-" * 60)

    for lyr in LAYERS:
        col = "rec_" + lyr["name"].replace(" ", "_")
        if N_EVENTS > 0:
            yr_lyr_max = events_df.groupby("year")[col].max()
            yr_lyr_sum = events_df.groupby("year")[col].sum()
            ann_lyr_max = annual_df["year"].map(yr_lyr_max).fillna(0.0)
            ann_lyr_sum = annual_df["year"].map(yr_lyr_sum).fillna(0.0)
            freq_pct = float((ann_lyr_max > 1e-6).mean()) * 100
            aal_lyr  = float(ann_lyr_sum.mean())
        else:
            freq_pct = aal_lyr = 0.0
        print(f"  {lyr['name']:<12} {lyr['attachment']/1e6:>6.0f}M "
              f"{lyr['limit']/1e6:>6.0f}M "
              f"{aal_lyr/1e6:>14.3f} "
              f"{freq_pct:>14.2f}%")

    print(f"\nOEP PML reduction (gross -> net, per occurrence):")
    print(f"  {'Return period':<14} {'Gross ($M)':>12} {'Net ($M)':>12} "
          f"{'Reduction ($M)':>16} {'Reduction %':>12}")
    print("  " + "-" * 68)
    for rp, g, n in [(100, pml_g100, pml_n100), (250, pml_g250, pml_n250)]:
        red_abs = g - n
        red_pct = red_abs / g * 100 if g > 0 else 0.0
        print(f"  1-in-{rp:<9}  {g/1e6:>10.1f}  {n/1e6:>10.1f}  "
              f"{red_abs/1e6:>14.1f}  {red_pct:>11.1f}%")

    # ---- OEP plot ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Gross and net OEP curves
    ax.plot(oep_g / 1e6, ep, color="#1f77b4", linewidth=2.2,
            label="OEP Gross (post-policy-deductible)")
    ax.plot(oep_n / 1e6, ep, color="#d62728", linewidth=2.2,
            label="OEP Net (post-XoL reinsurance)")

    # Shaded recovery band
    ax.fill_betweenx(ep, oep_n / 1e6, oep_g / 1e6,
                     alpha=0.10, color="#2ca02c", label="XoL recovery region")

    # Vertical lines at layer attachment points
    lyr_colors = ["#ff7f0e", "#9467bd", "#8c564b"]
    for lyr, col in zip(LAYERS, lyr_colors):
        att = lyr["attachment"] / 1e6
        ax.axvline(att, color=col, linestyle="--", linewidth=1.0, alpha=0.75)
        ax.text(att + 0.8, 6e-4,
                f"{lyr['name']}\n{att:.0f}M xs",
                fontsize=7.5, color=col, va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.8))

    # Horizontal return-period lines
    for rp, g, n in [(100, pml_g100, pml_n100), (250, pml_g250, pml_n250)]:
        p = 1 / rp
        ax.axhline(p, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.text(0.4, p * 1.25, f"1-in-{rp}", fontsize=7.5, color="grey", va="bottom")
        # Intersection dots
        ax.scatter([g / 1e6], [p], color="#1f77b4", s=40, zorder=6)
        ax.scatter([n / 1e6], [p], color="#d62728", s=40, zorder=6)

    # PML summary box (lower-right)
    pml_text = (
        f"OEP PML (USD M)\n"
        f"           Gross    Net\n"
        f"1-in-100:  {pml_g100/1e6:>5.1f}   {pml_n100/1e6:>5.1f}\n"
        f"1-in-250:  {pml_g250/1e6:>5.1f}   {pml_n250/1e6:>5.1f}"
    )
    ax.text(0.98, 0.04, pml_text, transform=ax.transAxes,
            fontsize=8.5, va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.9))

    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.set_xlim(left=0)
    ax.set_xlabel("Per-occurrence portfolio loss (USD million)", fontsize=11)
    ax.set_ylabel("Annual exceedance probability", fontsize=11)
    ax.set_title(
        "OEP: Gross vs Net of XoL reinsurance  --  Florida coastal homeowners\n"
        f"Programme: {FIRST_ATTACH/1e6:.0f}M to {TOP_OF_TOWER/1e6:.0f}M "
        f"({TOTAL_CAPACITY/1e6:.0f}M total capacity)  |  {N_YEARS:,} simulated years",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # Secondary right Y-axis: return period
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(ax.get_ylim())
    rp_ticks  = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 10000]
    ep_ticks  = [1 / rp for rp in rp_ticks if 1e-4 <= 1 / rp <= 1.0]
    rp_labels = [str(rp) for rp in rp_ticks if 1e-4 <= 1 / rp <= 1.0]
    ax2.set_yticks(ep_ticks)
    ax2.set_yticklabels(rp_labels)
    ax2.set_ylabel("Return period (years)", fontsize=10)

    p_fig = os.path.join(OUT_DIR, "ep_gross_vs_net.png")
    fig.savefig(p_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved -> {p_fig}")

    # ---- Save events_net.csv ---------------------------------------------
    out_ev = os.path.join(RESULTS_DIR, "events_net.csv")
    events_df.to_csv(out_ev, index=False)
    print(f"Saved: {out_ev}  ({len(events_df):,} rows)")
    print(f"  Columns: {list(events_df.columns)}")

    print()
    print("=" * 64)
    print("All structural asserts passed.")
    print("=" * 64)


if __name__ == "__main__":
    main()
