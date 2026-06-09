"""
AEP and OEP curves for the Florida hurricane cat model.

Starting from the same 100,000-year simulation used in aggregate_loss.py,
this script:
  1. Builds the empirical AEP curve (aggregate annual loss per year).
  2. Builds the empirical OEP curve (maximum single-event loss per year).
  3. Reads off the PML at the 1-in-100 and 1-in-250 return periods for both.
  4. Saves a combined plot to outputs/.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# ---------------------------------------------------------------------------
# Parameters — must match aggregate_loss.py exactly to reproduce the same S
# ---------------------------------------------------------------------------
LAMBDA = 0.7
MU     = 15.93423326
SIGMA  =  1.085658784
M      = 100_000
SEED   = 42

# ---------------------------------------------------------------------------
# Reproduce the 100,000 annual aggregate losses (identical to aggregate_loss.py)
# ---------------------------------------------------------------------------
np.random.seed(SEED)

event_counts   = np.random.poisson(lam=LAMBDA, size=M)
total_events   = int(event_counts.sum())
all_severities = np.random.lognormal(mean=MU, sigma=SIGMA, size=total_events)

split_indices = np.cumsum(event_counts[:-1])
groups        = np.split(all_severities, split_indices)

# AEP source: sum of all event losses within the year.
annual_losses = np.array([g.sum() for g in groups])

# OEP source: loss of the single largest event within the year.
# Years with no events (empty group) get a maximum of 0.
annual_max_losses = np.array([g.max() if g.size > 0 else 0.0 for g in groups])

# ---------------------------------------------------------------------------
# Build the AEP curve
#
# The empirical AEP assigns to the k-th largest loss (k = 1, 2, …, M)
# an exceedance probability of  EP(k) = k / M.
#
# Intuition: in M years, the largest loss was exceeded 1 time → EP = 1/M.
#            The second-largest was exceeded 2 times → EP = 2/M, and so on.
# ---------------------------------------------------------------------------
sorted_losses = np.sort(annual_losses)[::-1]          # descending: rank 1 = largest
ep            = np.arange(1, M + 1) / M               # EP[k-1] = k / M  (increasing)

# ---------------------------------------------------------------------------
# Read off PML values using linear interpolation on the EP curve.
#
# np.interp(x, xp, fp) requires xp to be increasing.
# Here ep is already increasing (1/M … 1) and sorted_losses is the
# corresponding y-values (decreasing).  np.interp handles that fine.
#
# Return period T  →  exceedance probability 1/T
# ---------------------------------------------------------------------------
aep_pml_100 = np.interp(1 / 100, ep, sorted_losses)   # 1-in-100-year AEP loss
aep_pml_250 = np.interp(1 / 250, ep, sorted_losses)   # 1-in-250-year AEP loss
aal         = annual_losses.mean()

# ---------------------------------------------------------------------------
# Build the OEP curve
#
# Same ranking logic as AEP, but applied to the per-year maximum event loss.
# OEP answers: "what is the probability that at least one event in a year
# exceeds loss X?" — it ignores accumulation within the year.
# OEP ≤ AEP always, because the max event ≤ the aggregate.
# ---------------------------------------------------------------------------
sorted_max_losses = np.sort(annual_max_losses)[::-1]   # descending

oep_pml_100 = np.interp(1 / 100, ep, sorted_max_losses)
oep_pml_250 = np.interp(1 / 250, ep, sorted_max_losses)

print("=== EP Curves — Key Metrics ===")
print(f"{'Metric':<35} {'AEP':>15}  {'OEP':>15}")
print("-" * 68)
print(f"{'AAL (simulated)':<35} USD {aal:>12,.0f}")
print(f"{'PML 1-in-100yr  (EP = 1.00%)':<35} USD {aep_pml_100:>12,.0f}  USD {oep_pml_100:>12,.0f}")
print(f"{'PML 1-in-250yr  (EP = 0.40%)':<35} USD {aep_pml_250:>12,.0f}  USD {oep_pml_250:>12,.0f}")

# ---------------------------------------------------------------------------
# Plot AEP and OEP together
#
# Convention in cat modeling:
#   X axis — loss amount (USD); AEP uses aggregate, OEP uses max-event
#   Y axis — exceedance probability (log scale, to show the tail clearly)
#
# Zero-loss years are excluded from both curves because log(0) is undefined.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# --- AEP curve ---
aep_mask = sorted_losses > 0
ax.plot(
    sorted_losses[aep_mask], ep[aep_mask],
    color="#1f77b4", linewidth=1.5, label="AEP (aggregate annual loss)",
)

# --- OEP curve ---
oep_mask = sorted_max_losses > 0
ax.plot(
    sorted_max_losses[oep_mask], ep[oep_mask],
    color="#2ca02c", linewidth=1.5, linestyle="--",
    label="OEP (max single-event loss)",
)

# --- Mark the four PML points ---
# Offsets are chosen so each label sits in a distinct quadrant:
#   AEP labels → upper-right  (curve sits to the right at each RP)
#   OEP labels → lower-left   (curve sits to the left at each RP)
# Within each curve, 1-in-100 goes up and 1-in-250 goes down, keeping
# the two RP labels separated even though they share the same X region.
pml_points = [
    #  rp   pml           color      label   (dx,  dy)
    (100, aep_pml_100, "#1f77b4", "AEP", ( 60,  18)),
    (250, aep_pml_250, "#1f4e7a", "AEP", ( 60, -32)),
    (100, oep_pml_100, "#2ca02c", "OEP", (-95,  18)),
    (250, oep_pml_250, "#1a6b1a", "OEP", (-95, -32)),
]
for rp, pml, color, label, (dx, dy) in pml_points:
    ax.scatter(pml, 1 / rp, zorder=5, color=color, s=60)
    ax.annotate(
        f"{label} 1-in-{rp}\n${pml/1e6:.1f}M",
        xy=(pml, 1 / rp),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8,
        color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.8, shrinkA=0, shrinkB=3),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
    )

# Shared return-period reference lines (one per RP, colour-neutral)
for rp, ls in [(100, "--"), (250, ":")]:
    ax.axhline(1 / rp, color="grey", linestyle=ls, linewidth=0.7, alpha=0.5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Loss (USD)", fontsize=11)
ax.set_ylabel("Exceedance Probability", fontsize=11)
ax.set_title(
    "AEP & OEP Curves — Florida Homeowners Hurricane Portfolio\n"
    "TIV USD 500M | 100,000 simulated years",
    fontsize=12,
)

# :.4g uses 4 significant figures: avoids "1e+03" for $1000M while still
# showing decimals for small values (e.g. 0.1 → "$0.1M").
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"${x/1e6:.4g}M")
)
# :.3g keeps enough precision so tiny EP values don't collapse to "0.00%".
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda y, _: f"{y*100:.3g}%")
)

ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
ax.legend(fontsize=10)
fig.tight_layout()

# ---------------------------------------------------------------------------
# Save to outputs/
# ---------------------------------------------------------------------------
output_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs", "ep_curves.png"
)
fig.savefig(output_path, dpi=150)
print(f"\nPlot saved to: {output_path}")
