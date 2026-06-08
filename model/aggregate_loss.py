"""
Aggregate annual loss simulation for the Florida hurricane cat model.

For each of the 100,000 simulated years we:
  1. Draw the number of events N from Poisson(lambda=0.7).
  2. Draw N individual losses from Lognormal(mu, sigma).
  3. Sum them to get the annual aggregate loss S.

Validation target: E[S] = lambda * E[X] = 0.7 * 15,000,000 = USD 10,500,000/year.
"""

import numpy as np

# --- Parameters ---
LAMBDA = 0.7              # Poisson rate: expected hurricane events per year
MU    = 15.93423326       # Lognormal location parameter (log-space mean)
SIGMA =  1.085658784      # Lognormal scale parameter  (log-space std dev)
M     = 100_000           # Simulated years
SEED  = 42

# Theoretical AAL: E[S] = lambda * E[X], where E[X] = exp(mu + sigma^2 / 2)
THEORETICAL_AAL = 10_500_000

# Fix the global random seed so results are reproducible across runs.
np.random.seed(SEED)

# --- Step 1: simulate annual event counts ---
# Each element is the number of hurricane events in that year.
event_counts = np.random.poisson(lam=LAMBDA, size=M)

# --- Step 2: simulate all individual event severities in one vectorised call ---
# Generating all severities at once is far faster than looping year by year.
# total_events is the sum of all N_i across the 100k years.
total_events = int(event_counts.sum())
all_severities = np.random.lognormal(mean=MU, sigma=SIGMA, size=total_events)

# --- Step 3: aggregate severities into annual losses ---
# np.split needs the cumulative event counts as cut-points (excluding the last).
# Each slice all_severities[cut_i : cut_{i+1}] corresponds to year i.
split_indices = np.cumsum(event_counts[:-1])
annual_losses = np.array(
    [group.sum() for group in np.split(all_severities, split_indices)]
)
# Years with zero events contribute a sum of 0 automatically (empty slice).

# --- Validation 1: simulated AAL vs theoretical ---
simulated_aal = annual_losses.mean()
aal_error_pct = abs(simulated_aal - THEORETICAL_AAL) / THEORETICAL_AAL * 100

print("=== Annual Aggregate Loss — Simulation Results ===")
print(f"Theoretical AAL:  USD {THEORETICAL_AAL:>15,.0f}")
print(f"Simulated AAL:    USD {simulated_aal:>15,.0f}")
print(f"Relative error:   {aal_error_pct:.3f}%")

# --- Validation 2: zero-loss years ---
# A year has zero loss when no events occur (N_i = 0).
# Theoretically P(N=0) = exp(-lambda) = exp(-0.7) ~ 49.66 %.
zero_loss_years = int((annual_losses == 0).sum())
zero_loss_pct   = zero_loss_years / M * 100
theoretical_zero_pct = np.exp(-LAMBDA) * 100

print(f"\nZero-loss years (simulated):   {zero_loss_years:>6,}  ({zero_loss_pct:.2f}%)")
print(f"Zero-loss years (theoretical): {theoretical_zero_pct:.2f}%")
