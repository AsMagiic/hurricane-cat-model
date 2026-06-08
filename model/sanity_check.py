"""
Sanity check: validate the frequency component of the hurricane cat model.

Before building severity or the EP curve, we confirm that the Poisson
frequency generator behaves as expected. The simulated mean should
converge to the theoretical lambda (λ = 0.7 events/year).
"""

import numpy as np

# --- Parameters ---
LAMBDA = 0.7       # Expected number of hurricane events per year
M = 100_000        # Number of simulated years
SEED = 42          # Fixed seed for reproducibility

# --- Simulate annual event counts ---
# Each entry is the number of hurricane events that hit the portfolio in one year.
# Poisson(λ) is the standard choice for rare-event frequency in cat models.
rng = np.random.default_rng(SEED)
event_counts = rng.poisson(lam=LAMBDA, size=M)

# --- Compute the simulated mean frequency ---
simulated_mean = event_counts.mean()

# --- Compare against the theoretical value ---
# By the law of large numbers, the simulated mean should converge to λ
# as M grows. A value close to 0.7 confirms the generator is working correctly.
print(f"Theoretical lambda:  {LAMBDA:.4f} events/year")
print(f"Simulated mean:      {simulated_mean:.4f} events/year  (over {M:,} years)")
print(f"Difference:          {abs(simulated_mean - LAMBDA):.6f}")
