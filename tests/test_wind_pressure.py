"""
Unit tests for calibration/wind_pressure.py.

Focus: predict_dp and fit_wpr as pure functions (no file I/O).

The real-b comparison (b_fit vs gradient-wind expectation 2.0) is a
runtime PRINT in run_calibration(), reviewed by the operator — NOT an
assert. test_b_sane_band verifies that fit_wpr recovers a known b from
clean synthetic data; it says nothing about the real WPR b.
"""

import numpy as np
import pytest

from calibration.wind_pressure import KT_TO_MPH, fit_wpr, predict_dp


class TestPredictDp:

    def test_unit_chain_exact(self):
        """predict_dp(100, a=0.5, b=2) == 0.5 * 100^2 == 5000.0 (exact float64)."""
        assert predict_dp(100.0, a=0.5, b=2.0) == 5000.0

    def test_positive(self):
        """predict_dp > 0 for any positive vmax_mph input."""
        v  = np.array([74.0, 100.0, 130.0, 160.0])
        dp = predict_dp(v, a=0.3, b=1.9)
        assert (dp > 0).all()

    def test_monotone_increasing(self):
        """For b > 0, predict_dp is strictly increasing with vmax_mph."""
        v  = np.linspace(74.0, 160.0, 20)
        dp = predict_dp(v, a=0.5, b=2.0)
        assert (np.diff(dp) > 0).all()


class TestFitWpr:
    """Synthetic-data tests. Fixed seed=0 for reproducibility."""

    @staticmethod
    def _synthetic(n: int = 200, a: float = 0.5, b: float = 2.0,
                   sigma: float = 0.05, seed: int = 0):
        rng = np.random.default_rng(seed)
        v   = rng.uniform(74.0, 160.0, n)
        dp  = a * v ** b * np.exp(rng.normal(0.0, sigma, n))
        return v, dp

    def test_b_sane_band(self):
        """
        fit_wpr recovers b=2.0 within ±0.5 from clean synthetic data.

        This is a pure unit test of the OLS estimator — it does NOT
        validate the real WPR b. The real-b check is the printed
        'b_fit vs 2.0' line in run_calibration(), reviewed by the operator.
        """
        v, dp = self._synthetic(b=2.0)
        _, b_fit, _, _ = fit_wpr(v, dp)
        assert 1.5 <= b_fit <= 2.5, (
            f"fit_wpr failed to recover b=2.0 from clean synthetic data; "
            f"got b={b_fit:.4f}. Out-of-band on REAL data is information, "
            "not necessarily a test failure."
        )

    def test_r_squared_noiseless(self):
        """On a noiseless power law, R² must be >= 0.999."""
        v  = np.linspace(74.0, 160.0, 100)
        dp = 0.5 * v ** 2.0
        _, _, _, r2 = fit_wpr(v, dp)
        assert r2 >= 0.999, f"Expected R²>=0.999 on noiseless data; got {r2:.6f}"

    def test_roundtrip_within_sigma(self):
        """
        >= 90% of observed pairs round-trip within exp(2·sigma_log) of fit.

        90% assertion threshold; population value ≈ 95.4% for normal residuals.
        Verifies that sigma_log correctly characterises log-space scatter.
        """
        v, dp = self._synthetic(sigma=0.05)
        a_fit, b_fit, sigma_log, _ = fit_wpr(v, dp)
        log_resid = np.abs(np.log(dp) - np.log(predict_dp(v, a_fit, b_fit)))
        in_band   = log_resid <= 2.0 * sigma_log
        assert in_band.mean() >= 0.90, (
            f"Only {in_band.mean() * 100:.1f}% of points within 2·sigma_log "
            f"(expected >= 90%); sigma_log={sigma_log:.4f}"
        )
